## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 20.6039733455


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050)
1: (-15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170)
2: (-8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106)
3: (-7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108)
4: (-10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.51 + 1.63 = 4.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -20.7075109, upper bound: 20.7075081

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7063131, upper bound: 20.7063127
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7063131, upper bound: 20.7063114
time: 0.61 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.37 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.37
Output dim: 3, lower bound: -20.7063131, upper bound: 20.7063127
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.37
Output dim: 3, lower bound: -20.7063131, upper bound: 20.7063114

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301523
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301521
time: 0.55 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301515
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301515
time: 0.54 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.58 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301523
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301521
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301515
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301515

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301516
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301516
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301538
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301516
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301521
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301507
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301509
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301524
time: 0.65 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.74 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301516
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301516
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301538
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301516
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301521
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301507
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301509
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301524

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301166
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301175
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301187
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301186
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301163
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301155
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301187
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301186
time: 0.61 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.77 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301189
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301166
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301175
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301187
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301186
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301163
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301155
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301187
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 3, lower bound: -20.6301190, upper bound: 20.6301186

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300765
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300741
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300739
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300740
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300759
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300720
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300765
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300755
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300765
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300743
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300741
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300740
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300759
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300720
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300752
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300755
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300756
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300730
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300745
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300746
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300749
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300725
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300760
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300713
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300743
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300769
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300745
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300769
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300770
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300768
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300760
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300770
time: 0.56 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300765
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300741
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300739
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300740
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300759
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300720
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300765
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300755
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300765
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300743
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300741
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300740
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300759
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300720
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300752
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300755
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300756
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300730
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300745
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300746
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300749
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300725
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300760
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300713
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300743
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300769
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300745
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300769
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300770
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300768
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300760
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 3, lower bound: -20.6300766, upper bound: 20.6300770

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
time: 0.58 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.83 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762540, upper bound: 20.3762542
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762543
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -20.3762543, upper bound: 20.3762542

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.15 + 232.92 = 237.06 seconds
