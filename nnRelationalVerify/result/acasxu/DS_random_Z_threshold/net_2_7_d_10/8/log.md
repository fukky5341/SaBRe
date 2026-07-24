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
execution time: IAR + RelationalAnalysis = 0.97 + 1.51 = 2.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -20.7075109, upper bound: 20.7075081

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7063131, upper bound: 20.7063127
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7063131, upper bound: 20.7063114
time: 0.48 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.93 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.93
Output dim: 3, lower bound: -20.7063131, upper bound: 20.7063127
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.93
Output dim: 3, lower bound: -20.7063131, upper bound: 20.7063114

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301523
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301521
time: 0.43 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6265726, upper bound: 20.6265676
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6265726, upper bound: 20.6265672
time: 0.45 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.75 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.75
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301523
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.75
Output dim: 3, lower bound: -20.6301545, upper bound: 20.6301521
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.75
Output dim: 3, lower bound: -20.6265726, upper bound: 20.6265676
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.75
Output dim: 3, lower bound: -20.6265726, upper bound: 20.6265672

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6272116, upper bound: 20.6272116
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6272120, upper bound: 20.6272096
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6272116, upper bound: 20.6272114
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6272120, upper bound: 20.6272070
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6265265, upper bound: 20.6265248
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6265264, upper bound: 20.6265203
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5733407, upper bound: 20.5733366
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5733407, upper bound: 20.5733381
time: 0.41 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.79 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.79
Output dim: 3, lower bound: -20.6272116, upper bound: 20.6272116
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.79
Output dim: 3, lower bound: -20.6272120, upper bound: 20.6272096
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.79
Output dim: 3, lower bound: -20.6272116, upper bound: 20.6272114
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.79
Output dim: 3, lower bound: -20.6272120, upper bound: 20.6272070
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.79
Output dim: 3, lower bound: -20.6265265, upper bound: 20.6265248
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.79
Output dim: 3, lower bound: -20.6265264, upper bound: 20.6265203
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 1.79
Output dim: 3, lower bound: -20.5733407, upper bound: 20.5733366
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.79
Output dim: 3, lower bound: -20.5733407, upper bound: 20.5733381

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6272116, upper bound: 20.6272061
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6272116, upper bound: 20.6272102
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6271694, upper bound: 20.6271684
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6271691, upper bound: 20.6271672
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6271691, upper bound: 20.6271623
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6271691, upper bound: 20.6271678
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6271759, upper bound: 20.6271758
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6271759, upper bound: 20.6271724
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.4209202, upper bound: 20.4209200
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.4209202, upper bound: 20.4209202
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5731594, upper bound: 20.5731562
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5731594, upper bound: 20.5731562
time: 0.48 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.85 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.85
Output dim: 3, lower bound: -20.6272116, upper bound: 20.6272061
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.85
Output dim: 3, lower bound: -20.6272116, upper bound: 20.6272102
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.85
Output dim: 3, lower bound: -20.6271694, upper bound: 20.6271684
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.85
Output dim: 3, lower bound: -20.6271691, upper bound: 20.6271672
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.85
Output dim: 3, lower bound: -20.6271691, upper bound: 20.6271623
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.85
Output dim: 3, lower bound: -20.6271691, upper bound: 20.6271678
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.85
Output dim: 3, lower bound: -20.6271759, upper bound: 20.6271758
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.85
Output dim: 3, lower bound: -20.6271759, upper bound: 20.6271724
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.85
Output dim: 3, lower bound: -20.4209202, upper bound: 20.4209200
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.85
Output dim: 3, lower bound: -20.4209202, upper bound: 20.4209202
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.85
Output dim: 3, lower bound: -20.5731594, upper bound: 20.5731562
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.85
Output dim: 3, lower bound: -20.5731594, upper bound: 20.5731562

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6271759, upper bound: 20.6271722
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6271759, upper bound: 20.6271722
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6101450, upper bound: 20.6101422
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6101450, upper bound: 20.6101448
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6271691, upper bound: 20.6271678
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6271691, upper bound: 20.6271672
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6264957, upper bound: 20.6264952
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6264957, upper bound: 20.6264937
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3992419, upper bound: 20.3992417
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3992419, upper bound: 20.3992419
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6271333, upper bound: 20.6271314
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6271333, upper bound: 20.6271316
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3982386, upper bound: 20.3982385
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3982386, upper bound: 20.3982386
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101082
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101092
time: 0.44 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.78 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.6271759, upper bound: 20.6271722
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.6271759, upper bound: 20.6271722
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.6101450, upper bound: 20.6101422
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.6101450, upper bound: 20.6101448
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.6271691, upper bound: 20.6271678
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.6271691, upper bound: 20.6271672
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.6264957, upper bound: 20.6264952
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.6264957, upper bound: 20.6264937
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.3992419, upper bound: 20.3992417
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.3992419, upper bound: 20.3992419
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.6271333, upper bound: 20.6271314
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.6271333, upper bound: 20.6271316
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.3982386, upper bound: 20.3982385
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.3982386, upper bound: 20.3982386
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101082
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.78
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101092

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101079
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101096
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3982386, upper bound: 20.3982385
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3982386, upper bound: 20.3982385
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101096
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101091
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101095
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101093
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3992419, upper bound: 20.3992417
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3992419, upper bound: 20.3992419
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6264957, upper bound: 20.6264923
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6264957, upper bound: 20.6264923
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3992414, upper bound: 20.3992412
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3992414, upper bound: 20.3992410
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6264957, upper bound: 20.6264936
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6264957, upper bound: 20.6264923
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6264598, upper bound: 20.6264561
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6264598, upper bound: 20.6264576
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6269866, upper bound: 20.6269863
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6269866, upper bound: 20.6269862
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090916, upper bound: 20.6090911
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090916, upper bound: 20.6090914
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101093
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101095
time: 0.39 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.78 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101079
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101096
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.3982386, upper bound: 20.3982385
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.3982386, upper bound: 20.3982385
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101096
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101091
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101095
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101093
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.3992419, upper bound: 20.3992417
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.3992419, upper bound: 20.3992419
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6264957, upper bound: 20.6264923
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6264957, upper bound: 20.6264923
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.3992414, upper bound: 20.3992412
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.3992414, upper bound: 20.3992410
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6264957, upper bound: 20.6264936
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6264957, upper bound: 20.6264923
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6264598, upper bound: 20.6264561
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6264598, upper bound: 20.6264576
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6269866, upper bound: 20.6269863
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6269866, upper bound: 20.6269862
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6090916, upper bound: 20.6090911
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6090916, upper bound: 20.6090914
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101093
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 3, lower bound: -20.6101099, upper bound: 20.6101095

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100685
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100671
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100685
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100686
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100687
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100681
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6099474, upper bound: 20.6099473
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6099474, upper bound: 20.6099469
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3727615, upper bound: 20.3727606
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3727615, upper bound: 20.3727606
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090916, upper bound: 20.6090874
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090916, upper bound: 20.6090896
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090857, upper bound: 20.6090844
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090857, upper bound: 20.6090857
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6264598, upper bound: 20.6264568
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6264598, upper bound: 20.6264578
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090857, upper bound: 20.6090850
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090857, upper bound: 20.6090855
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6263499, upper bound: 20.6263479
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6263499, upper bound: 20.6263493
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263134
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263138
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3754213, upper bound: 20.3754213
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3754213, upper bound: 20.3754213
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6099064, upper bound: 20.6099061
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6099064, upper bound: 20.6099058
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089285
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089286
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089286
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089288
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100656
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100682
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100684
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100684
time: 0.42 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.81 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100685
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100671
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100685
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100686
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100687
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100681
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6099474, upper bound: 20.6099473
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6099474, upper bound: 20.6099469
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.3727615, upper bound: 20.3727606
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.3727615, upper bound: 20.3727606
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6090916, upper bound: 20.6090874
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6090916, upper bound: 20.6090896
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6090857, upper bound: 20.6090844
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6090857, upper bound: 20.6090857
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6264598, upper bound: 20.6264568
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6264598, upper bound: 20.6264578
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6090857, upper bound: 20.6090850
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6090857, upper bound: 20.6090855
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6263499, upper bound: 20.6263479
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6263499, upper bound: 20.6263493
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263134
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263138
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.3754213, upper bound: 20.3754213
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.3754213, upper bound: 20.3754213
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6099064, upper bound: 20.6099061
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6099064, upper bound: 20.6099058
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089285
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089286
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089286
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089288
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100656
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100682
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100684
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.81
Output dim: 3, lower bound: -20.6100688, upper bound: 20.6100684

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090501
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090487
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502374
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502374
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502374
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502374
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090482
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090482
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090495
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090484
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502374
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502372
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3722391, upper bound: 20.3722389
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3722389, upper bound: 20.3722380
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3722391, upper bound: 20.3722389
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3722391, upper bound: 20.3722389
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3727609, upper bound: 20.3727609
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3727609, upper bound: 20.3727609
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089255
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089256
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090485
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090481
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3737250, upper bound: 20.3737250
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3737250, upper bound: 20.3737250
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090502
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090501
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090484
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090501
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3737250, upper bound: 20.3737250
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3737250, upper bound: 20.3737250
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090493
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090484
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263113
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263118
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263134
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263134
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263113
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263125
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263120
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263126
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088874
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088874
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6099064, upper bound: 20.6099036
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6099064, upper bound: 20.6099062
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089269
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089263
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088877
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088877
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722382
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722382
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089288
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089259
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6099064, upper bound: 20.6099060
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6099064, upper bound: 20.6099060
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502374
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502374
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090500
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090503
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502372
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502372
time: 0.39 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.82 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090501
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090487
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502374
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502374
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502374
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502374
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090482
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090482
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090495
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090484
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502374
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502372
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3722391, upper bound: 20.3722389
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3722389, upper bound: 20.3722380
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3722391, upper bound: 20.3722389
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3722391, upper bound: 20.3722389
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3727609, upper bound: 20.3727609
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3727609, upper bound: 20.3727609
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089255
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089256
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090485
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090481
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3737250, upper bound: 20.3737250
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3737250, upper bound: 20.3737250
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090502
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090501
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090484
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090501
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3737250, upper bound: 20.3737250
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3737250, upper bound: 20.3737250
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090493
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090484
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263113
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263118
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263134
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263134
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263113
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263125
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263120
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6263140, upper bound: 20.6263126
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088874
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088874
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6099064, upper bound: 20.6099036
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6099064, upper bound: 20.6099062
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089269
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089263
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088877
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088877
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722382
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722382
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089288
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6089290, upper bound: 20.6089259
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6099064, upper bound: 20.6099060
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6099064, upper bound: 20.6099060
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502374
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502374
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090500
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.6090505, upper bound: 20.6090503
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502372
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.82
Output dim: 3, lower bound: -20.3502374, upper bound: 20.3502372

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088875
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088863
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088859
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088844
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502207
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502207
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088875
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088846
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722382
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722373
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088875
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088856
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088852
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088874
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088860
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088879
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088846
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088846
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088872
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088877
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088874
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088843
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088878
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088877
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088873
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088850
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088876
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088859
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088875
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088874
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088860
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088864
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088849
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088878
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088860
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088879
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088833
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088860
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088877
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088862
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088876
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088860
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722373
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722382
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3722373, upper bound: 20.3722382
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722382
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088876
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088875
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088875
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088875
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722382
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722382
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496775, upper bound: 20.3496773
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496775, upper bound: 20.3496773
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088856
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088877
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088851
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088878
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208
time: 0.47 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 1.96 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088875
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088863
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088859
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088844
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502207
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502207
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088875
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088846
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722382
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722373
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088875
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088852
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088874
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088860
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088879
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088846
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088846
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088872
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088877
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088874
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088843
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088878
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088877
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088873
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088850
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088876
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088859
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088875
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088874
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088860
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088864
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3749114, upper bound: 20.3749114
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088849
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088878
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088860
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088879
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088833
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088860
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088877
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088862
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088876
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088860
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722373
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722382
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3722373, upper bound: 20.3722382
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722382
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088876
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088875
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088875
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088875
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722382
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3722382, upper bound: 20.3722382
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3496775, upper bound: 20.3496773
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3496775, upper bound: 20.3496773
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088877
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088851
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.6088879, upper bound: 20.6088878
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -20.3502208, upper bound: 20.3502208

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050
1: -15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170
2: -8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106
3: -7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108
4: -10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
time: 0.41 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 2.00 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496561
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.00
Output dim: 3, lower bound: -20.3496562, upper bound: 20.3496562

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.48 + 308.06 = 310.54 seconds
