## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.20078461439999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548)
1: (-0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007)
2: (0.1967014, 1.0225546, 0.1967014, 1.0225546, -0.8258532, 0.8258532)
3: (-0.0263850, 0.2818646, -0.0263850, 0.2818646, -0.3082497, 0.3082497)
4: (-0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820)
5: (-0.0893955, 0.1868499, -0.0893955, 0.1868499, -0.2762454, 0.2762454)
6: (-0.1348234, 0.2890729, -0.1348234, 0.2890729, -0.4238963, 0.4238963)
7: (-0.1084372, 0.2871872, -0.1084372, 0.2871872, -0.3956245, 0.3956245)
8: (-0.0535781, 0.1999389, -0.0535781, 0.1999389, -0.2535171, 0.2535171)
9: (-0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.65 + 2.36 = 4.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.76 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0225546, 0.1967014, 1.0225546, -0.8258532, 0.8258532
3: -0.0263850, 0.2818646, -0.0263850, 0.2818646, -0.3082497, 0.3082497
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0893955, 0.1868499, -0.0893955, 0.1868499, -0.2762454, 0.2762454
6: -0.1348234, 0.2890729, -0.1348234, 0.2890729, -0.4238963, 0.4238963
7: -0.1084372, 0.2871872, -0.1084372, 0.2871872, -0.3956245, 0.3956245
8: -0.0535781, 0.1999389, -0.0535781, 0.1999389, -0.2535171, 0.2535171
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.04 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0225546, 0.1967014, 1.0225546, -0.8258532, 0.8258532
3: -0.0263850, 0.2818646, -0.0263850, 0.2818646, -0.3082497, 0.3082497
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0893955, 0.1868499, -0.0893955, 0.1868499, -0.2762454, 0.2762454
6: -0.1348234, 0.2890729, -0.1348234, 0.2890729, -0.4238963, 0.4238963
7: -0.1084372, 0.2871872, -0.1084372, 0.2871872, -0.3956245, 0.3956245
8: -0.0535781, 0.1999389, -0.0535781, 0.1999389, -0.2535171, 0.2535171
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.77 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.77
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.77
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.77
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.77
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0225546, 0.1967014, 1.0225546, -0.8258532, 0.8258532
3: -0.0263850, 0.2818646, -0.0263850, 0.2818646, -0.3082497, 0.3082497
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0893955, 0.1868499, -0.0893955, 0.1868499, -0.2762454, 0.2762454
6: -0.1348234, 0.2890729, -0.1348234, 0.2890729, -0.4238963, 0.4238963
7: -0.1084372, 0.2871872, -0.1084372, 0.2871872, -0.3956245, 0.3956245
8: -0.0535781, 0.1999389, -0.0535781, 0.1999389, -0.2535171, 0.2535171
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0225546, 0.1967014, 1.0225546, -0.8258532, 0.8258532
3: -0.0263850, 0.2818646, -0.0263850, 0.2818646, -0.3082497, 0.3082497
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0893955, 0.1868499, -0.0893955, 0.1868499, -0.2762454, 0.2762454
6: -0.1348234, 0.2890729, -0.1348234, 0.2890729, -0.4238963, 0.4238963
7: -0.1084372, 0.2871872, -0.1084372, 0.2871872, -0.3956245, 0.3956245
8: -0.0535781, 0.1999389, -0.0535781, 0.1999389, -0.2535171, 0.2535171
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0225546, 0.1967014, 1.0225546, -0.8258532, 0.8258532
3: -0.0263850, 0.2818646, -0.0263850, 0.2818646, -0.3082497, 0.3082497
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0893955, 0.1868499, -0.0893955, 0.1868499, -0.2762454, 0.2762454
6: -0.1348234, 0.2890729, -0.1348234, 0.2890729, -0.4238963, 0.4238963
7: -0.1084372, 0.2871872, -0.1084372, 0.2871872, -0.3956245, 0.3956245
8: -0.0535781, 0.1999389, -0.0535781, 0.1999389, -0.2535171, 0.2535171
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0225546, 0.1967014, 1.0225546, -0.8258532, 0.8258532
3: -0.0263850, 0.2818646, -0.0263850, 0.2818646, -0.3082497, 0.3082497
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0893955, 0.1868499, -0.0893955, 0.1868499, -0.2762454, 0.2762454
6: -0.1348234, 0.2890729, -0.1348234, 0.2890729, -0.4238963, 0.4238963
7: -0.1084372, 0.2871872, -0.1084372, 0.2871872, -0.3956245, 0.3956245
8: -0.0535781, 0.1999389, -0.0535781, 0.1999389, -0.2535171, 0.2535171
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481
time: 0.78 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.28 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 3, lower bound: -0.2008481, upper bound: 0.2008481

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0225546, 0.1967014, 1.0225546, -0.8258532, 0.8258532
3: -0.0263850, 0.2818646, -0.0263850, 0.2818646, -0.3082497, 0.3082497
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0893955, 0.1868499, -0.0893955, 0.1868499, -0.2762454, 0.2762454
6: -0.1348234, 0.2890729, -0.1348234, 0.2890729, -0.4238963, 0.4238963
7: -0.1084372, 0.2871872, -0.1084372, 0.2871872, -0.3956245, 0.3956245
8: -0.0535781, 0.1999389, -0.0535781, 0.1999389, -0.2535171, 0.2535171
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0225546, 0.1967014, 1.0225546, -0.8258532, 0.8258532
3: -0.0263850, 0.2818646, -0.0263850, 0.2818646, -0.3082497, 0.3082497
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0893955, 0.1868499, -0.0893955, 0.1868499, -0.2762454, 0.2762454
6: -0.1348234, 0.2890729, -0.1348234, 0.2890729, -0.4238963, 0.4238963
7: -0.1084372, 0.2871872, -0.1084372, 0.2871872, -0.3956245, 0.3956245
8: -0.0535781, 0.1999389, -0.0535781, 0.1999389, -0.2535171, 0.2535171
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0225546, 0.1967014, 1.0225546, -0.8258532, 0.8258532
3: -0.0263850, 0.2818646, -0.0263850, 0.2818646, -0.3082497, 0.3082497
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0893955, 0.1868499, -0.0893955, 0.1868499, -0.2762454, 0.2762454
6: -0.1348234, 0.2890729, -0.1348234, 0.2890729, -0.4238963, 0.4238963
7: -0.1084372, 0.2871872, -0.1084372, 0.2871872, -0.3956245, 0.3956245
8: -0.0535781, 0.1999389, -0.0535781, 0.1999389, -0.2535171, 0.2535171
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0225546, 0.1967014, 1.0225546, -0.8258532, 0.8258532
3: -0.0263850, 0.2818646, -0.0263850, 0.2818646, -0.3082497, 0.3082497
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0893955, 0.1868499, -0.0893955, 0.1868499, -0.2762454, 0.2762454
6: -0.1348234, 0.2890729, -0.1348234, 0.2890729, -0.4238963, 0.4238963
7: -0.1084372, 0.2871872, -0.1084372, 0.2871872, -0.3956245, 0.3956245
8: -0.0535781, 0.1999389, -0.0535781, 0.1999389, -0.2535171, 0.2535171
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0225546, 0.1967014, 1.0225546, -0.8258532, 0.8258532
3: -0.0263850, 0.2818646, -0.0263850, 0.2818646, -0.3082497, 0.3082497
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0893955, 0.1868499, -0.0893955, 0.1868499, -0.2762454, 0.2762454
6: -0.1348234, 0.2890729, -0.1348234, 0.2890729, -0.4238963, 0.4238963
7: -0.1084372, 0.2871872, -0.1084372, 0.2871872, -0.3956245, 0.3956245
8: -0.0535781, 0.1999389, -0.0535781, 0.1999389, -0.2535171, 0.2535171
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0225546, 0.1967014, 1.0225546, -0.8258532, 0.8258532
3: -0.0263850, 0.2818646, -0.0263850, 0.2818646, -0.3082497, 0.3082497
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0893955, 0.1868499, -0.0893955, 0.1868499, -0.2762454, 0.2762454
6: -0.1348234, 0.2890729, -0.1348234, 0.2890729, -0.4238963, 0.4238963
7: -0.1084372, 0.2871872, -0.1084372, 0.2871872, -0.3956245, 0.3956245
8: -0.0535781, 0.1999389, -0.0535781, 0.1999389, -0.2535171, 0.2535171
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0225546, 0.1967014, 1.0225546, -0.8258532, 0.8258532
3: -0.0263850, 0.2818646, -0.0263850, 0.2818646, -0.3082497, 0.3082497
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0893955, 0.1868499, -0.0893955, 0.1868499, -0.2762454, 0.2762454
6: -0.1348234, 0.2890729, -0.1348234, 0.2890729, -0.4238963, 0.4238963
7: -0.1084372, 0.2871872, -0.1084372, 0.2871872, -0.3956245, 0.3956245
8: -0.0535781, 0.1999389, -0.0535781, 0.1999389, -0.2535171, 0.2535171
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0225546, 0.1967014, 1.0225546, -0.8258532, 0.8258532
3: -0.0263850, 0.2818646, -0.0263850, 0.2818646, -0.3082497, 0.3082497
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0893955, 0.1868499, -0.0893955, 0.1868499, -0.2762454, 0.2762454
6: -0.1348234, 0.2890729, -0.1348234, 0.2890729, -0.4238963, 0.4238963
7: -0.1084372, 0.2871872, -0.1084372, 0.2871872, -0.3956245, 0.3956245
8: -0.0535781, 0.1999389, -0.0535781, 0.1999389, -0.2535171, 0.2535171
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.81 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.29 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.02 + 49.12 = 53.13 seconds
