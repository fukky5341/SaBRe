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
execution time: IAR + RelationalAnalysis = 0.72 + 2.20 = 2.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.69 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.69
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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

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

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.31 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
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

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

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

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
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

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.71 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.20 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

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

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

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

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.73 seconds

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

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

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

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

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

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.39 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.70 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.00 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2000167, upper bound: 0.2000167
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2000167, upper bound: 0.2000167
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.19 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2000167, upper bound: 0.2000167
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2000167, upper bound: 0.2000167
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004835, upper bound: 0.2004835
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004835, upper bound: 0.2004835
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004149, upper bound: 0.2004149
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004149, upper bound: 0.2004149
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.61 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2004835, upper bound: 0.2004835
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2004835, upper bound: 0.2004835
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2004149, upper bound: 0.2004149
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2004149, upper bound: 0.2004149
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2008986, upper bound: 0.2008986
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2008986, upper bound: 0.2008986
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2001152, upper bound: 0.2001152
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2001152, upper bound: 0.2001152
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 60

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1994600, upper bound: 0.1994600
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1994600, upper bound: 0.1994600
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1996012, upper bound: 0.1996012
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1996012, upper bound: 0.1996012
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.55 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2008986, upper bound: 0.2008986
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2008986, upper bound: 0.2008986
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2001152, upper bound: 0.2001152
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2001152, upper bound: 0.2001152
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.1994600, upper bound: 0.1994600
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.1994600, upper bound: 0.1994600
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.1996012, upper bound: 0.1996012
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.1996012, upper bound: 0.1996012
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.92 + 598.06 = 600.98 seconds
