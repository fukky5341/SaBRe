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
execution time: IAR + RelationalAnalysis = 2.40 + 2.55 = 4.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.19 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.48 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.48
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.48
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967015, 1.0198972, 0.1967014, 1.0225546, -0.8258532, 0.8231956
3: -0.0243124, 0.2818646, -0.0263850, 0.2818646, -0.3061770, 0.3082497
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0874891, 0.1868499, -0.0893955, 0.1868499, -0.2743390, 0.2762454
6: -0.1325822, 0.2890729, -0.1348234, 0.2890729, -0.4216550, 0.4238963
7: -0.1066403, 0.2871872, -0.1084372, 0.2871872, -0.3938276, 0.3956245
8: -0.0515544, 0.1999389, -0.0535781, 0.1999389, -0.2514933, 0.2535171
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.1070234, 0.1273829, -0.0911179, 0.1273829, -0.2344063, 0.2185007
2: 0.1967013, 1.0544252, 0.1967014, 1.0215276, -0.8248261, 0.8577235
3: -0.0512403, 0.2818646, -0.0255841, 0.2818646, -0.3331048, 0.3074486
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.1122566, 0.1868500, -0.0886588, 0.1868500, -0.2991065, 0.2755087
6: -0.1616995, 0.2890729, -0.1339573, 0.2890729, -0.4507724, 0.4230302
7: -0.1299852, 0.2871872, -0.1077428, 0.2871872, -0.4171725, 0.3949300
8: -0.0778472, 0.1999390, -0.0527961, 0.1999390, -0.2777861, 0.2527350
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.29 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.86 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279924, 0.2164623, -0.3444547, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967015, 1.0198972, 0.1967015, 1.0198972, -0.8231956, 0.8231956
3: -0.0243124, 0.2818646, -0.0243124, 0.2818646, -0.3061770, 0.3061770
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0874891, 0.1868499, -0.0874891, 0.1868499, -0.2743390, 0.2743390
6: -0.1325822, 0.2890729, -0.1325822, 0.2890729, -0.4216551, 0.4216551
7: -0.1066403, 0.2871872, -0.1066403, 0.2871872, -0.3938276, 0.3938276
8: -0.0515544, 0.1999389, -0.0515544, 0.1999389, -0.2514933, 0.2514933
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.23 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.27 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.1070234, 0.1273829, -0.2185007, 0.2344063
2: 0.1967015, 1.0198972, 0.1967013, 1.0544252, -0.8577235, 0.8231956
3: -0.0243124, 0.2818646, -0.0512403, 0.2818646, -0.3061770, 0.3331048
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0874891, 0.1868499, -0.1122566, 0.1868500, -0.2743391, 0.2991065
6: -0.1325822, 0.2890729, -0.1616995, 0.2890729, -0.4216551, 0.4507725
7: -0.1066403, 0.2871872, -0.1299852, 0.2871872, -0.3938276, 0.4171725
8: -0.0515544, 0.1999389, -0.0778472, 0.1999390, -0.2514934, 0.2777861
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.24 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444547, 0.3444548
1: -0.1070234, 0.1273829, -0.0911179, 0.1273829, -0.2344063, 0.2185007
2: 0.1967013, 1.0544252, 0.1967015, 1.0198972, -0.8231956, 0.8577235
3: -0.0512403, 0.2818646, -0.0243124, 0.2818646, -0.3331048, 0.3061770
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.1122566, 0.1868500, -0.0874891, 0.1868499, -0.2991065, 0.2743391
6: -0.1616995, 0.2890729, -0.1325822, 0.2890729, -0.4507724, 0.4216551
7: -0.1299852, 0.2871872, -0.1066403, 0.2871872, -0.4171725, 0.3938276
8: -0.0778472, 0.1999390, -0.0515544, 0.1999389, -0.2777861, 0.2514934
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.1070234, 0.1273829, -0.1070234, 0.1273829, -0.2344063, 0.2344063
2: 0.1967013, 1.0544252, 0.1967013, 1.0544252, -0.8577235, 0.8577236
3: -0.0512403, 0.2818646, -0.0512403, 0.2818646, -0.3331048, 0.3331048
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.1122566, 0.1868500, -0.1122566, 0.1868500, -0.2991065, 0.2991065
6: -0.1616995, 0.2890729, -0.1616995, 0.2890729, -0.4507724, 0.4507724
7: -0.1299852, 0.2871872, -0.1299852, 0.2871872, -0.4171725, 0.4171725
8: -0.0778472, 0.1999390, -0.0778472, 0.1999390, -0.2777861, 0.2777861
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.04 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.55 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185008, 0.2185008
2: 0.1967013, 1.0138386, 0.1967015, 1.0198972, -0.8231956, 0.8171372
3: -0.0195876, 0.2818646, -0.0243124, 0.2818646, -0.3014521, 0.3061770
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0831433, 0.1868500, -0.0874891, 0.1868499, -0.2699932, 0.2743391
6: -0.1274733, 0.2890729, -0.1325822, 0.2890729, -0.4165462, 0.4216551
7: -0.1025442, 0.2871873, -0.1066403, 0.2871872, -0.3897314, 0.3938276
8: -0.0469409, 0.1999389, -0.0515544, 0.1999389, -0.2468798, 0.2514934
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 1.0091932, 0.1967014, 1.0173097, -0.8206083, 0.8124917
3: -0.0159645, 0.2818646, -0.0222947, 0.2818646, -0.2978291, 0.3041592
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0798110, 0.1868499, -0.0856333, 0.1868499, -0.2666609, 0.2724832
6: -0.1235557, 0.2890729, -0.1304006, 0.2890729, -0.4126286, 0.4194734
7: -0.0994033, 0.2871873, -0.1048911, 0.2871872, -0.3865905, 0.3920784
8: -0.0434034, 0.1999390, -0.0495843, 0.1999389, -0.2433423, 0.2495232
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185008
2: 0.1967013, 0.9961182, 0.1967014, 0.9174118, -0.7207104, 0.7994168
3: -0.0057606, 0.2818646, 0.0484848, 0.2818646, -0.2876253, 0.2333799
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0704257, 0.1868500, -0.0636812, 0.1868500, -0.2572757, 0.2505312
6: -0.1125307, 0.2890729, -0.0513771, 0.2890729, -0.4016036, 0.3404500
7: -0.0905571, 0.2871873, -0.0421730, 0.2871872, -0.3777443, 0.3293602
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.31 seconds

## Relational analysis of NS_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444547
1: -0.0911179, 0.1273829, -0.0941756, 0.1273828, -0.2185007, 0.2215585
2: 0.1967015, 1.0198972, 0.1967014, 1.0336711, -0.8369694, 0.8231956
3: -0.0243124, 0.2818646, -0.0350547, 0.2818646, -0.3061770, 0.3169193
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0874891, 0.1868499, -0.0973696, 0.1868500, -0.2743391, 0.2842195
6: -0.1325822, 0.2890729, -0.1441980, 0.2890729, -0.4216551, 0.4332709
7: -0.1066403, 0.2871872, -0.1159533, 0.2871873, -0.3938276, 0.4031405
8: -0.0515544, 0.1999389, -0.0620433, 0.1999389, -0.2514933, 0.2619822
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## Relational analysis of NS_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273828, -0.0911178, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 0.9174118, 0.1967013, 0.9961182, -0.7994168, 0.7207104
3: 0.0484848, 0.2818646, -0.0057606, 0.2818646, -0.2333799, 0.2876253
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0704257, 0.1868500, -0.2505312, 0.2572757
6: -0.0513771, 0.2890729, -0.1125307, 0.2890729, -0.3404500, 0.4016036
7: -0.0421730, 0.2871872, -0.0905571, 0.2871873, -0.3293602, 0.3777443
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444547, 0.3444547
1: -0.0941756, 0.1273828, -0.0911179, 0.1273829, -0.2215585, 0.2185007
2: 0.1967014, 1.0336711, 0.1967015, 1.0198972, -0.8231956, 0.8369694
3: -0.0350547, 0.2818646, -0.0243124, 0.2818646, -0.3169193, 0.3061770
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0973696, 0.1868500, -0.0874891, 0.1868499, -0.2842195, 0.2743391
6: -0.1441980, 0.2890729, -0.1325822, 0.2890729, -0.4332708, 0.4216551
7: -0.1159533, 0.2871873, -0.1066403, 0.2871872, -0.4031405, 0.3938276
8: -0.0620433, 0.1999389, -0.0515544, 0.1999389, -0.2619822, 0.2514933
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.15 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444548
1: -0.1070234, 0.1273829, -0.1034809, 0.1273829, -0.2344063, 0.2308638
2: 0.1967013, 1.0544252, 0.1967014, 1.0487027, -0.8520012, 0.8577236
3: -0.0512403, 0.2818646, -0.0467774, 0.2818646, -0.3331048, 0.3286420
4: -0.2053552, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.1122566, 0.1868500, -0.1081518, 0.1868500, -0.2991065, 0.2950017
6: -0.1616995, 0.2890729, -0.1568738, 0.2890729, -0.4507724, 0.4459467
7: -0.1299852, 0.2871872, -0.1261162, 0.2871872, -0.4171725, 0.4133034
8: -0.0778472, 0.1999390, -0.0734896, 0.1999389, -0.2777861, 0.2734286
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
time: 0.97 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.1054557, 0.1273829, -0.1003349, 0.1273829, -0.2328386, 0.2277178
2: 0.1967014, 1.0518925, 0.1967014, 1.0436209, -0.8469194, 0.8551911
3: -0.0492653, 0.2818646, -0.0428142, 0.2818646, -0.3311299, 0.3246788
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.1104401, 0.1868500, -0.1045065, 0.1868499, -0.2972900, 0.2913564
6: -0.1595640, 0.2890729, -0.1525883, 0.2890729, -0.4486369, 0.4416612
7: -0.1282730, 0.2871873, -0.1226803, 0.2871873, -0.4154603, 0.4098675
8: -0.0759188, 0.1999389, -0.0696198, 0.1999389, -0.2758577, 0.2695587
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
time: 0.96 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.86 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.9901263, 0.1967014, 0.8853604, -0.6886589, 0.7934248
3: -0.0010699, 0.2818646, 0.0672506, 0.2818647, -0.2829345, 0.2146140
4: -0.2053551, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0661114, 0.1868500, -0.0636812, 0.1868499, -0.2529613, 0.2505312
6: -0.1074807, 0.2890729, -0.0306596, 0.2890729, -0.3965536, 0.3197325
7: -0.0864906, 0.2871872, -0.0379298, 0.2871872, -0.3736778, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185008, 0.2185007
2: 0.1967013, 1.0138386, 0.1967014, 0.9990969, -0.8023955, 0.8171372
3: -0.0195876, 0.2818646, -0.0080906, 0.2818646, -0.3014521, 0.2899552
4: -0.2053551, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0831433, 0.1868500, -0.0725688, 0.1868500, -0.2699933, 0.2594187
6: -0.1274733, 0.2890729, -0.1150415, 0.2890729, -0.4165461, 0.4041144
7: -0.1025442, 0.2871873, -0.0925770, 0.2871873, -0.3897315, 0.3797643
8: -0.0469409, 0.1999389, -0.0372933, 0.1999389, -0.2468798, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9853868, 0.1967014, 0.8832294, -0.6865277, 0.7886851
3: 0.0026404, 0.2818646, 0.0683906, 0.2818646, -0.2792242, 0.2134740
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868499, -0.2505311, 0.2505311
6: -0.1034860, 0.2890729, -0.0292835, 0.2890729, -0.3925590, 0.3183564
7: -0.0832738, 0.2871872, -0.0379298, 0.2871872, -0.3704611, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 1.0091932, 0.1967014, 0.9965099, -0.7998084, 0.8124918
3: -0.0159645, 0.2818646, -0.0060671, 0.2818646, -0.2978291, 0.2879317
4: -0.2053551, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0798110, 0.1868499, -0.0707076, 0.1868500, -0.2666609, 0.2575576
6: -0.1235557, 0.2890729, -0.1128606, 0.2890729, -0.4126286, 0.4019335
7: -0.0994033, 0.2871873, -0.0908228, 0.2871872, -0.3865905, 0.3780101
8: -0.0434034, 0.1999390, -0.0372933, 0.1999389, -0.2433423, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9948581, 0.1967014, 0.8968494, -0.7001479, 0.7981567
3: -0.0047742, 0.2818647, 0.0608644, 0.2818646, -0.2866388, 0.2210002
4: -0.2053551, 0.1310269, -0.2053552, 0.1310268, -0.3363820, 0.3363820
5: -0.0695185, 0.1868499, -0.0636812, 0.1868500, -0.2563684, 0.2505311
6: -0.1114687, 0.2890729, -0.0379278, 0.2890729, -0.4005416, 0.3270006
7: -0.0897019, 0.2871873, -0.0379298, 0.2871872, -0.3768891, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_B1_B1_A1

### Relational analysis result of NS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.08 seconds

## Relational analysis of NS_A1_B2_B1_B1_A2

### Relational analysis result of NS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.9958842, 0.1967014, 0.9140833, -0.7173819, 0.7991828
3: -0.0055775, 0.2818646, 0.0504888, 0.2818646, -0.2874422, 0.2313759
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0702574, 0.1868499, -0.0636812, 0.1868500, -0.2571074, 0.2505312
6: -0.1123337, 0.2890729, -0.0491156, 0.2890729, -0.4014066, 0.3381884
7: -0.0903984, 0.2871872, -0.0405730, 0.2871873, -0.3775856, 0.3277603
8: -0.0372933, 0.1999389, -0.0372934, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

## Relational analysis of NS_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279924, 0.2164623, -0.3444547, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967015, 1.0198972, 0.1967014, 1.0278074, -0.8311057, 0.8231956
3: -0.0243124, 0.2818646, -0.0304814, 0.2818646, -0.3061769, 0.3123461
4: -0.2053551, 0.1310269, -0.2053552, 0.1310268, -0.3363820, 0.3363820
5: -0.0874891, 0.1868499, -0.0931633, 0.1868499, -0.2743390, 0.2800132
6: -0.1325822, 0.2890729, -0.1392529, 0.2890729, -0.4216551, 0.4283258
7: -0.1066403, 0.2871872, -0.1119886, 0.2871873, -0.3938276, 0.3991758
8: -0.0515544, 0.1999389, -0.0575780, 0.1999390, -0.2514934, 0.2575169
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_B2_B1_B1

### Relational analysis result of NS_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2

### Relational analysis result of NS_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279924, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0173097, 0.1967014, 1.0231981, -0.8264967, 0.8206083
3: -0.0222947, 0.2818646, -0.0268868, 0.2818646, -0.3041592, 0.3087514
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0856333, 0.1868499, -0.0898570, 0.1868500, -0.2724832, 0.2767069
6: -0.1304006, 0.2890729, -0.1353659, 0.2890729, -0.4194734, 0.4244388
7: -0.1048911, 0.2871872, -0.1088723, 0.2871872, -0.3920784, 0.3960595
8: -0.0495843, 0.1999389, -0.0540680, 0.1999390, -0.2495232, 0.2540070
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
time: 1.12 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8968494, 0.1967014, 0.9948581, -0.7981567, 0.7001479
3: 0.0608644, 0.2818646, -0.0047742, 0.2818647, -0.2210002, 0.2866388
4: -0.2053552, 0.1310268, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0695185, 0.1868499, -0.2505311, 0.2563684
6: -0.0379278, 0.2890729, -0.1114687, 0.2890729, -0.3270006, 0.4005416
7: -0.0379298, 0.2871872, -0.0897019, 0.2871873, -0.3251171, 0.3768891
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.22 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.12 seconds

## BFS NS instance: NS_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273828, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9140833, 0.1967014, 0.9958842, -0.7991828, 0.7173818
3: 0.0504888, 0.2818646, -0.0055775, 0.2818646, -0.2313759, 0.2874422
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0702574, 0.1868499, -0.2505312, 0.2571074
6: -0.0491156, 0.2890729, -0.1123337, 0.2890729, -0.3381884, 0.4014066
7: -0.0405730, 0.2871873, -0.0903984, 0.2871872, -0.3277603, 0.3775856
8: -0.0372934, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.10 seconds

## BFS NS instance: NS_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279924, 0.2164623, -0.3444547, 0.3444547
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0278074, 0.1967015, 1.0198972, -0.8231956, 0.8311058
3: -0.0304814, 0.2818646, -0.0243124, 0.2818646, -0.3123461, 0.3061769
4: -0.2053552, 0.1310268, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0931633, 0.1868499, -0.0874891, 0.1868499, -0.2800132, 0.2743390
6: -0.1392529, 0.2890729, -0.1325822, 0.2890729, -0.4283258, 0.4216551
7: -0.1119886, 0.2871873, -0.1066403, 0.2871872, -0.3991758, 0.3938276
8: -0.0575780, 0.1999390, -0.0515544, 0.1999389, -0.2575169, 0.2514934
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2

### Relational analysis result of NS_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0231981, 0.1967014, 1.0173097, -0.8206083, 0.8264967
3: -0.0268868, 0.2818646, -0.0222947, 0.2818646, -0.3087514, 0.3041592
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0898570, 0.1868500, -0.0856333, 0.1868499, -0.2767069, 0.2724832
6: -0.1353659, 0.2890729, -0.1304006, 0.2890729, -0.4244388, 0.4194734
7: -0.1088723, 0.2871872, -0.1048911, 0.2871872, -0.3960595, 0.3920784
8: -0.0540680, 0.1999390, -0.0495843, 0.1999389, -0.2540070, 0.2495232
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273828, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9174118, 0.1967014, 1.0249617, -0.8282603, 0.7207104
3: 0.0484848, 0.2818646, -0.0282622, 0.2818646, -0.2333799, 0.3101269
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0911221, 0.1868500, -0.2505312, 0.2779720
6: -0.0513771, 0.2890729, -0.1368532, 0.2890729, -0.3404500, 0.4259261
7: -0.0421730, 0.2871872, -0.1100646, 0.2871872, -0.3293602, 0.3972518
8: -0.0372933, 0.1999389, -0.0554110, 0.1999389, -0.2372322, 0.2553500
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
time: 0.97 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0941756, 0.1273828, -0.1034809, 0.1273829, -0.2215585, 0.2308638
2: 0.1967014, 1.0336711, 0.1967014, 1.0487027, -0.8520011, 0.8369694
3: -0.0350547, 0.2818646, -0.0467774, 0.2818646, -0.3169193, 0.3286421
4: -0.2053551, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0973696, 0.1868500, -0.1081518, 0.1868500, -0.2842195, 0.2950017
6: -0.1441980, 0.2890729, -0.1568738, 0.2890729, -0.4332708, 0.4459467
7: -0.1159533, 0.2871873, -0.1261162, 0.2871872, -0.4031405, 0.4133034
8: -0.0620433, 0.1999389, -0.0734896, 0.1999389, -0.2619822, 0.2734285
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.93 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 0.9150447, 0.1967014, 1.0198543, -0.8231528, 0.7183430
3: 0.0499100, 0.2818646, -0.0242790, 0.2818646, -0.2319546, 0.3061436
4: -0.2053552, 0.1310269, -0.2053552, 0.1310268, -0.3363820, 0.3363821
5: -0.0636812, 0.1868499, -0.0874584, 0.1868499, -0.2505311, 0.2743083
6: -0.0497688, 0.2890729, -0.1325462, 0.2890729, -0.3388417, 0.4216191
7: -0.0410352, 0.2871872, -0.1066114, 0.2871872, -0.3282224, 0.3937986
8: -0.0372933, 0.1999389, -0.0515218, 0.1999390, -0.2372323, 0.2514607
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
time: 1.07 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
time: 1.07 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0926430, 0.1273829, -0.1003349, 0.1273829, -0.2200259, 0.2277178
2: 0.1967014, 1.0311956, 0.1967014, 1.0436209, -0.8469194, 0.8344941
3: -0.0331241, 0.2818647, -0.0428142, 0.2818646, -0.3149886, 0.3246788
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0955938, 0.1868500, -0.1045065, 0.1868499, -0.2824437, 0.2913564
6: -0.1421103, 0.2890729, -0.1525883, 0.2890729, -0.4311832, 0.4416612
7: -0.1142795, 0.2871872, -0.1226803, 0.2871873, -0.4014668, 0.4098675
8: -0.0601582, 0.1999389, -0.0696198, 0.1999389, -0.2600971, 0.2695587
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 1.05 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961
time: 0.95 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.11 seconds
NS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
NS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
NS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
NS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
NS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
NS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
NS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2004961

## BFS NS instance: NS_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9888560, 0.1967014, 0.8655015, -0.6688000, 0.7921544
3: -0.0000754, 0.2818646, 0.0778735, 0.2818646, -0.2819401, 0.2039912
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0651967, 0.1868499, -0.0636812, 0.1868500, -0.2520466, 0.2505311
6: -0.1064100, 0.2890729, -0.0182637, 0.2890729, -0.3954829, 0.3073366
7: -0.0856283, 0.2871872, -0.0379298, 0.2871872, -0.3728155, 0.3251170
8: -0.0372934, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 70

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.9898950, 0.1967014, 0.8822462, -0.6855447, 0.7931934
3: -0.0008888, 0.2818646, 0.0689165, 0.2818646, -0.2827534, 0.2129481
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0659447, 0.1868500, -0.0636812, 0.1868500, -0.2527947, 0.2505312
6: -0.1072856, 0.2890729, -0.0286486, 0.2890729, -0.3963585, 0.3177215
7: -0.0863335, 0.2871872, -0.0379298, 0.2871873, -0.3735207, 0.3251170
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 0.9933610, 0.1967014, 0.9990969, -0.8023955, 0.7966596
3: -0.0036024, 0.2818646, -0.0080906, 0.2818646, -0.2854669, 0.2899551
4: -0.2053551, 0.1310268, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0684406, 0.1868499, -0.0725688, 0.1868500, -0.2552905, 0.2594187
6: -0.1102071, 0.2890729, -0.1150415, 0.2890729, -0.3992799, 0.4041144
7: -0.0886860, 0.2871872, -0.0925770, 0.2871873, -0.3758732, 0.3797643
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760215, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9590479, 0.1967014, 0.9894093, -0.7927079, 0.7623464
3: 0.0232207, 0.2818646, -0.0005087, 0.2818646, -0.2586439, 0.2823734
4: -0.2053551, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0655952, 0.1868499, -0.2505311, 0.2524452
6: -0.0812855, 0.2890729, -0.1068765, 0.2890729, -0.3703585, 0.3959494
7: -0.0653780, 0.2871872, -0.0860041, 0.2871872, -0.3525653, 0.3731914
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.9841122, 0.1967014, 0.8633748, -0.6666733, 0.7874107
3: 0.0036383, 0.2818646, 0.0790112, 0.2818646, -0.2782264, 0.2028534
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505311
6: -0.1024118, 0.2890729, -0.0170335, 0.2890729, -0.3914847, 0.3061064
7: -0.0824089, 0.2871872, -0.0379298, 0.2871873, -0.3695961, 0.3251170
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 70

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.9851462, 0.1967014, 0.8800996, -0.6833982, 0.7884449
3: 0.0028286, 0.2818646, 0.0700646, 0.2818646, -0.2790361, 0.2118000
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.1032834, 0.2890729, -0.0272626, 0.2890729, -0.3923563, 0.3163354
7: -0.0831106, 0.2871873, -0.0379298, 0.2871872, -0.3702978, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185008
2: 0.1967014, 0.9886634, 0.1967014, 0.9965099, -0.7998083, 0.7919620
3: 0.0000753, 0.2818646, -0.0060671, 0.2818646, -0.2817893, 0.2879318
4: -0.2053551, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0650580, 0.1868500, -0.0707076, 0.1868500, -0.2519080, 0.2575576
6: -0.1062477, 0.2890729, -0.1128606, 0.2890729, -0.3953206, 0.4019335
7: -0.0854977, 0.2871872, -0.0908228, 0.2871872, -0.3726850, 0.3780101
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 217

## Relational analysis of NS_A1_B1_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009211, upper bound: 0.2009856
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.23 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9545982, 0.1967014, 0.9868395, -0.7901381, 0.7578967
3: 0.0260966, 0.2818646, 0.0015030, 0.2818646, -0.2557681, 0.2803616
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0637448, 0.1868500, -0.2505312, 0.2505947
6: -0.0778171, 0.2890729, -0.1047105, 0.2890729, -0.3668900, 0.3937834
7: -0.0624200, 0.2871873, -0.0842599, 0.2871872, -0.3496073, 0.3714472
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8842952, 0.1967014, 0.8968494, -0.7001478, 0.6875938
3: 0.0678203, 0.2818646, 0.0608644, 0.2818646, -0.2140443, 0.2210002
4: -0.2053552, 0.1310268, -0.2053552, 0.1310268, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505311
6: -0.0299719, 0.2890729, -0.0379278, 0.2890729, -0.3190448, 0.3270007
7: -0.0379298, 0.2871873, -0.0379298, 0.2871872, -0.3251171, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_B1_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.00 seconds

## Relational analysis of NS_A1_B2_B1_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2007515, upper bound: 0.1989681
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 0.9978466, 0.1967014, 0.8968494, -0.7001479, 0.8011451
3: -0.0071136, 0.2818646, 0.0608644, 0.2818646, -0.2889781, 0.2210002
4: -0.2053552, 0.1310269, -0.2053552, 0.1310268, -0.3363820, 0.3363820
5: -0.0716701, 0.1868500, -0.0636812, 0.1868500, -0.2585201, 0.2505312
6: -0.1139873, 0.2890729, -0.0379278, 0.2890729, -0.4030602, 0.3270007
7: -0.0917300, 0.2871872, -0.0379298, 0.2871872, -0.3789173, 0.3251171
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372323, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_B1_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.1989681
time: 0.97 seconds

## Relational analysis of NS_A1_B2_B1_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2007515, upper bound: 0.1989681
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273828, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.8851618, 0.1967014, 0.9140833, -0.7173817, 0.6884602
3: 0.0673567, 0.2818646, 0.0504888, 0.2818646, -0.2145078, 0.2313758
4: -0.2053551, 0.1310268, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0305315, 0.2890729, -0.0491156, 0.2890729, -0.3196044, 0.3381884
7: -0.0379298, 0.2871873, -0.0405730, 0.2871873, -0.3251171, 0.3277603
8: -0.0372933, 0.1999389, -0.0372934, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_B1_B2_A1_B1

### Relational analysis result of NS_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.21 seconds

## Relational analysis of NS_A1_B2_B1_B2_A1_B2

### Relational analysis result of NS_A1_B2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2007738, upper bound: 0.1989681
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.9988632, 0.1967014, 0.9140833, -0.7173819, 0.8021617
3: -0.0079096, 0.2818646, 0.0504888, 0.2818646, -0.2897742, 0.2313759
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0724022, 0.1868500, -0.0636812, 0.1868500, -0.2592522, 0.2505312
6: -0.1148442, 0.2890729, -0.0491156, 0.2890729, -0.4039172, 0.3381884
7: -0.0924201, 0.2871872, -0.0405730, 0.2871873, -0.3796074, 0.3277603
8: -0.0372933, 0.1999390, -0.0372934, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.1989681
time: 1.18 seconds

## Relational analysis of NS_A1_B2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2007738, upper bound: 0.1989681
time: 1.38 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185008
2: 0.1967015, 1.0198972, 0.1967014, 1.0069320, -0.8102305, 0.8231956
3: -0.0243124, 0.2818646, -0.0142010, 0.2818646, -0.3061769, 0.2960657
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0874891, 0.1868499, -0.0781890, 0.1868499, -0.2743390, 0.2650389
6: -0.1325822, 0.2890729, -0.1216488, 0.2890729, -0.4216551, 0.4107217
7: -0.1066403, 0.2871872, -0.0978745, 0.2871872, -0.3938276, 0.3850617
8: -0.0515544, 0.1999389, -0.0416816, 0.1999389, -0.2514933, 0.2416205
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_B2_B1_B1_A1

### Relational analysis result of NS_A1_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.16 seconds

## Relational analysis of NS_A1_B2_B2_B1_B1_A2

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0102069, 0.1967014, 0.9726555, -0.7759538, 0.8135054
3: -0.0167550, 0.2818646, 0.0126072, 0.2818646, -0.2986196, 0.2692575
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0805381, 0.1868500, -0.0636812, 0.1868500, -0.2673880, 0.2505312
6: -0.1244104, 0.2890729, -0.0927558, 0.2890729, -0.4134833, 0.3818287
7: -0.1000886, 0.2871873, -0.0746333, 0.2871872, -0.3872759, 0.3618206
8: -0.0441752, 0.1999389, -0.0372933, 0.1999390, -0.2441142, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B2_B2_B1_B2_B1

### Relational analysis result of NS_A1_B2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.18 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2_B2

### Relational analysis result of NS_A1_B2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9967450, 0.1967014, 1.0231981, -0.8264965, 0.8000435
3: -0.0062512, 0.2818646, -0.0268868, 0.2818646, -0.2881158, 0.3087513
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0708770, 0.1868500, -0.0898570, 0.1868500, -0.2577269, 0.2767069
6: -0.1130589, 0.2890729, -0.1353659, 0.2890729, -0.4021318, 0.4244388
7: -0.0909824, 0.2871872, -0.1088723, 0.2871872, -0.3781697, 0.3960595
8: -0.0372933, 0.1999390, -0.0540680, 0.1999390, -0.2372323, 0.2540070
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 217

## Relational analysis of NS_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
time: 0.85 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9626185, 0.1967014, 1.0137522, -0.8170507, 0.7659172
3: 0.0204690, 0.2818646, -0.0195201, 0.2818646, -0.2613956, 0.3013847
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0830813, 0.1868500, -0.2505312, 0.2699312
6: -0.0842917, 0.2890729, -0.1274004, 0.2890729, -0.3733646, 0.4164732
7: -0.0678176, 0.2871872, -0.1024857, 0.2871872, -0.3550049, 0.3896729
8: -0.0372933, 0.1999389, -0.0468751, 0.1999389, -0.2372322, 0.2468140
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 211

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
time: 0.87 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8968494, 0.1967014, 0.8842952, -0.6875938, 0.7001478
3: 0.0608644, 0.2818646, 0.0678203, 0.2818646, -0.2210002, 0.2140443
4: -0.2053552, 0.1310268, -0.2053552, 0.1310268, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505311, 0.2505312
6: -0.0379278, 0.2890729, -0.0299719, 0.2890729, -0.3270007, 0.3190448
7: -0.0379298, 0.2871872, -0.0379298, 0.2871873, -0.3251171, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007515
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8968494, 0.1967013, 0.9978466, -0.8011451, 0.7001479
3: 0.0608644, 0.2818646, -0.0071136, 0.2818646, -0.2210002, 0.2889781
4: -0.2053552, 0.1310268, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0716701, 0.1868500, -0.2505312, 0.2585201
6: -0.0379278, 0.2890729, -0.1139873, 0.2890729, -0.3270007, 0.4030602
7: -0.0379298, 0.2871872, -0.0917300, 0.2871872, -0.3251171, 0.3789173
8: -0.0372933, 0.1999389, -0.0372933, 0.1999390, -0.2372323, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273828, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.9140833, 0.1967014, 0.8851618, -0.6884601, 0.7173818
3: 0.0504888, 0.2818646, 0.0673567, 0.2818646, -0.2313758, 0.2145078
4: -0.2053551, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0491156, 0.2890729, -0.0305315, 0.2890729, -0.3381884, 0.3196044
7: -0.0405730, 0.2871873, -0.0379298, 0.2871873, -0.3277603, 0.3251171
8: -0.0372934, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007738
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273828, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9140833, 0.1967014, 0.9988632, -0.8021617, 0.7173818
3: 0.0504888, 0.2818646, -0.0079096, 0.2818646, -0.2313759, 0.2897742
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0724022, 0.1868500, -0.2505312, 0.2592522
6: -0.0491156, 0.2890729, -0.1148442, 0.2890729, -0.3381884, 0.4039172
7: -0.0405730, 0.2871873, -0.0924201, 0.2871872, -0.3277603, 0.3796074
8: -0.0372934, 0.1999389, -0.0372933, 0.1999390, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 1.0069320, 0.1967015, 1.0198972, -0.8231956, 0.8102306
3: -0.0142010, 0.2818646, -0.0243124, 0.2818646, -0.2960657, 0.3061769
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0781890, 0.1868499, -0.0874891, 0.1868499, -0.2650389, 0.2743390
6: -0.1216488, 0.2890729, -0.1325822, 0.2890729, -0.4107217, 0.4216551
7: -0.0978745, 0.2871872, -0.1066403, 0.2871872, -0.3850617, 0.3938276
8: -0.0416816, 0.1999389, -0.0515544, 0.1999389, -0.2416205, 0.2514933
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A2_A1_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9726555, 0.1967014, 1.0102069, -0.8135054, 0.7759539
3: 0.0126072, 0.2818646, -0.0167550, 0.2818646, -0.2692575, 0.2986196
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0805381, 0.1868500, -0.2505312, 0.2673880
6: -0.0927558, 0.2890729, -0.1244104, 0.2890729, -0.3818287, 0.4134833
7: -0.0746333, 0.2871872, -0.1000886, 0.2871873, -0.3618206, 0.3872759
8: -0.0372933, 0.1999390, -0.0441752, 0.1999389, -0.2372322, 0.2441142
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B1_A2_A1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0231981, 0.1967014, 0.9967450, -0.8000435, 0.8264965
3: -0.0268868, 0.2818646, -0.0062512, 0.2818646, -0.3087513, 0.2881158
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0898570, 0.1868500, -0.0708770, 0.1868500, -0.2767069, 0.2577269
6: -0.1353659, 0.2890729, -0.1130589, 0.2890729, -0.4244388, 0.4021318
7: -0.1088723, 0.2871872, -0.0909824, 0.2871872, -0.3960595, 0.3781697
8: -0.0540680, 0.1999390, -0.0372933, 0.1999390, -0.2540070, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0137522, 0.1967014, 0.9626185, -0.7659172, 0.8170506
3: -0.0195201, 0.2818646, 0.0204690, 0.2818646, -0.3013847, 0.2613956
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0830813, 0.1868500, -0.0636812, 0.1868499, -0.2699312, 0.2505312
6: -0.1274004, 0.2890729, -0.0842917, 0.2890729, -0.4164732, 0.3733646
7: -0.1024857, 0.2871872, -0.0678176, 0.2871872, -0.3896729, 0.3550049
8: -0.0468751, 0.1999389, -0.0372933, 0.1999389, -0.2468140, 0.2372322
9: -0.1232193, 0.1760216, -0.1232192, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 211

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A2_B1_A2_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8968494, 0.1967014, 1.0237236, -0.8270220, 0.7001479
3: 0.0608644, 0.2818646, -0.0272965, 0.2818646, -0.2210002, 0.3091611
4: -0.2053552, 0.1310268, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0902338, 0.1868500, -0.2505312, 0.2770837
6: -0.0379278, 0.2890729, -0.1358090, 0.2890729, -0.3270006, 0.4248819
7: -0.0379298, 0.2871872, -0.1092273, 0.2871873, -0.3251171, 0.3964146
8: -0.0372933, 0.1999389, -0.0544681, 0.1999389, -0.2372322, 0.2544070
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 70

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007846
time: 1.12 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273828, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9140833, 0.1967014, 1.0247320, -0.8280306, 0.7173819
3: 0.0504888, 0.2818646, -0.0280832, 0.2818646, -0.2313759, 0.3099478
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0909574, 0.1868500, -0.2505312, 0.2778074
6: -0.0491156, 0.2890729, -0.1366597, 0.2890729, -0.3381885, 0.4257326
7: -0.0405730, 0.2871873, -0.1099095, 0.2871872, -0.3277603, 0.3970967
8: -0.0372934, 0.1999389, -0.0552363, 0.1999389, -0.2372323, 0.2551752
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007968
time: 1.12 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0941756, 0.1273828, -0.0911178, 0.1273828, -0.2215585, 0.2185007
2: 0.1967014, 1.0336711, 0.1967014, 1.0278919, -0.8311903, 0.8369694
3: -0.0350547, 0.2818646, -0.0305474, 0.2818646, -0.3169193, 0.3124120
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0973696, 0.1868500, -0.0932238, 0.1868499, -0.2842195, 0.2800738
6: -0.1441980, 0.2890729, -0.1393242, 0.2890729, -0.4332708, 0.4283971
7: -0.1159533, 0.2871873, -0.1120457, 0.2871872, -0.4031405, 0.3992330
8: -0.0620433, 0.1999389, -0.0576422, 0.1999389, -0.2619822, 0.2575811
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.91 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 1.0241792, 0.1967014, 0.9928026, -0.7961010, 0.8274775
3: -0.0276519, 0.2818646, -0.0031649, 0.2818646, -0.3095165, 0.2850295
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0905606, 0.1868500, -0.0680383, 0.1868500, -0.2774106, 0.2548883
6: -0.1361933, 0.2890729, -0.1097362, 0.2890729, -0.4252661, 0.3988091
7: -0.1095355, 0.2871872, -0.0883068, 0.2871872, -0.3967227, 0.3754941
8: -0.0548151, 0.1999389, -0.0372933, 0.1999389, -0.2547540, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 1.45 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8947107, 0.1967014, 1.0186323, -0.8219309, 0.6980093
3: 0.0621344, 0.2818646, -0.0233262, 0.2818646, -0.2197302, 0.3051908
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0865820, 0.1868499, -0.2505311, 0.2734319
6: -0.0365695, 0.2890729, -0.1315157, 0.2890729, -0.3256424, 0.4205886
7: -0.0379298, 0.2871872, -0.1057853, 0.2871873, -0.3251171, 0.3929725
8: -0.0372933, 0.1999389, -0.0505914, 0.1999389, -0.2372323, 0.2505303
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 70

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1989681, upper bound: 0.1989681
time: 0.94 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9117156, 0.1967014, 1.0196109, -0.8229094, 0.7150140
3: 0.0519143, 0.2818646, -0.0240891, 0.2818646, -0.2299503, 0.3059537
4: -0.2053552, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0872837, 0.1868499, -0.2505311, 0.2741337
6: -0.0475067, 0.2890729, -0.1323409, 0.2890729, -0.3365796, 0.4214138
7: -0.0394349, 0.2871872, -0.1064468, 0.2871872, -0.3266221, 0.3936341
8: -0.0372933, 0.1999389, -0.0513363, 0.1999389, -0.2372322, 0.2512753
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1989681, upper bound: 0.1989681
time: 0.95 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0926430, 0.1273829, -0.0911179, 0.1273829, -0.2200259, 0.2185008
2: 0.1967014, 1.0311956, 0.1967014, 1.0228343, -0.8261330, 0.8344941
3: -0.0331241, 0.2818647, -0.0266032, 0.2818646, -0.3149887, 0.3084678
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0955938, 0.1868500, -0.0895961, 0.1868500, -0.2824437, 0.2764460
6: -0.1421103, 0.2890729, -0.1350593, 0.2890729, -0.4311832, 0.4241322
7: -0.1142795, 0.2871872, -0.1086264, 0.2871872, -0.4014667, 0.3958136
8: -0.0601582, 0.1999389, -0.0537911, 0.1999389, -0.2600971, 0.2537300
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 217

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2003881, upper bound: 0.2009856
time: 0.89 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.99 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.58 seconds
NS_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009211, upper bound: 0.2009856
NS_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2007515, upper bound: 0.1989681
NS_A1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.1989681
NS_A1_B2_B1_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2007515, upper bound: 0.1989681
NS_A1_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2007738, upper bound: 0.1989681
NS_A1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.1989681
NS_A1_B2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2007738, upper bound: 0.1989681
NS_A1_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
NS_A1_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
NS_A1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
NS_A1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
NS_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007515
NS_A2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
NS_A2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
NS_A2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007738
NS_A2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
NS_A2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
NS_A2_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007846
NS_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
NS_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007968
NS_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
NS_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.1989681, upper bound: 0.1989681
NS_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
NS_A2_B2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.1989681, upper bound: 0.1989681
NS_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2009856
NS_A2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2003881, upper bound: 0.2009856
NS_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8797994, 0.1967014, 0.8655015, -0.6688000, 0.6830980
3: 0.0702252, 0.2818646, 0.0778735, 0.2818646, -0.2116394, 0.2039912
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505311
6: -0.0270687, 0.2890729, -0.0182637, 0.2890729, -0.3161416, 0.3073366
7: -0.0379298, 0.2871872, -0.0379298, 0.2871872, -0.3251170, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273828, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9918293, 0.1967014, 0.8655015, -0.6688000, 0.7951277
3: -0.0024030, 0.2818646, 0.0778735, 0.2818646, -0.2842677, 0.2039912
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0673375, 0.1868499, -0.0636812, 0.1868500, -0.2541875, 0.2505311
6: -0.1089159, 0.2890729, -0.0182637, 0.2890729, -0.3979888, 0.3073366
7: -0.0876462, 0.2871873, -0.0379298, 0.2871872, -0.3748335, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 70

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.8806850, 0.1967014, 0.8822462, -0.6855446, 0.6839835
3: 0.0697516, 0.2818646, 0.0689165, 0.2818646, -0.2121130, 0.2129481
4: -0.2053552, 0.1310268, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0276406, 0.2890729, -0.0286486, 0.2890729, -0.3167135, 0.3177215
7: -0.0379298, 0.2871872, -0.0379298, 0.2871873, -0.3251171, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967013, 0.9928572, 0.1967014, 0.8822462, -0.6855447, 0.7961558
3: -0.0032078, 0.2818646, 0.0689165, 0.2818646, -0.2850724, 0.2129481
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0680777, 0.1868500, -0.0636812, 0.1868500, -0.2549276, 0.2505312
6: -0.1097823, 0.2890729, -0.0286486, 0.2890729, -0.3988552, 0.3177215
7: -0.0883440, 0.2871873, -0.0379298, 0.2871873, -0.3755312, 0.3251171
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 94

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 0.9933610, 0.1967014, 0.9930869, -0.7963853, 0.7966596
3: -0.0036024, 0.2818646, -0.0033875, 0.2818646, -0.2854670, 0.2852522
4: -0.2053551, 0.1310268, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0684406, 0.1868499, -0.0682430, 0.1868499, -0.2552905, 0.2550929
6: -0.1102071, 0.2890729, -0.1099758, 0.2890729, -0.3992799, 0.3990487
7: -0.0886860, 0.2871872, -0.0884998, 0.2871873, -0.3758732, 0.3756870
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760215, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 217

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 0.9933610, 0.1967014, 0.9885262, -0.7918245, 0.7966596
3: -0.0036024, 0.2818646, 0.0001830, 0.2818646, -0.2854670, 0.2816817
4: -0.2053551, 0.1310268, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0684406, 0.1868499, -0.0649591, 0.1868499, -0.2552905, 0.2518090
6: -0.1102071, 0.2890729, -0.1061318, 0.2890729, -0.3992799, 0.3952047
7: -0.0886860, 0.2871872, -0.0854044, 0.2871872, -0.3758732, 0.3725916
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760215, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 217

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9403793, 0.1967014, 0.9882724, -0.7915709, 0.7436779
3: 0.0346570, 0.2818646, 0.0003815, 0.2818646, -0.2472076, 0.2814831
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0647764, 0.1868500, -0.2505312, 0.2516264
6: -0.0669824, 0.2890729, -0.1059180, 0.2890729, -0.3560553, 0.3949909
7: -0.0534942, 0.2871872, -0.0852323, 0.2871872, -0.3406815, 0.3724195
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 211

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9556425, 0.1967014, 0.9891971, -0.7924957, 0.7589408
3: 0.0254679, 0.2818646, -0.0003425, 0.2818646, -0.2563968, 0.2822072
4: -0.2053551, 0.1310268, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0654423, 0.1868499, -0.2505312, 0.2522923
6: -0.0786130, 0.2890729, -0.1066974, 0.2890729, -0.3676859, 0.3957704
7: -0.0630760, 0.2871872, -0.0858599, 0.2871872, -0.3502632, 0.3730471
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.8759586, 0.1967014, 0.8633748, -0.6666733, 0.6792570
3: 0.0722798, 0.2818646, 0.0790112, 0.2818646, -0.2095848, 0.2028534
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505311, 0.2505312
6: -0.0246742, 0.2890729, -0.0170335, 0.2890729, -0.3137471, 0.3061063
7: -0.0379298, 0.2871872, -0.0379298, 0.2871873, -0.3251171, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.9872615, 0.1967014, 0.8633748, -0.6666734, 0.7905599
3: 0.0011728, 0.2818646, 0.0790112, 0.2818646, -0.2806918, 0.2028534
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363821
5: -0.0640486, 0.1868499, -0.0636812, 0.1868500, -0.2508985, 0.2505311
6: -0.1050661, 0.2890729, -0.0170335, 0.2890729, -0.3941390, 0.3061064
7: -0.0845463, 0.2871872, -0.0379298, 0.2871873, -0.3717335, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 70

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273828, -0.0911178, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.8768352, 0.1967014, 0.8800996, -0.6833981, 0.6801335
3: 0.0718110, 0.2818646, 0.0700646, 0.2818646, -0.2100536, 0.2118000
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505311, 0.2505311
6: -0.0252116, 0.2890729, -0.0272626, 0.2890729, -0.3142845, 0.3163355
7: -0.0379298, 0.2871873, -0.0379298, 0.2871872, -0.3251170, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273828, -0.0911178, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.9882873, 0.1967014, 0.8800996, -0.6833982, 0.7915856
3: 0.0003698, 0.2818646, 0.0700646, 0.2818646, -0.2814949, 0.2118000
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0647871, 0.1868499, -0.0636812, 0.1868500, -0.2516370, 0.2505311
6: -0.1059306, 0.2890729, -0.0272626, 0.2890729, -0.3950035, 0.3163355
7: -0.0852424, 0.2871872, -0.0379298, 0.2871872, -0.3724296, 0.3251170
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372323, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279924, 0.2164623, -0.3444547, 0.3444547
1: -0.1198041, 0.1273829, -0.0911179, 0.1273829, -0.2471870, 0.2185007
2: 0.1967014, 1.0762050, 0.1967014, 0.9959603, -0.7992587, 0.8795034
3: -0.0684561, 0.2818647, -0.0056369, 0.2818646, -0.3503206, 0.2875015
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.1280911, 0.1868499, -0.0703119, 0.1868500, -0.3149410, 0.2571619
6: -0.1800289, 0.2890729, -0.1123974, 0.2890729, -0.4691018, 0.4014703
7: -0.1449102, 0.2871873, -0.0904498, 0.2871872, -0.4320974, 0.3776371
8: -0.0930101, 0.1999389, -0.0372933, 0.1999389, -0.2929490, 0.2372323
9: -0.1363897, 0.1760216, -0.1232193, 0.1760216, -0.3124112, 0.2992409

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2006467, upper bound: 0.2009856
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009211, upper bound: 0.2009856
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9883142, 0.1967014, 0.9965099, -0.7998084, 0.7916126
3: 0.0003489, 0.2818646, -0.0060671, 0.2818646, -0.2815157, 0.2879318
4: -0.2053552, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0648065, 0.1868499, -0.0707076, 0.1868500, -0.2516564, 0.2575576
6: -0.1059532, 0.2890729, -0.1128606, 0.2890729, -0.3950261, 0.4019335
7: -0.0852606, 0.2871872, -0.0908228, 0.2871872, -0.3724479, 0.3780101
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9372965, 0.1967014, 0.9857109, -0.7890095, 0.7405951
3: 0.0365130, 0.2818646, 0.0023867, 0.2818646, -0.2453516, 0.2794779
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505311, 0.2505312
6: -0.0648879, 0.2890729, -0.1037593, 0.2890729, -0.3539608, 0.3928322
7: -0.0519567, 0.2871872, -0.0834939, 0.2871873, -0.3391439, 0.3706812
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 211

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273828, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9519273, 0.1967014, 0.9866276, -0.7899261, 0.7552257
3: 0.0277045, 0.2818646, 0.0016691, 0.2818646, -0.2541601, 0.2801955
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868499, -0.2505312, 0.2505312
6: -0.0757815, 0.2890729, -0.1045318, 0.2890729, -0.3648544, 0.3936046
7: -0.0607427, 0.2871872, -0.0841159, 0.2871873, -0.3479300, 0.3713032
8: -0.0372933, 0.1999389, -0.0372933, 0.1999390, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8842952, 0.1967014, 0.8932446, -0.6965431, 0.6875938
3: 0.0678203, 0.2818646, 0.0629645, 0.2818646, -0.2140444, 0.2189001
4: -0.2053552, 0.1310268, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505311, 0.2505312
6: -0.0299719, 0.2890729, -0.0356741, 0.2890729, -0.3190448, 0.3247469
7: -0.0379298, 0.2871873, -0.0379298, 0.2871872, -0.3251170, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2007515, upper bound: 0.1989681
time: 1.23 seconds

## Relational analysis of NS_A1_B2_B1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2007515, upper bound: 0.1989681
time: 2.71 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273828, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9918293, 0.1967014, 0.8968494, -0.7001479, 0.7951276
3: -0.0024030, 0.2818646, 0.0608644, 0.2818646, -0.2842676, 0.2210002
4: -0.2053551, 0.1310269, -0.2053552, 0.1310268, -0.3363820, 0.3363820
5: -0.0673375, 0.1868499, -0.0636812, 0.1868500, -0.2541875, 0.2505311
6: -0.1089159, 0.2890729, -0.0379278, 0.2890729, -0.3979888, 0.3270006
7: -0.0876462, 0.2871873, -0.0379298, 0.2871872, -0.3748335, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 94

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_B1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.1986192
time: 1.02 seconds

## Relational analysis of NS_A1_B2_B1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.1986192
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273828, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8851618, 0.1967014, 0.9089901, -0.7122884, 0.6884602
3: 0.0673567, 0.2818646, 0.0535552, 0.2818646, -0.2145079, 0.2283095
4: -0.2053551, 0.1310268, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0305315, 0.2890729, -0.0457265, 0.2890729, -0.3196044, 0.3347993
7: -0.0379298, 0.2871873, -0.0381249, 0.2871873, -0.3251171, 0.3253122
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B2_B1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.18 seconds

## Relational analysis of NS_A1_B2_B1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967013, 0.9928572, 0.1967014, 0.9140833, -0.7173819, 0.7961558
3: -0.0032078, 0.2818646, 0.0504888, 0.2818646, -0.2850725, 0.2313759
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0680777, 0.1868500, -0.0636812, 0.1868500, -0.2549276, 0.2505312
6: -0.1097823, 0.2890729, -0.0491156, 0.2890729, -0.3988553, 0.3381884
7: -0.0883440, 0.2871873, -0.0405730, 0.2871873, -0.3755312, 0.3277603
8: -0.0372933, 0.1999390, -0.0372934, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_B1_B2_A2_A1_A1

### Relational analysis result of NS_A1_B2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.1986192
time: 1.30 seconds

## Relational analysis of NS_A1_B2_B1_B2_A2_A1_A2

### Relational analysis result of NS_A1_B2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.1986192
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185008, 0.2185008
2: 0.1967013, 1.0138386, 0.1967014, 1.0069320, -0.8102305, 0.8171372
3: -0.0195876, 0.2818646, -0.0142010, 0.2818646, -0.3014521, 0.2960657
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0831433, 0.1868500, -0.0781890, 0.1868499, -0.2699933, 0.2650390
6: -0.1274733, 0.2890729, -0.1216488, 0.2890729, -0.4165461, 0.4107217
7: -0.1025442, 0.2871873, -0.0978745, 0.2871872, -0.3897314, 0.3850617
8: -0.0469409, 0.1999389, -0.0416816, 0.1999389, -0.2468798, 0.2416205
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 217

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_A1

### Relational analysis result of NS_A1_B2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_A2

### Relational analysis result of NS_A1_B2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 1.0091932, 0.1967014, 1.0069320, -0.8102305, 0.8124917
3: -0.0159645, 0.2818646, -0.0142010, 0.2818646, -0.2978291, 0.2960657
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0798110, 0.1868499, -0.0781890, 0.1868499, -0.2666609, 0.2650389
6: -0.1235557, 0.2890729, -0.1216488, 0.2890729, -0.4126286, 0.4107217
7: -0.0994033, 0.2871873, -0.0978745, 0.2871872, -0.3865905, 0.3850617
8: -0.0434034, 0.1999390, -0.0416816, 0.1999389, -0.2433423, 0.2416205
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 217

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 217

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A1

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.17 seconds

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A2

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0090613, 0.1967014, 0.9561567, -0.7594547, 0.8123598
3: -0.0158616, 0.2818646, 0.0251584, 0.2818646, -0.2977263, 0.2567062
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0797164, 0.1868500, -0.0636812, 0.1868500, -0.2665663, 0.2505312
6: -0.1234444, 0.2890729, -0.0790050, 0.2890729, -0.4125173, 0.3680779
7: -0.0993141, 0.2871873, -0.0633988, 0.2871872, -0.3865013, 0.3505861
8: -0.0433029, 0.1999389, -0.0372933, 0.1999389, -0.2432418, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A1_B2_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_B2_B1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185008
2: 0.1967014, 1.0099928, 0.1967013, 0.9702570, -0.7735555, 0.8132915
3: -0.0165884, 0.2818646, 0.0144845, 0.2818646, -0.2984529, 0.2673801
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0803847, 0.1868499, -0.0636812, 0.1868499, -0.2672347, 0.2505311
6: -0.1242302, 0.2890729, -0.0907347, 0.2890729, -0.4133030, 0.3798076
7: -0.0999441, 0.2871872, -0.0730058, 0.2871872, -0.3871313, 0.3601930
8: -0.0440125, 0.1999389, -0.0372933, 0.1999389, -0.2439514, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 194

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_B2_B1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9967450, 0.1967014, 1.0022207, -0.8055191, 0.8000435
3: -0.0062512, 0.2818646, -0.0105268, 0.2818646, -0.2881158, 0.2923914
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0708770, 0.1868500, -0.0748095, 0.1868500, -0.2577270, 0.2616594
6: -0.1130589, 0.2890729, -0.1176759, 0.2890729, -0.4021317, 0.4067488
7: -0.0909824, 0.2871872, -0.0946891, 0.2871872, -0.3781697, 0.3818763
8: -0.0372933, 0.1999390, -0.0380939, 0.1999389, -0.2372323, 0.2380329
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 217

### Candidate
type: B, layer: 1, pos: 217

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
time: 1.09 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 0.9967450, 0.1967014, 0.9679593, -0.7712579, 0.8000435
3: -0.0062512, 0.2818646, 0.0162834, 0.2818646, -0.2881159, 0.2655813
4: -0.2053551, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0708770, 0.1868500, -0.0636812, 0.1868499, -0.2577269, 0.2505312
6: -0.1130589, 0.2890729, -0.0887979, 0.2890729, -0.4021318, 0.3778709
7: -0.0909824, 0.2871872, -0.0714463, 0.2871872, -0.3781697, 0.3586335
8: -0.0372933, 0.1999390, -0.0372933, 0.1999390, -0.2372323, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2002387
time: 1.11 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273828, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9427301, 0.1967014, 1.0125285, -0.8158270, 0.7460285
3: 0.0332419, 0.2818646, -0.0185658, 0.2818646, -0.2486228, 0.3004304
4: -0.2053551, 0.1310268, -0.2053552, 0.1310268, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0822035, 0.1868500, -0.2505312, 0.2690535
6: -0.0687713, 0.2890729, -0.1263684, 0.2890729, -0.3578442, 0.4154412
7: -0.0549662, 0.2871872, -0.1016584, 0.2871873, -0.3421535, 0.3888456
8: -0.0372933, 0.1999390, -0.0459432, 0.1999390, -0.2372323, 0.2458821
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 194

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 217

## Relational analysis of NS_A1_B2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
time: 0.93 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185008
2: 0.1967015, 0.9589968, 0.1967014, 1.0135052, -0.8168037, 0.7622954
3: 0.0232568, 0.2818646, -0.0193275, 0.2818646, -0.2586078, 0.3011921
4: -0.2053551, 0.1310268, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0829041, 0.1868500, -0.2505312, 0.2697540
6: -0.0812445, 0.2890729, -0.1271920, 0.2890729, -0.3703174, 0.4162649
7: -0.0653415, 0.2871872, -0.1023187, 0.2871872, -0.3525288, 0.3895059
8: -0.0372933, 0.1999390, -0.0466870, 0.1999389, -0.2372323, 0.2466259
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 211

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 217

## Relational analysis of NS_A1_B2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_B2_B2_A2_A2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004954
time: 0.87 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_A2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004954
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8932446, 0.1967014, 0.8842952, -0.6875938, 0.6965431
3: 0.0629645, 0.2818646, 0.0678203, 0.2818646, -0.2189001, 0.2140444
4: -0.2053552, 0.1310269, -0.2053552, 0.1310268, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505311
6: -0.0356741, 0.2890729, -0.0299719, 0.2890729, -0.3247470, 0.3190448
7: -0.0379298, 0.2871872, -0.0379298, 0.2871873, -0.3251171, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007515
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007515
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.8968494, 0.1967014, 0.9918293, -0.7951277, 0.7001479
3: 0.0608644, 0.2818646, -0.0024030, 0.2818646, -0.2210002, 0.2842676
4: -0.2053552, 0.1310268, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0673375, 0.1868499, -0.2505311, 0.2541875
6: -0.0379278, 0.2890729, -0.1089159, 0.2890729, -0.3270006, 0.3979888
7: -0.0379298, 0.2871872, -0.0876462, 0.2871873, -0.3251171, 0.3748335
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8947107, 0.1967014, 0.9872615, -0.7905600, 0.6980093
3: 0.0621344, 0.2818646, 0.0011728, 0.2818646, -0.2197302, 0.2806918
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0640486, 0.1868499, -0.2505311, 0.2508985
6: -0.0365695, 0.2890729, -0.1050661, 0.2890729, -0.3256424, 0.3941390
7: -0.0379298, 0.2871872, -0.0845463, 0.2871872, -0.3251171, 0.3717335
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 70

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.9089901, 0.1967014, 0.8851618, -0.6884601, 0.7122885
3: 0.0535552, 0.2818646, 0.0673567, 0.2818646, -0.2283095, 0.2145079
4: -0.2053552, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0457265, 0.2890729, -0.0305315, 0.2890729, -0.3347993, 0.3196044
7: -0.0381249, 0.2871873, -0.0379298, 0.2871873, -0.3253122, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A2_B1_A1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.28 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.19 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273828, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9140833, 0.1967013, 0.9928572, -0.7961558, 0.7173819
3: 0.0504888, 0.2818646, -0.0032078, 0.2818646, -0.2313759, 0.2850725
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0680777, 0.1868500, -0.2505312, 0.2549276
6: -0.0491156, 0.2890729, -0.1097823, 0.2890729, -0.3381884, 0.3988553
7: -0.0405730, 0.2871873, -0.0883440, 0.2871873, -0.3277603, 0.3755312
8: -0.0372934, 0.1999389, -0.0372933, 0.1999390, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.9117156, 0.1967014, 0.9882873, -0.7915857, 0.7150140
3: 0.0519143, 0.2818646, 0.0003698, 0.2818646, -0.2299503, 0.2814948
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0647871, 0.1868499, -0.2505311, 0.2516370
6: -0.0475067, 0.2890729, -0.1059306, 0.2890729, -0.3365796, 0.3950036
7: -0.0394349, 0.2871872, -0.0852424, 0.2871872, -0.3266221, 0.3724296
8: -0.0372933, 0.1999389, -0.0372933, 0.1999390, -0.2372323, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 217

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185008, 0.2185008
2: 0.1967014, 1.0069320, 0.1967013, 1.0138386, -0.8171372, 0.8102306
3: -0.0142010, 0.2818646, -0.0195876, 0.2818646, -0.2960657, 0.3014522
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0781890, 0.1868499, -0.0831433, 0.1868500, -0.2650390, 0.2699933
6: -0.1216488, 0.2890729, -0.1274733, 0.2890729, -0.4107217, 0.4165461
7: -0.0978745, 0.2871872, -0.1025442, 0.2871873, -0.3850617, 0.3897314
8: -0.0416816, 0.1999389, -0.0469409, 0.1999389, -0.2416205, 0.2468798
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 217

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 217

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0069320, 0.1967013, 1.0091932, -0.8124917, 0.8102306
3: -0.0142010, 0.2818646, -0.0159645, 0.2818646, -0.2960657, 0.2978291
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0781890, 0.1868499, -0.0798110, 0.1868499, -0.2650389, 0.2666609
6: -0.1216488, 0.2890729, -0.1235557, 0.2890729, -0.4107217, 0.4126286
7: -0.0978745, 0.2871872, -0.0994033, 0.2871873, -0.3850617, 0.3865905
8: -0.0416816, 0.1999389, -0.0434034, 0.1999390, -0.2416205, 0.2433423
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 217

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9561567, 0.1967014, 1.0090613, -0.8123598, 0.7594547
3: 0.0251584, 0.2818646, -0.0158616, 0.2818646, -0.2567062, 0.2977263
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0797164, 0.1868500, -0.2505312, 0.2665663
6: -0.0790050, 0.2890729, -0.1234444, 0.2890729, -0.3680779, 0.4125173
7: -0.0633988, 0.2871872, -0.0993141, 0.2871873, -0.3505861, 0.3865013
8: -0.0372933, 0.1999389, -0.0433029, 0.1999389, -0.2372323, 0.2432418
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 194

### Candidate
type: A, layer: 1, pos: 217

## Relational analysis of NS_A2_B1_A2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_A1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273828, -0.0911179, 0.1273829, -0.2185008, 0.2185007
2: 0.1967013, 0.9702570, 0.1967014, 1.0099928, -0.8132915, 0.7735556
3: 0.0144845, 0.2818646, -0.0165884, 0.2818646, -0.2673801, 0.2984529
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0803847, 0.1868499, -0.2505311, 0.2672347
6: -0.0907347, 0.2890729, -0.1242302, 0.2890729, -0.3798076, 0.4133030
7: -0.0730058, 0.2871872, -0.0999441, 0.2871872, -0.3601930, 0.3871313
8: -0.0372933, 0.1999389, -0.0440125, 0.1999389, -0.2372323, 0.2439514
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 194

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A2_A1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0022207, 0.1967014, 0.9967450, -0.8000435, 0.8055193
3: -0.0105268, 0.2818646, -0.0062512, 0.2818646, -0.2923914, 0.2881158
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0748095, 0.1868500, -0.0708770, 0.1868500, -0.2616594, 0.2577270
6: -0.1176759, 0.2890729, -0.1130589, 0.2890729, -0.4067488, 0.4021317
7: -0.0946891, 0.2871872, -0.0909824, 0.2871872, -0.3818763, 0.3781697
8: -0.0380939, 0.1999389, -0.0372933, 0.1999390, -0.2380329, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 217

### Candidate
type: A, layer: 1, pos: 217

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185008
2: 0.1967014, 0.9679593, 0.1967014, 0.9967450, -0.8000435, 0.7712579
3: 0.0162834, 0.2818646, -0.0062512, 0.2818646, -0.2655813, 0.2881159
4: -0.2053551, 0.1310268, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0708770, 0.1868500, -0.2505312, 0.2577269
6: -0.0887979, 0.2890729, -0.1130589, 0.2890729, -0.3778709, 0.4021318
7: -0.0714463, 0.2871872, -0.0909824, 0.2871872, -0.3586335, 0.3781697
8: -0.0372933, 0.1999390, -0.0372933, 0.1999390, -0.2372323, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2002387, upper bound: 0.2009856
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911178, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 1.0125285, 0.1967014, 0.9427301, -0.7460285, 0.8158270
3: -0.0185658, 0.2818646, 0.0332419, 0.2818646, -0.3004304, 0.2486228
4: -0.2053552, 0.1310268, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0822035, 0.1868500, -0.0636812, 0.1868500, -0.2690535, 0.2505312
6: -0.1263684, 0.2890729, -0.0687713, 0.2890729, -0.4154412, 0.3578442
7: -0.1016584, 0.2871873, -0.0549662, 0.2871872, -0.3888456, 0.3421535
8: -0.0459432, 0.1999390, -0.0372933, 0.1999390, -0.2458821, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273828, -0.0911179, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 1.0135052, 0.1967015, 0.9589968, -0.7622955, 0.8168038
3: -0.0193275, 0.2818646, 0.0232568, 0.2818646, -0.3011921, 0.2586078
4: -0.2053551, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0829041, 0.1868500, -0.0636812, 0.1868499, -0.2697540, 0.2505312
6: -0.1271920, 0.2890729, -0.0812445, 0.2890729, -0.4162649, 0.3703174
7: -0.1023187, 0.2871872, -0.0653415, 0.2871872, -0.3895059, 0.3525288
8: -0.0466870, 0.1999389, -0.0372933, 0.1999390, -0.2466259, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 211

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004954, upper bound: 0.2009856
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004954, upper bound: 0.2009856
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8968494, 0.1967014, 1.0265564, -0.8298545, 0.7001479
3: 0.0608644, 0.2818646, -0.0295058, 0.2818646, -0.2210002, 0.3113704
4: -0.2053552, 0.1310268, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0922658, 0.1868499, -0.2505311, 0.2791158
6: -0.0379278, 0.2890729, -0.1381978, 0.2890729, -0.3270007, 0.4272707
7: -0.0379298, 0.2871872, -0.1111426, 0.2871873, -0.3251171, 0.3983299
8: -0.0372933, 0.1999389, -0.0566252, 0.1999390, -0.2372323, 0.2565641
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 70

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 217

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 1.01 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273828, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9140833, 0.1967014, 0.9120885, -0.7153869, 0.7173817
3: 0.0504888, 0.2818646, 0.0516897, 0.2818646, -0.2313759, 0.2301750
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505311, 0.2505312
6: -0.0491156, 0.2890729, -0.0477602, 0.2890729, -0.3381884, 0.3368331
7: -0.0405730, 0.2871873, -0.0396142, 0.2871872, -0.3277603, 0.3268015
8: -0.0372934, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_B1_A1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007968
time: 1.07 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007968
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273828, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9140833, 0.1967014, 1.0275760, -0.8308742, 0.7173818
3: 0.0504888, 0.2818646, -0.0303011, 0.2818646, -0.2313759, 0.3121657
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0929972, 0.1868500, -0.2505312, 0.2798472
6: -0.0491156, 0.2890729, -0.1390578, 0.2890729, -0.3381884, 0.4281307
7: -0.0405730, 0.2871873, -0.1118321, 0.2871872, -0.3277603, 0.3990194
8: -0.0372934, 0.1999389, -0.0574018, 0.1999389, -0.2372323, 0.2573407
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 1.06 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911178, 0.1273829, -0.0911178, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 1.0278074, 0.1967014, 1.0278919, -0.8311903, 0.8311058
3: -0.0304814, 0.2818646, -0.0305474, 0.2818646, -0.3123460, 0.3124120
4: -0.2053552, 0.1310268, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0931633, 0.1868499, -0.0932238, 0.1868499, -0.2800132, 0.2800737
6: -0.1392529, 0.2890729, -0.1393242, 0.2890729, -0.4283258, 0.4283971
7: -0.1119886, 0.2871873, -0.1120457, 0.2871872, -0.3991759, 0.3992330
8: -0.0575780, 0.1999390, -0.0576422, 0.1999389, -0.2575169, 0.2575812
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 217

### Candidate
type: B, layer: 1, pos: 217

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of NS_A2_B2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 124

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 1.08 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 1.0231981, 0.1967014, 1.0278919, -0.8311903, 0.8264967
3: -0.0268868, 0.2818646, -0.0305474, 0.2818646, -0.3087514, 0.3124120
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0898570, 0.1868500, -0.0932238, 0.1868499, -0.2767068, 0.2800738
6: -0.1353659, 0.2890729, -0.1393242, 0.2890729, -0.4244388, 0.4283971
7: -0.1088723, 0.2871872, -0.1120457, 0.2871872, -0.3960595, 0.3992329
8: -0.0540680, 0.1999390, -0.0576422, 0.1999389, -0.2540070, 0.2575812
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 217

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 217

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 124

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2000602, upper bound: 0.2009856
time: 1.18 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2000602, upper bound: 0.2009856
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 1.0183117, 0.1967014, 0.9928026, -0.7961010, 0.8216101
3: -0.0230759, 0.2818646, -0.0031649, 0.2818646, -0.3049405, 0.2850295
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0863519, 0.1868500, -0.0680383, 0.1868500, -0.2732019, 0.2548883
6: -0.1312452, 0.2890729, -0.1097362, 0.2890729, -0.4203181, 0.3988091
7: -0.1055684, 0.2871872, -0.0883068, 0.2871872, -0.3927557, 0.3754941
8: -0.0503471, 0.1999389, -0.0372933, 0.1999389, -0.2502860, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 217

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A2_B2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.98 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 1.0136454, 0.1967014, 0.9928026, -0.7961010, 0.8169440
3: -0.0194368, 0.2818646, -0.0031649, 0.2818646, -0.3013014, 0.2850295
4: -0.2053551, 0.1310268, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0830047, 0.1868499, -0.0680383, 0.1868500, -0.2698546, 0.2548882
6: -0.1273103, 0.2890729, -0.1097362, 0.2890729, -0.4163832, 0.3988090
7: -0.1024135, 0.2871872, -0.0883068, 0.2871872, -0.3896008, 0.3754940
8: -0.0467937, 0.1999389, -0.0372933, 0.1999389, -0.2467327, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 217

### Candidate
type: A, layer: 1, pos: 217

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 0.96 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8947107, 0.1967014, 1.0219506, -0.8252490, 0.6980093
3: 0.0621344, 0.2818646, -0.0259139, 0.2818646, -0.2197302, 0.3077785
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0889621, 0.1868500, -0.2505312, 0.2758120
6: -0.0365695, 0.2890729, -0.1343139, 0.2890729, -0.3256424, 0.4233868
7: -0.0379298, 0.2871872, -0.1080288, 0.2871873, -0.3251171, 0.3952160
8: -0.0372933, 0.1999389, -0.0531181, 0.1999389, -0.2372322, 0.2530570
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 70

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_B1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 1.04 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_B2

### Relational analysis result of NS_A2_B2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9117156, 0.1967014, 1.0229504, -0.8262486, 0.7150140
3: 0.0519143, 0.2818646, -0.0266933, 0.2818646, -0.2299503, 0.3085579
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0896790, 0.1868500, -0.2505312, 0.2765290
6: -0.0475067, 0.2890729, -0.1351568, 0.2890729, -0.3365796, 0.4242297
7: -0.0394349, 0.2871872, -0.1087045, 0.2871872, -0.3266221, 0.3958918
8: -0.0372933, 0.1999389, -0.0538792, 0.1999389, -0.2372322, 0.2538181
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 217

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 0.93 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0917917, 0.1273829, -0.0911179, 0.1273829, -0.2191745, 0.2185008
2: 0.1967014, 1.0298204, 0.1967014, 1.0277464, -0.8310448, 0.8331189
3: -0.0320514, 0.2818646, -0.0304340, 0.2818646, -0.3139160, 0.3122986
4: -0.2053551, 0.1310269, -0.2053552, 0.1310268, -0.3363820, 0.3363820
5: -0.0946072, 0.1868499, -0.0931195, 0.1868499, -0.2814572, 0.2799694
6: -0.1409504, 0.2890729, -0.1392016, 0.2890729, -0.4300234, 0.4282744
7: -0.1133496, 0.2871873, -0.1119474, 0.2871872, -0.4005369, 0.3991347
8: -0.0591109, 0.1999389, -0.0575316, 0.1999389, -0.2590498, 0.2574705
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 217

### Candidate
type: A, layer: 1, pos: 217

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2003881, upper bound: 0.2009856
time: 1.01 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2003881, upper bound: 0.2009856
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0926430, 0.1273829, -0.0911179, 0.1273829, -0.2200259, 0.2185007
2: 0.1967014, 1.0311956, 0.1967014, 1.0212241, -0.8245225, 0.8344942
3: -0.0331241, 0.2818647, -0.0253471, 0.2818646, -0.3149887, 0.3072117
4: -0.2053551, 0.1310269, -0.2053552, 0.1310268, -0.3363820, 0.3363820
5: -0.0955938, 0.1868500, -0.0884408, 0.1868500, -0.2824437, 0.2752908
6: -0.1421103, 0.2890729, -0.1337012, 0.2890729, -0.4311832, 0.4227740
7: -0.1142795, 0.2871872, -0.1075375, 0.2871872, -0.4014667, 0.3947247
8: -0.0601582, 0.1999389, -0.0525647, 0.1999389, -0.2600971, 0.2525036
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 217

### Candidate
type: A, layer: 1, pos: 217

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 1.00 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
time: 1.05 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 8.37 seconds
NS_A1_B1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2006467, upper bound: 0.2009856
NS_A1_B1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009211, upper bound: 0.2009856
NS_A1_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2007515, upper bound: 0.1989681
NS_A1_B2_B1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2007515, upper bound: 0.1989681
NS_A1_B2_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.1986192
NS_A1_B2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.1986192
NS_A1_B2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.1986192
NS_A1_B2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.1986192
NS_A1_B2_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
NS_A1_B2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
NS_A1_B2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2002387
NS_A1_B2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
NS_A1_B2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
NS_A1_B2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004961
NS_A1_B2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004954
NS_A1_B2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2004954
NS_A2_B1_A1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007515
NS_A2_B1_A1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007515
NS_A2_B1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B1_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B1_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B1_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B1_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2002387, upper bound: 0.2009856
NS_A2_B1_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B1_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2004954, upper bound: 0.2009856
NS_A2_B1_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2004954, upper bound: 0.2009856
NS_A2_B2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007968
NS_A2_B2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1989681, upper bound: 0.2007968
NS_A2_B2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2000602, upper bound: 0.2009856
NS_A2_B2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2000602, upper bound: 0.2009856
NS_A2_B2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.1986192, upper bound: 0.2009856
NS_A2_B2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2003881, upper bound: 0.2009856
NS_A2_B2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2003881, upper bound: 0.2009856
NS_A2_B2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856
NS_A2_B2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 3, lower bound: -0.2004961, upper bound: 0.2009856

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8797994, 0.1967014, 0.8608515, -0.6641501, 0.6830980
3: 0.0702252, 0.2818646, 0.0803609, 0.2818646, -0.2116394, 0.2015037
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505311
6: -0.0270687, 0.2890729, -0.0155802, 0.2890729, -0.3161416, 0.3046531
7: -0.0379298, 0.2871872, -0.0379298, 0.2871872, -0.3251170, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8797994, 0.1967014, 0.8573989, -0.6606975, 0.6830980
3: 0.0702252, 0.2818646, 0.0822078, 0.2818646, -0.2116394, 0.1996569
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0270687, 0.2890729, -0.0135914, 0.2890729, -0.3161417, 0.3026643
7: -0.0379298, 0.2871872, -0.0379298, 0.2871873, -0.3251171, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 60

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2006478
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2006478
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.9556608, 0.1967014, 0.8646574, -0.6679559, 0.7589592
3: 0.0254566, 0.2818646, 0.0783251, 0.2818646, -0.2564080, 0.2035396
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505311, 0.2505312
6: -0.0786273, 0.2890729, -0.0177724, 0.2890729, -0.3677001, 0.3068453
7: -0.0630876, 0.2871872, -0.0379298, 0.2871872, -0.3502749, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 70

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 60

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2003073
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2003073
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.9697925, 0.1967014, 0.8615644, -0.6648629, 0.7730911
3: 0.0148482, 0.2818646, 0.0799797, 0.2818646, -0.2670164, 0.2018850
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505311, 0.2505312
6: -0.0903431, 0.2890729, -0.0159907, 0.2890729, -0.3794160, 0.3050635
7: -0.0726906, 0.2871872, -0.0379298, 0.2871872, -0.3598778, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232192, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 70

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 60

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2003073
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2003073
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273828, -0.2185007, 0.2185007
2: 0.1967014, 0.8806850, 0.1967014, 0.8777658, -0.6810644, 0.6839836
3: 0.0697516, 0.2818646, 0.0713131, 0.2818646, -0.2121131, 0.2105515
4: -0.2053552, 0.1310268, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505312, 0.2505312
6: -0.0276406, 0.2890729, -0.0257823, 0.2890729, -0.3167135, 0.3148552
7: -0.0379298, 0.2871872, -0.0379298, 0.2871872, -0.3251170, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.64 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.95 + 596.60 = 601.55 seconds
