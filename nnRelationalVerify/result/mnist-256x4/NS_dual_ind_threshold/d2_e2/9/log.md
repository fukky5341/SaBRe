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
execution time: IAR + RelationalAnalysis = 1.77 + 2.35 = 4.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.80 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.80
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.80
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9996660, 0.1967014, 1.0213277, -0.8246263, 0.8029646
3: -0.0085344, 0.2818646, -0.0254283, 0.2818646, -0.2903989, 0.3072928
4: -0.2053551, 0.1310268, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0729769, 0.1868500, -0.0885154, 0.1868499, -0.2598268, 0.2753653
6: -0.1155213, 0.2890729, -0.1337888, 0.2890729, -0.4045942, 0.4228616
7: -0.0929617, 0.2871873, -0.1076077, 0.2871873, -0.3801490, 0.3947950
8: -0.0372933, 0.1999389, -0.0526439, 0.1999389, -0.2372322, 0.2525828
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0190661, 0.1967014, 1.0223223, -0.8256208, 0.8223646
3: -0.0236642, 0.2818646, -0.0262038, 0.2818646, -0.3055288, 0.3080683
4: -0.2053551, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0868930, 0.1868500, -0.0892287, 0.1868500, -0.2737429, 0.2760786
6: -0.1318814, 0.2890729, -0.1346276, 0.2890729, -0.4209543, 0.4237003
7: -0.1060784, 0.2871873, -0.1082801, 0.2871873, -0.3932657, 0.3954673
8: -0.0509215, 0.1999389, -0.0534011, 0.1999389, -0.2508604, 0.2533400
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.52 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9996660, 0.1967014, 0.9996660, -0.8029646, 0.8029646
3: -0.0085344, 0.2818646, -0.0085344, 0.2818646, -0.2903989, 0.2903989
4: -0.2053551, 0.1310268, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0729769, 0.1868500, -0.0729769, 0.1868500, -0.2598268, 0.2598268
6: -0.1155213, 0.2890729, -0.1155213, 0.2890729, -0.4045942, 0.4045942
7: -0.0929617, 0.2871873, -0.0929617, 0.2871873, -0.3801490, 0.3801490
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9996660, 0.1967014, 1.0190661, -0.8223646, 0.8029646
3: -0.0085344, 0.2818646, -0.0236642, 0.2818646, -0.2903989, 0.3055288
4: -0.2053551, 0.1310268, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0729769, 0.1868500, -0.0868930, 0.1868500, -0.2598268, 0.2737429
6: -0.1155213, 0.2890729, -0.1318814, 0.2890729, -0.4045942, 0.4209543
7: -0.0929617, 0.2871873, -0.1060784, 0.2871873, -0.3801490, 0.3932657
8: -0.0372933, 0.1999389, -0.0509215, 0.1999389, -0.2372322, 0.2508604
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0190661, 0.1967014, 0.9996660, -0.8029646, 0.8223646
3: -0.0236642, 0.2818646, -0.0085344, 0.2818646, -0.3055288, 0.2903989
4: -0.2053551, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0868930, 0.1868500, -0.0729769, 0.1868500, -0.2737429, 0.2598268
6: -0.1318814, 0.2890729, -0.1155213, 0.2890729, -0.4209543, 0.4045942
7: -0.1060784, 0.2871873, -0.0929617, 0.2871873, -0.3932657, 0.3801490
8: -0.0509215, 0.1999389, -0.0372933, 0.1999389, -0.2508604, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 1.0190661, 0.1967014, 1.0190661, -0.8223646, 0.8223646
3: -0.0236642, 0.2818646, -0.0236642, 0.2818646, -0.3055288, 0.3055288
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0868930, 0.1868500, -0.0868930, 0.1868500, -0.2737429, 0.2737429
6: -0.1318814, 0.2890729, -0.1318814, 0.2890729, -0.4209543, 0.4209543
7: -0.1060784, 0.2871873, -0.1060784, 0.2871873, -0.3932657, 0.3932657
8: -0.0509215, 0.1999389, -0.0509215, 0.1999389, -0.2508604, 0.2508604
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.57 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8679512, 0.1967014, 0.9758343, -0.7791328, 0.6712499
3: 0.0765631, 0.2818646, 0.0101185, 0.2818646, -0.2053016, 0.2717461
4: -0.2053552, 0.1310269, -0.2053552, 0.1310268, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0197655, 0.2890729, -0.0954351, 0.2890729, -0.3088384, 0.3845080
7: -0.0379298, 0.2871872, -0.0767908, 0.2871872, -0.3251170, 0.3639780
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9788760, 0.1967014, 0.9996660, -0.8029646, 0.7821746
3: 0.0077373, 0.2818646, -0.0085344, 0.2818646, -0.2741273, 0.2903989
4: -0.2053551, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0729769, 0.1868500, -0.2505312, 0.2598268
6: -0.0979986, 0.2890729, -0.1155213, 0.2890729, -0.3870715, 0.4045942
7: -0.0788552, 0.2871872, -0.0929617, 0.2871873, -0.3660425, 0.3801489
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8679512, 0.1967014, 0.9952821, -0.7985806, 0.6712499
3: 0.0765631, 0.2818646, -0.0051061, 0.2818646, -0.2053015, 0.2869707
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0698237, 0.1868500, -0.2505312, 0.2566736
6: -0.0197655, 0.2890729, -0.1118260, 0.2890729, -0.3088384, 0.4008989
7: -0.0379298, 0.2871872, -0.0899896, 0.2871872, -0.3251170, 0.3771768
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9788760, 0.1967014, 1.0190661, -0.8223646, 0.7821746
3: 0.0077373, 0.2818646, -0.0236642, 0.2818646, -0.2741273, 0.3055288
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0868930, 0.1868500, -0.2505312, 0.2737429
6: -0.0979986, 0.2890729, -0.1318814, 0.2890729, -0.3870715, 0.4209543
7: -0.0788552, 0.2871872, -0.1060784, 0.2871873, -0.3660425, 0.3932657
8: -0.0372933, 0.1999389, -0.0509215, 0.1999389, -0.2372323, 0.2508604
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444547
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8846240, 0.1967014, 0.9758343, -0.7791328, 0.6879225
3: 0.0676446, 0.2818646, 0.0101185, 0.2818646, -0.2142200, 0.2717461
4: -0.2053551, 0.1310269, -0.2053552, 0.1310268, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0301840, 0.2890729, -0.0954351, 0.2890729, -0.3192569, 0.3845080
7: -0.0379298, 0.2871872, -0.0767908, 0.2871872, -0.3251170, 0.3639781
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9982831, 0.1967014, 0.9996660, -0.8029646, 0.8015814
3: -0.0074553, 0.2818646, -0.0085344, 0.2818646, -0.2893198, 0.2903989
4: -0.2053552, 0.1310269, -0.2053551, 0.1310268, -0.3363820, 0.3363820
5: -0.0719844, 0.1868500, -0.0729769, 0.1868500, -0.2588344, 0.2598268
6: -0.1143551, 0.2890729, -0.1155213, 0.2890729, -0.4034280, 0.4045942
7: -0.0920262, 0.2871872, -0.0929617, 0.2871873, -0.3792135, 0.3801489
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372323, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8846240, 0.1967014, 0.9952821, -0.7985806, 0.6879226
3: 0.0676446, 0.2818646, -0.0051061, 0.2818646, -0.2142200, 0.2869707
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0698237, 0.1868500, -0.2505312, 0.2566736
6: -0.0301840, 0.2890729, -0.1118260, 0.2890729, -0.3192570, 0.4008988
7: -0.0379298, 0.2871872, -0.0899896, 0.2871872, -0.3251170, 0.3771769
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9982831, 0.1967014, 1.0190661, -0.8223646, 0.8015814
3: -0.0074553, 0.2818646, -0.0236642, 0.2818646, -0.2893198, 0.3055288
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0719844, 0.1868500, -0.0868930, 0.1868500, -0.2588344, 0.2737429
6: -0.1143551, 0.2890729, -0.1318814, 0.2890729, -0.4034280, 0.4209543
7: -0.0920262, 0.2871872, -0.1060784, 0.2871873, -0.3792135, 0.3932657
8: -0.0372933, 0.1999390, -0.0509215, 0.1999389, -0.2372322, 0.2508605
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.89 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279924, 0.2164623, -0.3444547, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8679512, 0.1967014, 0.8679512, -0.6712498, 0.6712498
3: 0.0765631, 0.2818646, 0.0765631, 0.2818646, -0.2053016, 0.2053016
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868499, -0.2505312, 0.2505312
6: -0.0197655, 0.2890729, -0.0197655, 0.2890729, -0.3088383, 0.3088383
7: -0.0379298, 0.2871872, -0.0379298, 0.2871872, -0.3251170, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8679512, 0.1967014, 0.9788760, -0.7821746, 0.6712499
3: 0.0765631, 0.2818646, 0.0077373, 0.2818646, -0.2053015, 0.2741273
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505311
6: -0.0197655, 0.2890729, -0.0979986, 0.2890729, -0.3088383, 0.3870715
7: -0.0379298, 0.2871872, -0.0788552, 0.2871872, -0.3251171, 0.3660424
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279924, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9788760, 0.1967014, 0.8679512, -0.6712499, 0.7821746
3: 0.0077373, 0.2818646, 0.0765631, 0.2818646, -0.2741273, 0.2053015
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505311, 0.2505312
6: -0.0979986, 0.2890729, -0.0197655, 0.2890729, -0.3870715, 0.3088383
7: -0.0788552, 0.2871872, -0.0379298, 0.2871872, -0.3660424, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9788760, 0.1967014, 0.9788760, -0.7821746, 0.7821745
3: 0.0077373, 0.2818646, 0.0077373, 0.2818646, -0.2741273, 0.2741273
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0979986, 0.2890729, -0.0979986, 0.2890729, -0.3870715, 0.3870715
7: -0.0788552, 0.2871872, -0.0788552, 0.2871872, -0.3660425, 0.3660425
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8679512, 0.1967014, 0.8846240, -0.6879225, 0.6712498
3: 0.0765631, 0.2818646, 0.0676446, 0.2818646, -0.2053016, 0.2142200
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0197655, 0.2890729, -0.0301840, 0.2890729, -0.3088383, 0.3192569
7: -0.0379298, 0.2871872, -0.0379298, 0.2871872, -0.3251171, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.19 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8679512, 0.1967014, 0.9982831, -0.8015814, 0.6712498
3: 0.0765631, 0.2818646, -0.0074553, 0.2818646, -0.2053015, 0.2893199
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363821, 0.3363820
5: -0.0636812, 0.1868499, -0.0719844, 0.1868500, -0.2505312, 0.2588344
6: -0.0197655, 0.2890729, -0.1143551, 0.2890729, -0.3088384, 0.4034280
7: -0.0379298, 0.2871872, -0.0920262, 0.2871872, -0.3251170, 0.3792135
8: -0.0372933, 0.1999389, -0.0372933, 0.1999390, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9788760, 0.1967014, 0.8846240, -0.6879226, 0.7821745
3: 0.0077373, 0.2818646, 0.0676446, 0.2818646, -0.2741273, 0.2142200
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0979986, 0.2890729, -0.0301840, 0.2890729, -0.3870715, 0.3192569
7: -0.0788552, 0.2871872, -0.0379298, 0.2871872, -0.3660425, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.43 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9788760, 0.1967014, 0.9982831, -0.8015814, 0.7821746
3: 0.0077373, 0.2818646, -0.0074553, 0.2818646, -0.2741273, 0.2893199
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0719844, 0.1868500, -0.2505312, 0.2588344
6: -0.0979986, 0.2890729, -0.1143551, 0.2890729, -0.3870715, 0.4034279
7: -0.0788552, 0.2871872, -0.0920262, 0.2871872, -0.3660424, 0.3792135
8: -0.0372933, 0.1999389, -0.0372933, 0.1999390, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8846240, 0.1967014, 0.8679512, -0.6712498, 0.6879225
3: 0.0676446, 0.2818646, 0.0765631, 0.2818646, -0.2142200, 0.2053016
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505312, 0.2505312
6: -0.0301840, 0.2890729, -0.0197655, 0.2890729, -0.3192568, 0.3088383
7: -0.0379298, 0.2871872, -0.0379298, 0.2871872, -0.3251170, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8846240, 0.1967014, 0.9788760, -0.7821745, 0.6879226
3: 0.0676446, 0.2818646, 0.0077373, 0.2818646, -0.2142200, 0.2741273
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0301840, 0.2890729, -0.0979986, 0.2890729, -0.3192569, 0.3870715
7: -0.0379298, 0.2871872, -0.0788552, 0.2871872, -0.3251171, 0.3660425
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9982831, 0.1967014, 0.8679512, -0.6712498, 0.8015814
3: -0.0074553, 0.2818646, 0.0765631, 0.2818646, -0.2893199, 0.2053015
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363821
5: -0.0719844, 0.1868500, -0.0636812, 0.1868499, -0.2588344, 0.2505312
6: -0.1143551, 0.2890729, -0.0197655, 0.2890729, -0.4034280, 0.3088384
7: -0.0920262, 0.2871872, -0.0379298, 0.2871872, -0.3792135, 0.3251170
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.29 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9982831, 0.1967014, 0.9788760, -0.7821746, 0.8015813
3: -0.0074553, 0.2818646, 0.0077373, 0.2818646, -0.2893199, 0.2741273
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0719844, 0.1868500, -0.0636812, 0.1868500, -0.2588344, 0.2505312
6: -0.1143551, 0.2890729, -0.0979986, 0.2890729, -0.4034279, 0.3870715
7: -0.0920262, 0.2871872, -0.0788552, 0.2871872, -0.3792135, 0.3660424
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444547
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8846240, 0.1967014, 0.8846240, -0.6879225, 0.6879225
3: 0.0676446, 0.2818646, 0.0676446, 0.2818646, -0.2142200, 0.2142200
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0301840, 0.2890729, -0.0301840, 0.2890729, -0.3192568, 0.3192569
7: -0.0379298, 0.2871872, -0.0379298, 0.2871872, -0.3251171, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8846240, 0.1967014, 0.9982831, -0.8015814, 0.6879226
3: 0.0676446, 0.2818646, -0.0074553, 0.2818646, -0.2142200, 0.2893199
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0719844, 0.1868500, -0.2505312, 0.2588344
6: -0.0301840, 0.2890729, -0.1143551, 0.2890729, -0.3192569, 0.4034279
7: -0.0379298, 0.2871872, -0.0920262, 0.2871872, -0.3251170, 0.3792135
8: -0.0372933, 0.1999389, -0.0372933, 0.1999390, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.12 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9982831, 0.1967014, 0.8846240, -0.6879226, 0.8015814
3: -0.0074553, 0.2818646, 0.0676446, 0.2818646, -0.2893199, 0.2142200
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0719844, 0.1868500, -0.0636812, 0.1868500, -0.2588344, 0.2505312
6: -0.1143551, 0.2890729, -0.0301840, 0.2890729, -0.4034279, 0.3192569
7: -0.0920262, 0.2871872, -0.0379298, 0.2871872, -0.3792135, 0.3251170
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9982831, 0.1967014, 0.9982831, -0.8015814, 0.8015814
3: -0.0074553, 0.2818646, -0.0074553, 0.2818646, -0.2893199, 0.2893199
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363821, 0.3363821
5: -0.0719844, 0.1868500, -0.0719844, 0.1868500, -0.2588344, 0.2588344
6: -0.1143551, 0.2890729, -0.1143551, 0.2890729, -0.4034280, 0.4034280
7: -0.0920262, 0.2871872, -0.0920262, 0.2871872, -0.3792135, 0.3792135
8: -0.0372933, 0.1999390, -0.0372933, 0.1999390, -0.2372323, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.54 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8502830, 0.1967014, 0.8679512, -0.6712498, 0.6535813
3: 0.0860144, 0.2818646, 0.0765631, 0.2818646, -0.1958502, 0.2053016
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505312, 0.2505312
6: -0.0094924, 0.2890729, -0.0197655, 0.2890729, -0.2985653, 0.3088384
7: -0.0379298, 0.2871873, -0.0379298, 0.2871872, -0.3251170, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8167245, 0.1967014, 0.8586656, -0.6619640, 0.6200230
3: 0.1011953, 0.2818646, 0.0815303, 0.2818646, -0.1806694, 0.2003343
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0098373, 0.2890729, -0.0143210, 0.2890729, -0.2792355, 0.3033939
7: -0.0379298, 0.2871873, -0.0379298, 0.2871873, -0.3251171, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8502830, 0.1967014, 0.9788760, -0.7821746, 0.6535814
3: 0.0860144, 0.2818646, 0.0077373, 0.2818646, -0.1958501, 0.2741273
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0094924, 0.2890729, -0.0979986, 0.2890729, -0.2985653, 0.3870715
7: -0.0379298, 0.2871873, -0.0788552, 0.2871872, -0.3251170, 0.3660425
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 0.8167245, 0.1967014, 0.9691353, -0.7724338, 0.6200230
3: 0.1011953, 0.2818646, 0.0153626, 0.2818646, -0.1806694, 0.2665021
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0098373, 0.2890729, -0.0897893, 0.2890729, -0.2792356, 0.3788621
7: -0.0379298, 0.2871873, -0.0722446, 0.2871872, -0.3251170, 0.3594319
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992408

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 0.9587211, 0.1967014, 0.8679512, -0.6712499, 0.7620193
3: 0.0234510, 0.2818647, 0.0765631, 0.2818646, -0.2584136, 0.2053016
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505312, 0.2505312
6: -0.0810231, 0.2890729, -0.0197655, 0.2890729, -0.3700960, 0.3088384
7: -0.0651449, 0.2871873, -0.0379298, 0.2871872, -0.3523321, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9272496, 0.1967014, 0.8586656, -0.6619640, 0.7305481
3: 0.0425619, 0.2818646, 0.0815303, 0.2818646, -0.2393027, 0.2003343
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0580613, 0.2890729, -0.0143210, 0.2890729, -0.3471342, 0.3033939
7: -0.0469453, 0.2871873, -0.0379298, 0.2871873, -0.3341326, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 0.9587211, 0.1967014, 0.9788760, -0.7821746, 0.7620193
3: 0.0234510, 0.2818647, 0.0077373, 0.2818646, -0.2584136, 0.2741274
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0810231, 0.2890729, -0.0979986, 0.2890729, -0.3700960, 0.3870715
7: -0.0651449, 0.2871873, -0.0788552, 0.2871872, -0.3523322, 0.3660425
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9272496, 0.1967014, 0.9691353, -0.7724338, 0.7305481
3: 0.0425619, 0.2818646, 0.0153626, 0.2818646, -0.2393027, 0.2665021
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505311
6: -0.0580613, 0.2890729, -0.0897893, 0.2890729, -0.3471342, 0.3788622
7: -0.0469453, 0.2871873, -0.0722446, 0.2871872, -0.3341326, 0.3594319
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8502830, 0.1967014, 0.8846240, -0.6879225, 0.6535813
3: 0.0860144, 0.2818646, 0.0676446, 0.2818646, -0.1958502, 0.2142200
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0094924, 0.2890729, -0.0301840, 0.2890729, -0.2985653, 0.3192569
7: -0.0379298, 0.2871873, -0.0379298, 0.2871872, -0.3251170, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8167245, 0.1967014, 0.8758416, -0.6791402, 0.6200231
3: 0.1011953, 0.2818646, 0.0723423, 0.2818646, -0.1806694, 0.2095223
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0098373, 0.2890729, -0.0246027, 0.2890729, -0.2792355, 0.3136756
7: -0.0379298, 0.2871873, -0.0379298, 0.2871872, -0.3251170, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.14 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8502830, 0.1967014, 0.9982831, -0.8015814, 0.6535814
3: 0.0860144, 0.2818646, -0.0074553, 0.2818646, -0.1958501, 0.2893199
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0719844, 0.1868500, -0.2505312, 0.2588344
6: -0.0094924, 0.2890729, -0.1143551, 0.2890729, -0.2985653, 0.4034280
7: -0.0379298, 0.2871873, -0.0920262, 0.2871872, -0.3251170, 0.3792135
8: -0.0372933, 0.1999389, -0.0372933, 0.1999390, -0.2372323, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8167245, 0.1967014, 0.9887264, -0.7920250, 0.6200230
3: 0.1011953, 0.2818646, 0.0000259, 0.2818646, -0.1806694, 0.2818387
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0651034, 0.1868499, -0.2505311, 0.2519534
6: 0.0098373, 0.2890729, -0.1063009, 0.2890729, -0.2792356, 0.3953737
7: -0.0379298, 0.2871873, -0.0855405, 0.2871873, -0.3251171, 0.3727278
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 0.9587211, 0.1967014, 0.8846240, -0.6879225, 0.7620193
3: 0.0234510, 0.2818647, 0.0676446, 0.2818646, -0.2584136, 0.2142201
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0810231, 0.2890729, -0.0301840, 0.2890729, -0.3700960, 0.3192570
7: -0.0651449, 0.2871873, -0.0379298, 0.2871872, -0.3523322, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9272496, 0.1967014, 0.8758416, -0.6791401, 0.7305481
3: 0.0425619, 0.2818646, 0.0723423, 0.2818646, -0.2393028, 0.2095223
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505311
6: -0.0580613, 0.2890729, -0.0246027, 0.2890729, -0.3471342, 0.3136756
7: -0.0469453, 0.2871873, -0.0379298, 0.2871872, -0.3341326, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 0.9587211, 0.1967014, 0.9982831, -0.8015814, 0.7620193
3: 0.0234510, 0.2818647, -0.0074553, 0.2818646, -0.2584136, 0.2893200
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0719844, 0.1868500, -0.2505312, 0.2588344
6: -0.0810231, 0.2890729, -0.1143551, 0.2890729, -0.3700960, 0.4034280
7: -0.0651449, 0.2871873, -0.0920262, 0.2871872, -0.3523321, 0.3792135
8: -0.0372933, 0.1999389, -0.0372933, 0.1999390, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9272496, 0.1967014, 0.9887264, -0.7920250, 0.7305481
3: 0.0425619, 0.2818646, 0.0000259, 0.2818646, -0.2393027, 0.2818387
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0651034, 0.1868499, -0.2505311, 0.2519534
6: -0.0580613, 0.2890729, -0.1063009, 0.2890729, -0.3471342, 0.3953738
7: -0.0469453, 0.2871873, -0.0855405, 0.2871873, -0.3341326, 0.3727278
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8663821, 0.1967014, 0.8679512, -0.6712499, 0.6696807
3: 0.0774025, 0.2818646, 0.0765631, 0.2818646, -0.2044621, 0.2053016
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505312, 0.2505312
6: -0.0188035, 0.2890729, -0.0197655, 0.2890729, -0.3078764, 0.3088383
7: -0.0379298, 0.2871873, -0.0379298, 0.2871872, -0.3251170, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8335435, 0.1967014, 0.8586656, -0.6619641, 0.6368421
3: 0.0939234, 0.2818646, 0.0815303, 0.2818646, -0.1879413, 0.2003344
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0001494, 0.2890729, -0.0143210, 0.2890729, -0.2889235, 0.3033939
7: -0.0379298, 0.2871872, -0.0379298, 0.2871873, -0.3251171, 0.3251171
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 0.8663821, 0.1967014, 0.9788760, -0.7821746, 0.6696807
3: 0.0774025, 0.2818646, 0.0077373, 0.2818646, -0.2044621, 0.2741273
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0188035, 0.2890729, -0.0979986, 0.2890729, -0.3078763, 0.3870715
7: -0.0379298, 0.2871873, -0.0788552, 0.2871872, -0.3251171, 0.3660425
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 0.8335435, 0.1967014, 0.9691353, -0.7724338, 0.6368421
3: 0.0939234, 0.2818646, 0.0153626, 0.2818646, -0.1879413, 0.2665021
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0001494, 0.2890729, -0.0897893, 0.2890729, -0.2889235, 0.3788622
7: -0.0379298, 0.2871872, -0.0722446, 0.2871872, -0.3251170, 0.3594318
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9779652, 0.1967014, 0.8679512, -0.6712498, 0.7812636
3: 0.0084504, 0.2818646, 0.0765631, 0.2818646, -0.2734142, 0.2053016
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505312, 0.2505312
6: -0.0972310, 0.2890729, -0.0197655, 0.2890729, -0.3863039, 0.3088384
7: -0.0782370, 0.2871872, -0.0379298, 0.2871872, -0.3654242, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.22 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9449915, 0.1967014, 0.8586656, -0.6619641, 0.7482896
3: 0.0318805, 0.2818646, 0.0815303, 0.2818646, -0.2499841, 0.2003343
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505311, 0.2505312
6: -0.0704947, 0.2890729, -0.0143210, 0.2890729, -0.3595677, 0.3033939
7: -0.0563864, 0.2871872, -0.0379298, 0.2871873, -0.3435736, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9779652, 0.1967014, 0.9788760, -0.7821746, 0.7812636
3: 0.0084504, 0.2818646, 0.0077373, 0.2818646, -0.2734142, 0.2741273
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0972310, 0.2890729, -0.0979986, 0.2890729, -0.3863039, 0.3870715
7: -0.0782370, 0.2871872, -0.0788552, 0.2871872, -0.3654242, 0.3660425
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 0.9449915, 0.1967014, 0.9691353, -0.7724339, 0.7482896
3: 0.0318805, 0.2818646, 0.0153626, 0.2818646, -0.2499841, 0.2665020
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505311
6: -0.0704947, 0.2890729, -0.0897893, 0.2890729, -0.3595676, 0.3788622
7: -0.0563864, 0.2871872, -0.0722446, 0.2871872, -0.3435736, 0.3594318
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8663821, 0.1967014, 0.8846240, -0.6879225, 0.6696807
3: 0.0774025, 0.2818646, 0.0676446, 0.2818646, -0.2044621, 0.2142200
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0188035, 0.2890729, -0.0301840, 0.2890729, -0.3078763, 0.3192569
7: -0.0379298, 0.2871873, -0.0379298, 0.2871872, -0.3251171, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8335435, 0.1967014, 0.8758416, -0.6791402, 0.6368421
3: 0.0939234, 0.2818646, 0.0723423, 0.2818646, -0.1879413, 0.2095223
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0001494, 0.2890729, -0.0246027, 0.2890729, -0.2889234, 0.3136756
7: -0.0379298, 0.2871872, -0.0379298, 0.2871872, -0.3251170, 0.3251171
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 0.8663821, 0.1967014, 0.9982831, -0.8015814, 0.6696807
3: 0.0774025, 0.2818646, -0.0074553, 0.2818646, -0.2044621, 0.2893199
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0719844, 0.1868500, -0.2505312, 0.2588344
6: -0.0188035, 0.2890729, -0.1143551, 0.2890729, -0.3078764, 0.4034280
7: -0.0379298, 0.2871873, -0.0920262, 0.2871872, -0.3251170, 0.3792135
8: -0.0372933, 0.1999389, -0.0372933, 0.1999390, -0.2372323, 0.2372323
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8335435, 0.1967014, 0.9887264, -0.7920250, 0.6368421
3: 0.0939234, 0.2818646, 0.0000259, 0.2818646, -0.1879413, 0.2818387
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0651034, 0.1868499, -0.2505312, 0.2519534
6: 0.0001494, 0.2890729, -0.1063009, 0.2890729, -0.2889235, 0.3953738
7: -0.0379298, 0.2871872, -0.0855405, 0.2871873, -0.3251171, 0.3727278
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9779652, 0.1967014, 0.8846240, -0.6879226, 0.7812635
3: 0.0084504, 0.2818646, 0.0676446, 0.2818646, -0.2734142, 0.2142200
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0972310, 0.2890729, -0.0301840, 0.2890729, -0.3863039, 0.3192570
7: -0.0782370, 0.2871872, -0.0379298, 0.2871872, -0.3654242, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9449915, 0.1967014, 0.8758416, -0.6791402, 0.7482896
3: 0.0318805, 0.2818646, 0.0723423, 0.2818646, -0.2499841, 0.2095222
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505311
6: -0.0704947, 0.2890729, -0.0246027, 0.2890729, -0.3595676, 0.3136756
7: -0.0563864, 0.2871872, -0.0379298, 0.2871872, -0.3435736, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9779652, 0.1967014, 0.9982831, -0.8015814, 0.7812636
3: 0.0084504, 0.2818646, -0.0074553, 0.2818646, -0.2734142, 0.2893199
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0719844, 0.1868500, -0.2505312, 0.2588344
6: -0.0972310, 0.2890729, -0.1143551, 0.2890729, -0.3863039, 0.4034280
7: -0.0782370, 0.2871872, -0.0920262, 0.2871872, -0.3654242, 0.3792135
8: -0.0372933, 0.1999389, -0.0372933, 0.1999390, -0.2372323, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 0.9449915, 0.1967014, 0.9887264, -0.7920249, 0.7482896
3: 0.0318805, 0.2818646, 0.0000259, 0.2818646, -0.2499841, 0.2818387
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0651034, 0.1868499, -0.2505311, 0.2519534
6: -0.0704947, 0.2890729, -0.1063009, 0.2890729, -0.3595677, 0.3953738
7: -0.0563864, 0.2871872, -0.0855405, 0.2871873, -0.3435736, 0.3727278
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.57 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8502830, 0.1967014, 0.8502830, -0.6535813, 0.6535813
3: 0.0860144, 0.2818646, 0.0860144, 0.2818646, -0.1958502, 0.1958502
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0094924, 0.2890729, -0.0094924, 0.2890729, -0.2985653, 0.2985653
7: -0.0379298, 0.2871873, -0.0379298, 0.2871873, -0.3251171, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8502830, 0.1967014, 0.8167245, -0.6200230, 0.6535813
3: 0.0860144, 0.2818646, 0.1011953, 0.2818646, -0.1958502, 0.1806694
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0094924, 0.2890729, 0.0098373, 0.2890729, -0.2985653, 0.2792355
7: -0.0379298, 0.2871873, -0.0379298, 0.2871873, -0.3251171, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232192, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8167245, 0.1967014, 0.8502830, -0.6535813, 0.6200231
3: 0.1011953, 0.2818646, 0.0860144, 0.2818646, -0.1806694, 0.1958502
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0098373, 0.2890729, -0.0094924, 0.2890729, -0.2792355, 0.2985653
7: -0.0379298, 0.2871873, -0.0379298, 0.2871873, -0.3251171, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 5.15 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8167245, 0.1967014, 0.8167245, -0.6200230, 0.6200230
3: 0.1011953, 0.2818646, 0.1011953, 0.2818646, -0.1806694, 0.1806694
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0098373, 0.2890729, 0.0098373, 0.2890729, -0.2792355, 0.2792355
7: -0.0379298, 0.2871873, -0.0379298, 0.2871873, -0.3251171, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232192, 0.1760216, -0.1232192, 0.1760216, -0.2992408, 0.2992408

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8502830, 0.1967013, 0.9587211, -0.7620193, 0.6535813
3: 0.0860144, 0.2818646, 0.0234510, 0.2818647, -0.1958502, 0.2584136
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0094924, 0.2890729, -0.0810231, 0.2890729, -0.2985653, 0.3700960
7: -0.0379298, 0.2871873, -0.0651449, 0.2871873, -0.3251171, 0.3523322
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2008549, upper bound: 0.2009856
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1996619, upper bound: 0.2009856
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8502830, 0.1967014, 0.9272496, -0.7305481, 0.6535813
3: 0.0860144, 0.2818646, 0.0425619, 0.2818646, -0.1958502, 0.2393027
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505312, 0.2505312
6: -0.0094924, 0.2890729, -0.0580613, 0.2890729, -0.2985653, 0.3471342
7: -0.0379298, 0.2871873, -0.0469453, 0.2871873, -0.3251171, 0.3341326
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2008549, upper bound: 0.2009856
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1996619, upper bound: 0.2009856
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8167245, 0.1967013, 0.9587211, -0.7620192, 0.6200230
3: 0.1011953, 0.2818646, 0.0234510, 0.2818647, -0.1806694, 0.2584136
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0098373, 0.2890729, -0.0810231, 0.2890729, -0.2792356, 0.3700960
7: -0.0379298, 0.2871873, -0.0651449, 0.2871873, -0.3251171, 0.3523322
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8167245, 0.1967014, 0.9272496, -0.7305480, 0.6200229
3: 0.1011953, 0.2818646, 0.0425619, 0.2818646, -0.1806694, 0.2393027
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505311, 0.2505312
6: 0.0098373, 0.2890729, -0.0580613, 0.2890729, -0.2792356, 0.3471342
7: -0.0379298, 0.2871873, -0.0469453, 0.2871873, -0.3251171, 0.3341326
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 0.9587211, 0.1967014, 0.8502830, -0.6535813, 0.7620193
3: 0.0234510, 0.2818647, 0.0860144, 0.2818646, -0.2584136, 0.1958502
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0810231, 0.2890729, -0.0094924, 0.2890729, -0.3700960, 0.2985653
7: -0.0651449, 0.2871873, -0.0379298, 0.2871873, -0.3523322, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 0.9587211, 0.1967014, 0.8167245, -0.6200230, 0.7620193
3: 0.0234510, 0.2818647, 0.1011953, 0.2818646, -0.2584136, 0.1806694
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0810231, 0.2890729, 0.0098373, 0.2890729, -0.3700960, 0.2792356
7: -0.0651449, 0.2871873, -0.0379298, 0.2871873, -0.3523322, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232192, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9272496, 0.1967014, 0.8502830, -0.6535813, 0.7305481
3: 0.0425619, 0.2818646, 0.0860144, 0.2818646, -0.2393027, 0.1958502
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0580613, 0.2890729, -0.0094924, 0.2890729, -0.3471342, 0.2985653
7: -0.0469453, 0.2871873, -0.0379298, 0.2871873, -0.3341326, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9272496, 0.1967014, 0.8167245, -0.6200229, 0.7305481
3: 0.0425619, 0.2818646, 0.1011953, 0.2818646, -0.2393027, 0.1806694
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505311
6: -0.0580613, 0.2890729, 0.0098373, 0.2890729, -0.3471342, 0.2792356
7: -0.0469453, 0.2871873, -0.0379298, 0.2871873, -0.3341326, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232192, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 0.9587211, 0.1967013, 0.9587211, -0.7620192, 0.7620193
3: 0.0234510, 0.2818647, 0.0234510, 0.2818647, -0.2584136, 0.2584136
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0810231, 0.2890729, -0.0810231, 0.2890729, -0.3700960, 0.3700960
7: -0.0651449, 0.2871873, -0.0651449, 0.2871873, -0.3523322, 0.3523322
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 0.9587211, 0.1967014, 0.9272496, -0.7305481, 0.7620192
3: 0.0234510, 0.2818647, 0.0425619, 0.2818646, -0.2584136, 0.2393027
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505312, 0.2505312
6: -0.0810231, 0.2890729, -0.0580613, 0.2890729, -0.3700960, 0.3471342
7: -0.0651449, 0.2871873, -0.0469453, 0.2871873, -0.3523322, 0.3341326
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.45 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9272496, 0.1967013, 0.9587211, -0.7620192, 0.7305481
3: 0.0425619, 0.2818646, 0.0234510, 0.2818647, -0.2393028, 0.2584136
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0580613, 0.2890729, -0.0810231, 0.2890729, -0.3471342, 0.3700960
7: -0.0469453, 0.2871873, -0.0651449, 0.2871873, -0.3341326, 0.3523322
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9272496, 0.1967014, 0.9272496, -0.7305479, 0.7305480
3: 0.0425619, 0.2818646, 0.0425619, 0.2818646, -0.2393027, 0.2393027
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868499, -0.2505311, 0.2505311
6: -0.0580613, 0.2890729, -0.0580613, 0.2890729, -0.3471342, 0.3471342
7: -0.0469453, 0.2871873, -0.0469453, 0.2871873, -0.3341326, 0.3341326
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992408

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8502830, 0.1967014, 0.8663821, -0.6696807, 0.6535813
3: 0.0860144, 0.2818646, 0.0774025, 0.2818646, -0.1958502, 0.2044621
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0094924, 0.2890729, -0.0188035, 0.2890729, -0.2985653, 0.3078764
7: -0.0379298, 0.2871873, -0.0379298, 0.2871873, -0.3251171, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232192, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8502830, 0.1967014, 0.8335435, -0.6368421, 0.6535813
3: 0.0860144, 0.2818646, 0.0939234, 0.2818646, -0.1958502, 0.1879413
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0094924, 0.2890729, 0.0001494, 0.2890729, -0.2985653, 0.2889235
7: -0.0379298, 0.2871873, -0.0379298, 0.2871872, -0.3251170, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999390, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8167245, 0.1967014, 0.8663821, -0.6696807, 0.6200231
3: 0.1011953, 0.2818646, 0.0774025, 0.2818646, -0.1806694, 0.2044621
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0098373, 0.2890729, -0.0188035, 0.2890729, -0.2792355, 0.3078763
7: -0.0379298, 0.2871873, -0.0379298, 0.2871873, -0.3251171, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232192, 0.1760216, -0.1232192, 0.1760216, -0.2992408, 0.2992408

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8167245, 0.1967014, 0.8335435, -0.6368421, 0.6200230
3: 0.1011953, 0.2818646, 0.0939234, 0.2818646, -0.1806694, 0.1879413
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0098373, 0.2890729, 0.0001494, 0.2890729, -0.2792356, 0.2889234
7: -0.0379298, 0.2871873, -0.0379298, 0.2871872, -0.3251170, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999390, -0.2372323, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992408

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8502830, 0.1967014, 0.9779652, -0.7812635, 0.6535814
3: 0.0860144, 0.2818646, 0.0084504, 0.2818646, -0.1958502, 0.2734142
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0094924, 0.2890729, -0.0972310, 0.2890729, -0.2985653, 0.3863040
7: -0.0379298, 0.2871873, -0.0782370, 0.2871872, -0.3251170, 0.3654243
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004833, upper bound: 0.2009856
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8502830, 0.1967014, 0.9449915, -0.7482896, 0.6535811
3: 0.0860144, 0.2818646, 0.0318805, 0.2818646, -0.1958501, 0.2499841
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505312, 0.2505311
6: -0.0094924, 0.2890729, -0.0704947, 0.2890729, -0.2985653, 0.3595677
7: -0.0379298, 0.2871873, -0.0563864, 0.2871872, -0.3251170, 0.3435736
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2004833, upper bound: 0.2009856
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8167245, 0.1967014, 0.9779652, -0.7812635, 0.6200230
3: 0.1011953, 0.2818646, 0.0084504, 0.2818646, -0.1806694, 0.2734142
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0098373, 0.2890729, -0.0972310, 0.2890729, -0.2792356, 0.3863039
7: -0.0379298, 0.2871873, -0.0782370, 0.2871872, -0.3251170, 0.3654243
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8167245, 0.1967014, 0.9449915, -0.7482896, 0.6200229
3: 0.1011953, 0.2818646, 0.0318805, 0.2818646, -0.1806693, 0.2499841
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505311, 0.2505311
6: 0.0098373, 0.2890729, -0.0704947, 0.2890729, -0.2792356, 0.3595676
7: -0.0379298, 0.2871873, -0.0563864, 0.2871872, -0.3251170, 0.3435736
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185008
2: 0.1967013, 0.9587211, 0.1967014, 0.8663821, -0.6696806, 0.7620193
3: 0.0234510, 0.2818647, 0.0774025, 0.2818646, -0.2584137, 0.2044622
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0810231, 0.2890729, -0.0188035, 0.2890729, -0.3700960, 0.3078764
7: -0.0651449, 0.2871873, -0.0379298, 0.2871873, -0.3523322, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232192, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 0.9587211, 0.1967014, 0.8335435, -0.6368421, 0.7620193
3: 0.0234510, 0.2818647, 0.0939234, 0.2818646, -0.2584137, 0.1879413
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0810231, 0.2890729, 0.0001494, 0.2890729, -0.3700960, 0.2889235
7: -0.0651449, 0.2871873, -0.0379298, 0.2871872, -0.3523322, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999390, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185008
2: 0.1967014, 0.9272496, 0.1967014, 0.8663821, -0.6696806, 0.7305481
3: 0.0425619, 0.2818646, 0.0774025, 0.2818646, -0.2393028, 0.2044621
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0580613, 0.2890729, -0.0188035, 0.2890729, -0.3471342, 0.3078764
7: -0.0469453, 0.2871873, -0.0379298, 0.2871873, -0.3341326, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232192, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9272496, 0.1967014, 0.8335435, -0.6368420, 0.7305481
3: 0.0425619, 0.2818646, 0.0939234, 0.2818646, -0.2393028, 0.1879413
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0580613, 0.2890729, 0.0001494, 0.2890729, -0.3471342, 0.2889235
7: -0.0469453, 0.2871873, -0.0379298, 0.2871872, -0.3341326, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999390, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967013, 0.9587211, 0.1967014, 0.9779652, -0.7812636, 0.7620193
3: 0.0234510, 0.2818647, 0.0084504, 0.2818646, -0.2584136, 0.2734143
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0810231, 0.2890729, -0.0972310, 0.2890729, -0.3700960, 0.3863039
7: -0.0651449, 0.2871873, -0.0782370, 0.2871872, -0.3523322, 0.3654243
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.24 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185008
2: 0.1967013, 0.9587211, 0.1967014, 0.9449915, -0.7482896, 0.7620193
3: 0.0234510, 0.2818647, 0.0318805, 0.2818646, -0.2584136, 0.2499841
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505312, 0.2505311
6: -0.0810231, 0.2890729, -0.0704947, 0.2890729, -0.3700960, 0.3595677
7: -0.0651449, 0.2871873, -0.0563864, 0.2871872, -0.3523321, 0.3435736
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9272496, 0.1967014, 0.9779652, -0.7812636, 0.7305481
3: 0.0425619, 0.2818646, 0.0084504, 0.2818646, -0.2393027, 0.2734142
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0580613, 0.2890729, -0.0972310, 0.2890729, -0.3471342, 0.3863039
7: -0.0469453, 0.2871873, -0.0782370, 0.2871872, -0.3341326, 0.3654243
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279924, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911178, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185008
2: 0.1967014, 0.9272496, 0.1967014, 0.9449915, -0.7482895, 0.7305480
3: 0.0425619, 0.2818646, 0.0318805, 0.2818646, -0.2393027, 0.2499841
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868499, -0.2505311, 0.2505311
6: -0.0580613, 0.2890729, -0.0704947, 0.2890729, -0.3471342, 0.3595676
7: -0.0469453, 0.2871873, -0.0563864, 0.2871872, -0.3341326, 0.3435736
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8663821, 0.1967014, 0.8502830, -0.6535813, 0.6696807
3: 0.0774025, 0.2818646, 0.0860144, 0.2818646, -0.2044621, 0.1958502
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0188035, 0.2890729, -0.0094924, 0.2890729, -0.3078764, 0.2985653
7: -0.0379298, 0.2871873, -0.0379298, 0.2871873, -0.3251171, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8663821, 0.1967014, 0.8167245, -0.6200231, 0.6696807
3: 0.0774025, 0.2818646, 0.1011953, 0.2818646, -0.2044621, 0.1806694
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0188035, 0.2890729, 0.0098373, 0.2890729, -0.3078763, 0.2792355
7: -0.0379298, 0.2871873, -0.0379298, 0.2871873, -0.3251171, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232192, 0.1760216, -0.1232192, 0.1760216, -0.2992408, 0.2992408

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8335435, 0.1967014, 0.8502830, -0.6535813, 0.6368421
3: 0.0939234, 0.2818646, 0.0860144, 0.2818646, -0.1879413, 0.1958502
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0001494, 0.2890729, -0.0094924, 0.2890729, -0.2889235, 0.2985653
7: -0.0379298, 0.2871872, -0.0379298, 0.2871873, -0.3251171, 0.3251170
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8335435, 0.1967014, 0.8167245, -0.6200230, 0.6368421
3: 0.0939234, 0.2818646, 0.1011953, 0.2818646, -0.1879413, 0.1806694
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0001494, 0.2890729, 0.0098373, 0.2890729, -0.2889234, 0.2792356
7: -0.0379298, 0.2871872, -0.0379298, 0.2871873, -0.3251171, 0.3251170
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232192, 0.1760216, -0.2992408, 0.2992408

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 0.8663821, 0.1967013, 0.9587211, -0.7620193, 0.6696806
3: 0.0774025, 0.2818646, 0.0234510, 0.2818647, -0.2044622, 0.2584137
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0188035, 0.2890729, -0.0810231, 0.2890729, -0.3078764, 0.3700960
7: -0.0379298, 0.2871873, -0.0651449, 0.2871873, -0.3251171, 0.3523322
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.26 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 0.8663821, 0.1967014, 0.9272496, -0.7305481, 0.6696806
3: 0.0774025, 0.2818646, 0.0425619, 0.2818646, -0.2044621, 0.2393028
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505312, 0.2505312
6: -0.0188035, 0.2890729, -0.0580613, 0.2890729, -0.3078764, 0.3471342
7: -0.0379298, 0.2871873, -0.0469453, 0.2871873, -0.3251171, 0.3341326
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8335435, 0.1967013, 0.9587211, -0.7620192, 0.6368421
3: 0.0939234, 0.2818646, 0.0234510, 0.2818647, -0.1879413, 0.2584137
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0001494, 0.2890729, -0.0810231, 0.2890729, -0.2889235, 0.3700960
7: -0.0379298, 0.2871872, -0.0651449, 0.2871873, -0.3251171, 0.3523322
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8335435, 0.1967014, 0.9272496, -0.7305480, 0.6368420
3: 0.0939234, 0.2818646, 0.0425619, 0.2818646, -0.1879413, 0.2393028
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505312, 0.2505312
6: 0.0001494, 0.2890729, -0.0580613, 0.2890729, -0.2889235, 0.3471342
7: -0.0379298, 0.2871872, -0.0469453, 0.2871873, -0.3251171, 0.3341326
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 1.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9779652, 0.1967014, 0.8502830, -0.6535813, 0.7812636
3: 0.0084504, 0.2818646, 0.0860144, 0.2818646, -0.2734142, 0.1958502
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0972310, 0.2890729, -0.0094924, 0.2890729, -0.3863040, 0.2985653
7: -0.0782370, 0.2871872, -0.0379298, 0.2871873, -0.3654243, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.27 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9779652, 0.1967014, 0.8167245, -0.6200230, 0.7812636
3: 0.0084504, 0.2818646, 0.1011953, 0.2818646, -0.2734142, 0.1806694
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0972310, 0.2890729, 0.0098373, 0.2890729, -0.3863039, 0.2792356
7: -0.0782370, 0.2871872, -0.0379298, 0.2871873, -0.3654243, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232192, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9449915, 0.1967014, 0.8502830, -0.6535811, 0.7482896
3: 0.0318805, 0.2818646, 0.0860144, 0.2818646, -0.2499841, 0.1958501
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505311, 0.2505312
6: -0.0704947, 0.2890729, -0.0094924, 0.2890729, -0.3595677, 0.2985653
7: -0.0563864, 0.2871872, -0.0379298, 0.2871873, -0.3435736, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.35 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9449915, 0.1967014, 0.8167245, -0.6200229, 0.7482896
3: 0.0318805, 0.2818646, 0.1011953, 0.2818646, -0.2499841, 0.1806693
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505311, 0.2505311
6: -0.0704947, 0.2890729, 0.0098373, 0.2890729, -0.3595676, 0.2792356
7: -0.0563864, 0.2871872, -0.0379298, 0.2871873, -0.3435736, 0.3251170
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232192, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9779652, 0.1967013, 0.9587211, -0.7620193, 0.7812636
3: 0.0084504, 0.2818646, 0.0234510, 0.2818647, -0.2734143, 0.2584136
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0972310, 0.2890729, -0.0810231, 0.2890729, -0.3863039, 0.3700960
7: -0.0782370, 0.2871872, -0.0651449, 0.2871873, -0.3654243, 0.3523322
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.9779652, 0.1967014, 0.9272496, -0.7305481, 0.7812636
3: 0.0084504, 0.2818646, 0.0425619, 0.2818646, -0.2734142, 0.2393027
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505312, 0.2505312
6: -0.0972310, 0.2890729, -0.0580613, 0.2890729, -0.3863039, 0.3471342
7: -0.0782370, 0.2871872, -0.0469453, 0.2871873, -0.3654243, 0.3341326
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 0.9449915, 0.1967013, 0.9587211, -0.7620192, 0.7482896
3: 0.0318805, 0.2818646, 0.0234510, 0.2818647, -0.2499841, 0.2584136
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868500, -0.2505311, 0.2505312
6: -0.0704947, 0.2890729, -0.0810231, 0.2890729, -0.3595677, 0.3700960
7: -0.0563864, 0.2871872, -0.0651449, 0.2871873, -0.3435736, 0.3523321
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279924, 0.2164623, -0.3444548, 0.3444547
1: -0.0911179, 0.1273829, -0.0911178, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 0.9449915, 0.1967014, 0.9272496, -0.7305480, 0.7482895
3: 0.0318805, 0.2818646, 0.0425619, 0.2818646, -0.2499841, 0.2393027
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868499, -0.0636812, 0.1868499, -0.2505311, 0.2505311
6: -0.0704947, 0.2890729, -0.0580613, 0.2890729, -0.3595676, 0.3471342
7: -0.0563864, 0.2871872, -0.0469453, 0.2871873, -0.3435736, 0.3341326
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372322
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185008, 0.2185008
2: 0.1967014, 0.8663821, 0.1967014, 0.8663821, -0.6696807, 0.6696807
3: 0.0774025, 0.2818646, 0.0774025, 0.2818646, -0.2044621, 0.2044621
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0188035, 0.2890729, -0.0188035, 0.2890729, -0.3078764, 0.3078763
7: -0.0379298, 0.2871873, -0.0379298, 0.2871873, -0.3251171, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372323
9: -0.1232192, 0.1760216, -0.1232192, 0.1760216, -0.2992408, 0.2992408

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8663821, 0.1967014, 0.8335435, -0.6368421, 0.6696807
3: 0.0774025, 0.2818646, 0.0939234, 0.2818646, -0.2044621, 0.1879413
4: -0.2053551, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0188035, 0.2890729, 0.0001494, 0.2890729, -0.3078764, 0.2889234
7: -0.0379298, 0.2871873, -0.0379298, 0.2871872, -0.3251171, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999390, -0.2372323, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8335435, 0.1967014, 0.8663821, -0.6696807, 0.6368421
3: 0.0939234, 0.2818646, 0.0774025, 0.2818646, -0.1879413, 0.2044621
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0001494, 0.2890729, -0.0188035, 0.2890729, -0.2889234, 0.3078764
7: -0.0379298, 0.2871872, -0.0379298, 0.2871873, -0.3251171, 0.3251171
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232192, 0.1760216, -0.2992409, 0.2992408

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8335435, 0.1967014, 0.8335435, -0.6368421, 0.6368421
3: 0.0939234, 0.2818646, 0.0939234, 0.2818646, -0.1879413, 0.1879413
4: -0.2053552, 0.1310269, -0.2053552, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0001494, 0.2890729, 0.0001494, 0.2890729, -0.2889235, 0.2889235
7: -0.0379298, 0.2871872, -0.0379298, 0.2871872, -0.3251170, 0.3251170
8: -0.0372933, 0.1999390, -0.0372933, 0.1999390, -0.2372323, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185008, 0.2185007
2: 0.1967014, 0.8663821, 0.1967014, 0.9779652, -0.7812635, 0.6696807
3: 0.0774025, 0.2818646, 0.0084504, 0.2818646, -0.2044621, 0.2734142
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0188035, 0.2890729, -0.0972310, 0.2890729, -0.3078764, 0.3863040
7: -0.0379298, 0.2871873, -0.0782370, 0.2871872, -0.3251171, 0.3654243
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.20 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.17 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185008
2: 0.1967014, 0.8663821, 0.1967014, 0.9449915, -0.7482896, 0.6696805
3: 0.0774025, 0.2818646, 0.0318805, 0.2818646, -0.2044621, 0.2499841
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505312, 0.2505311
6: -0.0188035, 0.2890729, -0.0704947, 0.2890729, -0.3078764, 0.3595677
7: -0.0379298, 0.2871873, -0.0563864, 0.2871872, -0.3251170, 0.3435736
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372323, 0.2372322
9: -0.1232192, 0.1760216, -0.1232193, 0.1760216, -0.2992408, 0.2992409

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8335435, 0.1967014, 0.9779652, -0.7812635, 0.6368421
3: 0.0939234, 0.2818646, 0.0084504, 0.2818646, -0.1879413, 0.2734142
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: 0.0001494, 0.2890729, -0.0972310, 0.2890729, -0.2889235, 0.3863039
7: -0.0379298, 0.2871872, -0.0782370, 0.2871872, -0.3251170, 0.3654242
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1984992, upper bound: 0.2007772
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 1.12 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444547, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185007
2: 0.1967014, 0.8335435, 0.1967014, 0.9449915, -0.7482896, 0.6368421
3: 0.0939234, 0.2818646, 0.0318805, 0.2818646, -0.1879412, 0.2499841
4: -0.2053552, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868499, -0.2505312, 0.2505311
6: 0.0001494, 0.2890729, -0.0704947, 0.2890729, -0.2889235, 0.3595676
7: -0.0379298, 0.2871872, -0.0563864, 0.2871872, -0.3251170, 0.3435736
8: -0.0372933, 0.1999390, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232193, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1984992, upper bound: 0.2007772
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009796, upper bound: 0.2009856
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2009856, upper bound: 0.2009856
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1279925, 0.2164623, -0.1279925, 0.2164623, -0.3444548, 0.3444548
1: -0.0911179, 0.1273829, -0.0911179, 0.1273829, -0.2185007, 0.2185008
2: 0.1967014, 0.9779652, 0.1967014, 0.8663821, -0.6696807, 0.7812636
3: 0.0084504, 0.2818646, 0.0774025, 0.2818646, -0.2734142, 0.2044621
4: -0.2053551, 0.1310269, -0.2053551, 0.1310269, -0.3363820, 0.3363820
5: -0.0636812, 0.1868500, -0.0636812, 0.1868500, -0.2505312, 0.2505312
6: -0.0972310, 0.2890729, -0.0188035, 0.2890729, -0.3863040, 0.3078764
7: -0.0782370, 0.2871872, -0.0379298, 0.2871873, -0.3654243, 0.3251171
8: -0.0372933, 0.1999389, -0.0372933, 0.1999389, -0.2372322, 0.2372323
9: -0.1232193, 0.1760216, -0.1232192, 0.1760216, -0.2992409, 0.2992409

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.12 + 596.37 = 600.49 seconds
