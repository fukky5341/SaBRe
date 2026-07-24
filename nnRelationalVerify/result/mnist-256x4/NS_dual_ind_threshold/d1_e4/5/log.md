## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00262656


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0063999, 0.0063999)
1: (-0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015947, 0.0015947)
2: (0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0084510, 0.0084510)
3: (-0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0038465, 0.0038465)
4: (0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0016357, 0.0016357)
5: (0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0106292, 0.0106292)
6: (-0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026978, 0.0026978)
7: (-0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0069800, 0.0069800)
8: (-0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0036707, 0.0036707)
9: (-0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0042564, 0.0042564)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.81 + 2.28 = 4.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0032832, upper bound: 0.0032832

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0031766, upper bound: 0.0030743
time: 1.33 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0031766, upper bound: 0.0031766
time: 1.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.82 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.82
Output dim: 0, lower bound: -0.0031766, upper bound: 0.0030743
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.82
Output dim: 0, lower bound: -0.0031766, upper bound: 0.0031766

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.9890559, 0.9962307, 0.9889945, 0.9964978, -0.0060027, 0.0057921
1: -0.0039909, -0.0022032, -0.0040062, -0.0021366, -0.0014957, 0.0014432
2: 0.0016216, 0.0110959, 0.0012690, 0.0111769, -0.0076484, 0.0079264
3: -0.0063235, -0.0020112, -0.0063604, -0.0018507, -0.0036078, 0.0034812
4: 0.0008417, 0.0026755, 0.0007735, 0.0026912, -0.0014803, 0.0015341
5: 0.0009990, 0.0129152, 0.0005556, 0.0130170, -0.0096197, 0.0099694
6: -0.0017372, 0.0012873, -0.0017630, 0.0013998, -0.0025303, 0.0024416
7: -0.0076322, 0.0001930, -0.0076991, 0.0004842, -0.0065468, 0.0063171
8: -0.0035779, 0.0005373, -0.0036130, 0.0006905, -0.0034429, 0.0033221
9: -0.0024869, 0.0022849, -0.0026645, 0.0023256, -0.0038521, 0.0039922

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0030743, upper bound: 0.0030743
time: 1.77 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0030743, upper bound: 0.0030743
time: 1.82 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.9887190, 0.9964694, 0.9889695, 0.9965946, -0.0066619, 0.0059361
1: -0.0040749, -0.0021437, -0.0040125, -0.0021125, -0.0016600, 0.0014791
2: 0.0013064, 0.0115407, 0.0011410, 0.0112099, -0.0078386, 0.0087970
3: -0.0065260, -0.0018677, -0.0063754, -0.0017925, -0.0040040, 0.0035678
4: 0.0007807, 0.0027616, 0.0007487, 0.0026975, -0.0015171, 0.0017026
5: 0.0006025, 0.0134746, 0.0003946, 0.0130585, -0.0098589, 0.0110643
6: -0.0018792, 0.0013879, -0.0017736, 0.0014407, -0.0028082, 0.0025023
7: -0.0079996, 0.0004533, -0.0077264, 0.0005899, -0.0072657, 0.0064742
8: -0.0037711, 0.0006742, -0.0036274, 0.0007461, -0.0038210, 0.0034047
9: -0.0026457, 0.0025089, -0.0027290, 0.0023423, -0.0039479, 0.0044306

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0030743, upper bound: 0.0031766
time: 1.32 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0030743, upper bound: 0.0031766
time: 2.06 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.11 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 0, lower bound: -0.0030743, upper bound: 0.0030743
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 0, lower bound: -0.0030743, upper bound: 0.0030743
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 0, lower bound: -0.0030743, upper bound: 0.0031766
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 0, lower bound: -0.0030743, upper bound: 0.0031766

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.9890559, 0.9962307, 0.9890559, 0.9962307, -0.0056704, 0.0056704
1: -0.0039909, -0.0022032, -0.0039909, -0.0022032, -0.0014129, 0.0014129
2: 0.0016216, 0.0110959, 0.0016216, 0.0110959, -0.0074878, 0.0074878
3: -0.0063235, -0.0020112, -0.0063235, -0.0020112, -0.0034081, 0.0034081
4: 0.0008417, 0.0026755, 0.0008417, 0.0026755, -0.0014492, 0.0014492
5: 0.0009990, 0.0129152, 0.0009990, 0.0129152, -0.0094176, 0.0094176
6: -0.0017372, 0.0012873, -0.0017372, 0.0012873, -0.0023903, 0.0023903
7: -0.0076322, 0.0001930, -0.0076322, 0.0001930, -0.0061844, 0.0061844
8: -0.0035779, 0.0005373, -0.0035779, 0.0005373, -0.0032523, 0.0032523
9: -0.0024869, 0.0022849, -0.0024869, 0.0022849, -0.0037712, 0.0037712

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029072, upper bound: 0.0028527
time: 1.52 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029072, upper bound: 0.0029036
time: 1.16 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.9890559, 0.9962307, 0.9887190, 0.9964694, -0.0060507, 0.0061242
1: -0.0039909, -0.0022032, -0.0040749, -0.0021437, -0.0015077, 0.0015260
2: 0.0016216, 0.0110959, 0.0013064, 0.0115407, -0.0080869, 0.0079899
3: -0.0063235, -0.0020112, -0.0065260, -0.0018677, -0.0036367, 0.0036808
4: 0.0008417, 0.0026755, 0.0007807, 0.0027616, -0.0015652, 0.0015464
5: 0.0009990, 0.0129152, 0.0006025, 0.0134746, -0.0101712, 0.0100492
6: -0.0017372, 0.0012873, -0.0018792, 0.0013879, -0.0025506, 0.0025816
7: -0.0076322, 0.0001930, -0.0079996, 0.0004533, -0.0065992, 0.0066793
8: -0.0035779, 0.0005373, -0.0037711, 0.0006742, -0.0034704, 0.0035126
9: -0.0024869, 0.0022849, -0.0026457, 0.0025089, -0.0040730, 0.0040241

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029072, upper bound: 0.0028527
time: 1.70 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029072, upper bound: 0.0029036
time: 1.59 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.9887190, 0.9964694, 0.9890559, 0.9962307, -0.0061242, 0.0060507
1: -0.0040749, -0.0021437, -0.0039909, -0.0022032, -0.0015260, 0.0015077
2: 0.0013064, 0.0115407, 0.0016216, 0.0110959, -0.0079899, 0.0080869
3: -0.0065260, -0.0018677, -0.0063235, -0.0020112, -0.0036808, 0.0036367
4: 0.0007807, 0.0027616, 0.0008417, 0.0026755, -0.0015464, 0.0015652
5: 0.0006025, 0.0134746, 0.0009990, 0.0129152, -0.0100492, 0.0101712
6: -0.0018792, 0.0013879, -0.0017372, 0.0012873, -0.0025816, 0.0025506
7: -0.0079996, 0.0004533, -0.0076322, 0.0001930, -0.0066793, 0.0065992
8: -0.0037711, 0.0006742, -0.0035779, 0.0005373, -0.0035126, 0.0034704
9: -0.0026457, 0.0025089, -0.0024869, 0.0022849, -0.0040241, 0.0040730

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029036, upper bound: 0.0029306
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029036, upper bound: 0.0030248
time: 1.27 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.9887190, 0.9964694, 0.9887190, 0.9964694, -0.0058615, 0.0058615
1: -0.0040749, -0.0021437, -0.0040749, -0.0021437, -0.0014605, 0.0014605
2: 0.0013064, 0.0115407, 0.0013064, 0.0115407, -0.0077400, 0.0077400
3: -0.0065260, -0.0018677, -0.0065260, -0.0018677, -0.0035229, 0.0035229
4: 0.0007807, 0.0027616, 0.0007807, 0.0027616, -0.0014981, 0.0014981
5: 0.0006025, 0.0134746, 0.0006025, 0.0134746, -0.0097349, 0.0097349
6: -0.0018792, 0.0013879, -0.0018792, 0.0013879, -0.0024708, 0.0024708
7: -0.0079996, 0.0004533, -0.0079996, 0.0004533, -0.0063928, 0.0063928
8: -0.0037711, 0.0006742, -0.0037711, 0.0006742, -0.0033619, 0.0033619
9: -0.0026457, 0.0025089, -0.0026457, 0.0025089, -0.0038983, 0.0038983

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029036, upper bound: 0.0029339
time: 1.61 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029036, upper bound: 0.0030291
time: 1.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.83 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -0.0029072, upper bound: 0.0028527
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -0.0029072, upper bound: 0.0029036
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -0.0029072, upper bound: 0.0028527
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -0.0029072, upper bound: 0.0029036
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -0.0029036, upper bound: 0.0029306
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -0.0029036, upper bound: 0.0030248
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -0.0029036, upper bound: 0.0029339
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -0.0029036, upper bound: 0.0030291

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9890660, 0.9960648, 0.9890559, 0.9962307, -0.0056612, 0.0054871
1: -0.0039884, -0.0022445, -0.0039909, -0.0022032, -0.0014106, 0.0013672
2: 0.0018407, 0.0110825, 0.0016216, 0.0110959, -0.0072456, 0.0074756
3: -0.0063174, -0.0021109, -0.0063235, -0.0020112, -0.0034026, 0.0032979
4: 0.0008841, 0.0026729, 0.0008417, 0.0026755, -0.0014024, 0.0014469
5: 0.0012745, 0.0128983, 0.0009990, 0.0129152, -0.0091131, 0.0094023
6: -0.0017329, 0.0012173, -0.0017372, 0.0012873, -0.0023864, 0.0023130
7: -0.0076212, 0.0000120, -0.0076322, 0.0001930, -0.0061744, 0.0059844
8: -0.0035720, 0.0004422, -0.0035779, 0.0005373, -0.0032470, 0.0031471
9: -0.0023766, 0.0022781, -0.0024869, 0.0022849, -0.0036493, 0.0037651

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028531, upper bound: 0.0028531
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028531, upper bound: 0.0028531
time: 1.72 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9883941, 0.9957968, 0.9890687, 0.9960327, -0.0062421, 0.0055094
1: -0.0041558, -0.0023113, -0.0039877, -0.0022525, -0.0015554, 0.0013728
2: 0.0021946, 0.0119698, 0.0018831, 0.0110789, -0.0072751, 0.0082426
3: -0.0067212, -0.0022720, -0.0063158, -0.0021302, -0.0037517, 0.0033113
4: 0.0009526, 0.0028446, 0.0008924, 0.0026722, -0.0014081, 0.0015953
5: 0.0017197, 0.0140143, 0.0013279, 0.0128938, -0.0091502, 0.0103670
6: -0.0020161, 0.0011044, -0.0017318, 0.0012038, -0.0026313, 0.0023224
7: -0.0083540, -0.0002803, -0.0076182, -0.0000230, -0.0068079, 0.0060088
8: -0.0039574, 0.0002884, -0.0035705, 0.0004237, -0.0035802, 0.0031600
9: -0.0021983, 0.0027250, -0.0023552, 0.0022763, -0.0036641, 0.0041514

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026923, upper bound: 0.0027214
time: 1.23 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027157, upper bound: 0.0027157
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9890660, 0.9960648, 0.9887190, 0.9964694, -0.0060415, 0.0059408
1: -0.0039884, -0.0022445, -0.0040749, -0.0021437, -0.0015054, 0.0014803
2: 0.0018407, 0.0110825, 0.0013064, 0.0115407, -0.0078447, 0.0079777
3: -0.0063174, -0.0021109, -0.0065260, -0.0018677, -0.0036311, 0.0035706
4: 0.0008841, 0.0026729, 0.0007807, 0.0027616, -0.0015183, 0.0015441
5: 0.0012745, 0.0128983, 0.0006025, 0.0134746, -0.0098666, 0.0100339
6: -0.0017329, 0.0012173, -0.0018792, 0.0013879, -0.0025467, 0.0025042
7: -0.0076212, 0.0000120, -0.0079996, 0.0004533, -0.0065891, 0.0064792
8: -0.0035720, 0.0004422, -0.0037711, 0.0006742, -0.0034652, 0.0034074
9: -0.0023766, 0.0022781, -0.0026457, 0.0025089, -0.0039510, 0.0040180

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029306, upper bound: 0.0028527
time: 1.17 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029306, upper bound: 0.0028527
time: 1.74 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9883941, 0.9957968, 0.9887316, 0.9963004, -0.0066023, 0.0059632
1: -0.0041558, -0.0023113, -0.0040718, -0.0021858, -0.0016451, 0.0014859
2: 0.0021946, 0.0119698, 0.0015295, 0.0115242, -0.0078743, 0.0087183
3: -0.0067212, -0.0022720, -0.0065184, -0.0019693, -0.0039682, 0.0035841
4: 0.0009526, 0.0028446, 0.0008239, 0.0027584, -0.0015241, 0.0016874
5: 0.0017197, 0.0140143, 0.0008832, 0.0134538, -0.0099038, 0.0109653
6: -0.0020161, 0.0011044, -0.0018739, 0.0013167, -0.0027831, 0.0025137
7: -0.0083540, -0.0002803, -0.0079860, 0.0002690, -0.0072008, 0.0065037
8: -0.0039574, 0.0002884, -0.0037639, 0.0005773, -0.0037868, 0.0034202
9: -0.0021983, 0.0027250, -0.0025333, 0.0025006, -0.0039659, 0.0043910

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027957, upper bound: 0.0027212
time: 1.53 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028381, upper bound: 0.0027157
time: 1.50 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9887283, 0.9962865, 0.9890559, 0.9962307, -0.0061155, 0.0058652
1: -0.0040725, -0.0021893, -0.0039909, -0.0022032, -0.0015238, 0.0014615
2: 0.0015481, 0.0115284, 0.0016216, 0.0110959, -0.0077450, 0.0080754
3: -0.0065203, -0.0019777, -0.0063235, -0.0020112, -0.0036756, 0.0035252
4: 0.0008275, 0.0027592, 0.0008417, 0.0026755, -0.0014990, 0.0015630
5: 0.0009065, 0.0134591, 0.0009990, 0.0129152, -0.0097411, 0.0101567
6: -0.0018752, 0.0013108, -0.0017372, 0.0012873, -0.0025779, 0.0024724
7: -0.0079894, 0.0002537, -0.0076322, 0.0001930, -0.0066698, 0.0063969
8: -0.0037657, 0.0005693, -0.0035779, 0.0005373, -0.0035076, 0.0033640
9: -0.0025239, 0.0025027, -0.0024869, 0.0022849, -0.0039008, 0.0040672

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028527, upper bound: 0.0029306
time: 1.40 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028527, upper bound: 0.0029306
time: 1.66 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9879990, 0.9960977, 0.9890687, 0.9960327, -0.0067613, 0.0058917
1: -0.0042543, -0.0022363, -0.0039877, -0.0022525, -0.0016847, 0.0014681
2: 0.0017972, 0.0124915, 0.0018831, 0.0110789, -0.0077799, 0.0089282
3: -0.0069587, -0.0020911, -0.0063158, -0.0021302, -0.0040637, 0.0035411
4: 0.0008757, 0.0029456, 0.0008924, 0.0026722, -0.0015058, 0.0017280
5: 0.0012199, 0.0146704, 0.0013279, 0.0128938, -0.0097851, 0.0112293
6: -0.0021827, 0.0012312, -0.0017318, 0.0012038, -0.0028501, 0.0024836
7: -0.0087849, 0.0000479, -0.0076182, -0.0000230, -0.0073741, 0.0064257
8: -0.0041840, 0.0004610, -0.0035705, 0.0004237, -0.0038780, 0.0033792
9: -0.0023985, 0.0029877, -0.0023552, 0.0022763, -0.0039184, 0.0044967

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026923, upper bound: 0.0028458
time: 1.23 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027157, upper bound: 0.0028381
time: 1.24 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9887283, 0.9962865, 0.9887190, 0.9964694, -0.0058521, 0.0056817
1: -0.0040725, -0.0021893, -0.0040749, -0.0021437, -0.0014582, 0.0014157
2: 0.0015481, 0.0115284, 0.0013064, 0.0115407, -0.0075027, 0.0077276
3: -0.0065203, -0.0019777, -0.0065260, -0.0018677, -0.0035173, 0.0034149
4: 0.0008275, 0.0027592, 0.0007807, 0.0027616, -0.0014521, 0.0014957
5: 0.0009065, 0.0134591, 0.0006025, 0.0134746, -0.0094364, 0.0097193
6: -0.0018752, 0.0013108, -0.0018792, 0.0013879, -0.0024669, 0.0023951
7: -0.0079894, 0.0002537, -0.0079996, 0.0004533, -0.0063825, 0.0061967
8: -0.0037657, 0.0005693, -0.0037711, 0.0006742, -0.0033565, 0.0032588
9: -0.0025239, 0.0025027, -0.0026457, 0.0025089, -0.0037787, 0.0038920

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028589, upper bound: 0.0029339
time: 1.61 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028589, upper bound: 0.0029339
time: 1.69 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9879990, 0.9960977, 0.9887316, 0.9963004, -0.0065152, 0.0057243
1: -0.0042543, -0.0022363, -0.0040718, -0.0021858, -0.0016234, 0.0014264
2: 0.0017972, 0.0124915, 0.0015295, 0.0115242, -0.0075589, 0.0086032
3: -0.0069587, -0.0020911, -0.0065184, -0.0019693, -0.0039158, 0.0034405
4: 0.0008757, 0.0029456, 0.0008239, 0.0027584, -0.0014630, 0.0016651
5: 0.0012199, 0.0146704, 0.0008832, 0.0134538, -0.0095071, 0.0108206
6: -0.0021827, 0.0012312, -0.0018739, 0.0013167, -0.0027464, 0.0024130
7: -0.0087849, 0.0000479, -0.0079860, 0.0002690, -0.0071057, 0.0062432
8: -0.0041840, 0.0004610, -0.0037639, 0.0005773, -0.0037368, 0.0032832
9: -0.0023985, 0.0029877, -0.0025333, 0.0025006, -0.0038071, 0.0043330

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027159, upper bound: 0.0028566
time: 1.29 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027469, upper bound: 0.0028528
time: 1.62 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.48 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0028531, upper bound: 0.0028531
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0028531, upper bound: 0.0028531
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0026923, upper bound: 0.0027214
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0027157, upper bound: 0.0027157
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0029306, upper bound: 0.0028527
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0029306, upper bound: 0.0028527
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0027957, upper bound: 0.0027212
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0028381, upper bound: 0.0027157
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0028527, upper bound: 0.0029306
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0028527, upper bound: 0.0029306
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0026923, upper bound: 0.0028458
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0027157, upper bound: 0.0028381
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0028589, upper bound: 0.0029339
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0028589, upper bound: 0.0029339
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0027159, upper bound: 0.0028566
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -0.0027469, upper bound: 0.0028528

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9890660, 0.9960648, 0.9890660, 0.9960648, -0.0054778, 0.0054778
1: -0.0039884, -0.0022445, -0.0039884, -0.0022445, -0.0013649, 0.0013649
2: 0.0018407, 0.0110825, 0.0018407, 0.0110825, -0.0072334, 0.0072334
3: -0.0063174, -0.0021109, -0.0063174, -0.0021109, -0.0032923, 0.0032923
4: 0.0008841, 0.0026729, 0.0008841, 0.0026729, -0.0014000, 0.0014000
5: 0.0012745, 0.0128983, 0.0012745, 0.0128983, -0.0090978, 0.0090978
6: -0.0017329, 0.0012173, -0.0017329, 0.0012173, -0.0023091, 0.0023091
7: -0.0076212, 0.0000120, -0.0076212, 0.0000120, -0.0059744, 0.0059744
8: -0.0035720, 0.0004422, -0.0035720, 0.0004422, -0.0031419, 0.0031419
9: -0.0023766, 0.0022781, -0.0023766, 0.0022781, -0.0036431, 0.0036431

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027178, upper bound: 0.0026555
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027178, upper bound: 0.0026926
time: 1.61 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9890660, 0.9960648, 0.9883941, 0.9957968, -0.0053850, 0.0062092
1: -0.0039884, -0.0022445, -0.0041558, -0.0023113, -0.0013418, 0.0015472
2: 0.0018407, 0.0110825, 0.0021946, 0.0119698, -0.0081992, 0.0071108
3: -0.0063174, -0.0021109, -0.0067212, -0.0022720, -0.0032365, 0.0037319
4: 0.0008841, 0.0026729, 0.0009526, 0.0028446, -0.0015869, 0.0013763
5: 0.0012745, 0.0128983, 0.0017197, 0.0140143, -0.0103125, 0.0089436
6: -0.0017329, 0.0012173, -0.0020161, 0.0011044, -0.0022700, 0.0026174
7: -0.0076212, 0.0000120, -0.0083540, -0.0002803, -0.0058731, 0.0067720
8: -0.0035720, 0.0004422, -0.0039574, 0.0002884, -0.0030886, 0.0035614
9: -0.0023766, 0.0022781, -0.0021983, 0.0027250, -0.0041296, 0.0035814

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027178, upper bound: 0.0026555
time: 1.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027178, upper bound: 0.0026926
time: 1.26 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9883967, 0.9957643, 0.9890872, 0.9958423, -0.0060305, 0.0054113
1: -0.0041552, -0.0023194, -0.0039831, -0.0022999, -0.0015026, 0.0013483
2: 0.0022375, 0.0119662, 0.0021345, 0.0110546, -0.0071455, 0.0079632
3: -0.0067196, -0.0022915, -0.0063047, -0.0022446, -0.0036245, 0.0032523
4: 0.0009610, 0.0028439, 0.0009410, 0.0026675, -0.0013830, 0.0015413
5: 0.0017736, 0.0140098, 0.0016440, 0.0128632, -0.0089872, 0.0100157
6: -0.0020150, 0.0010907, -0.0017240, 0.0011236, -0.0025421, 0.0022811
7: -0.0083511, -0.0003157, -0.0075981, -0.0002306, -0.0065771, 0.0059018
8: -0.0039559, 0.0002698, -0.0035599, 0.0003146, -0.0034589, 0.0031037
9: -0.0021767, 0.0027232, -0.0022286, 0.0022640, -0.0035989, 0.0040107

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0025992
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0026536
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9884073, 0.9956402, 0.9885390, 0.9957395, -0.0061116, 0.0059495
1: -0.0041526, -0.0023503, -0.0041197, -0.0023256, -0.0015229, 0.0014825
2: 0.0024014, 0.0119524, 0.0022702, 0.0117784, -0.0078562, 0.0080703
3: -0.0067133, -0.0023662, -0.0066341, -0.0023064, -0.0036733, 0.0035758
4: 0.0009927, 0.0028412, 0.0009673, 0.0028076, -0.0015206, 0.0015620
5: 0.0019798, 0.0139924, 0.0018148, 0.0137736, -0.0098811, 0.0101504
6: -0.0020106, 0.0010383, -0.0019551, 0.0010802, -0.0025763, 0.0025079
7: -0.0083396, -0.0004512, -0.0081959, -0.0003428, -0.0066656, 0.0064888
8: -0.0039499, 0.0001986, -0.0038743, 0.0002556, -0.0035054, 0.0034124
9: -0.0020941, 0.0027162, -0.0021602, 0.0026286, -0.0039568, 0.0040647

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0025921
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0026448
time: 1.53 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9890660, 0.9960648, 0.9887283, 0.9962865, -0.0058560, 0.0059321
1: -0.0039884, -0.0022445, -0.0040725, -0.0021893, -0.0014592, 0.0014781
2: 0.0018407, 0.0110825, 0.0015481, 0.0115284, -0.0078332, 0.0077328
3: -0.0063174, -0.0021109, -0.0065203, -0.0019777, -0.0035196, 0.0035654
4: 0.0008841, 0.0026729, 0.0008275, 0.0027592, -0.0015161, 0.0014967
5: 0.0012745, 0.0128983, 0.0009065, 0.0134591, -0.0098522, 0.0097258
6: -0.0017329, 0.0012173, -0.0018752, 0.0013108, -0.0024685, 0.0025006
7: -0.0076212, 0.0000120, -0.0079894, 0.0002537, -0.0063868, 0.0064698
8: -0.0035720, 0.0004422, -0.0037657, 0.0005693, -0.0033588, 0.0034024
9: -0.0023766, 0.0022781, -0.0025239, 0.0025027, -0.0039452, 0.0038947

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028318, upper bound: 0.0026555
time: 1.56 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028318, upper bound: 0.0026926
time: 1.51 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9890660, 0.9960648, 0.9879990, 0.9960977, -0.0057524, 0.0067253
1: -0.0039884, -0.0022445, -0.0042543, -0.0022363, -0.0014333, 0.0016758
2: 0.0018407, 0.0110825, 0.0017972, 0.0124915, -0.0088806, 0.0075959
3: -0.0063174, -0.0021109, -0.0069587, -0.0020911, -0.0034573, 0.0040421
4: 0.0008841, 0.0026729, 0.0008757, 0.0029456, -0.0017188, 0.0014702
5: 0.0012745, 0.0128983, 0.0012199, 0.0146704, -0.0111695, 0.0095537
6: -0.0017329, 0.0012173, -0.0021827, 0.0012312, -0.0024248, 0.0028349
7: -0.0076212, 0.0000120, -0.0087849, 0.0000479, -0.0062738, 0.0073349
8: -0.0035720, 0.0004422, -0.0041840, 0.0004610, -0.0032993, 0.0038573
9: -0.0023766, 0.0022781, -0.0023985, 0.0029877, -0.0044728, 0.0038257

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028318, upper bound: 0.0026555
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028318, upper bound: 0.0026926
time: 1.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9883967, 0.9957643, 0.9887581, 0.9961005, -0.0063858, 0.0059091
1: -0.0041552, -0.0023194, -0.0040651, -0.0022356, -0.0015912, 0.0014724
2: 0.0022375, 0.0119662, 0.0017936, 0.0114891, -0.0078030, 0.0084324
3: -0.0067196, -0.0022915, -0.0065025, -0.0020895, -0.0038381, 0.0035516
4: 0.0009610, 0.0028439, 0.0008750, 0.0027516, -0.0015102, 0.0016321
5: 0.0017736, 0.0140098, 0.0012153, 0.0134097, -0.0098141, 0.0106058
6: -0.0020150, 0.0010907, -0.0018627, 0.0012324, -0.0026919, 0.0024909
7: -0.0083511, -0.0003157, -0.0079570, 0.0000509, -0.0069647, 0.0064448
8: -0.0039559, 0.0002698, -0.0037487, 0.0004626, -0.0036626, 0.0033892
9: -0.0021767, 0.0027232, -0.0024003, 0.0024829, -0.0039300, 0.0042470

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027294, upper bound: 0.0025992
time: 1.56 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027294, upper bound: 0.0026533
time: 1.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9884073, 0.9956402, 0.9881898, 0.9960179, -0.0064523, 0.0064004
1: -0.0041526, -0.0023503, -0.0042067, -0.0022562, -0.0016078, 0.0015948
2: 0.0024014, 0.0119524, 0.0019026, 0.0122394, -0.0084516, 0.0085203
3: -0.0067133, -0.0023662, -0.0068440, -0.0021391, -0.0038781, 0.0038468
4: 0.0009927, 0.0028412, 0.0008961, 0.0028968, -0.0016358, 0.0016491
5: 0.0019798, 0.0139924, 0.0013524, 0.0143534, -0.0106299, 0.0107162
6: -0.0020106, 0.0010383, -0.0021022, 0.0011976, -0.0027199, 0.0026980
7: -0.0083396, -0.0004512, -0.0085767, -0.0000391, -0.0070372, 0.0069805
8: -0.0039499, 0.0001986, -0.0040745, 0.0004153, -0.0037008, 0.0036710
9: -0.0020941, 0.0027162, -0.0023454, 0.0028608, -0.0042567, 0.0042913

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027666, upper bound: 0.0025921
time: 1.63 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027665, upper bound: 0.0026448
time: 1.63 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9887283, 0.9962865, 0.9890660, 0.9960648, -0.0059321, 0.0058560
1: -0.0040725, -0.0021893, -0.0039884, -0.0022445, -0.0014781, 0.0014592
2: 0.0015481, 0.0115284, 0.0018407, 0.0110825, -0.0077328, 0.0078332
3: -0.0065203, -0.0019777, -0.0063174, -0.0021109, -0.0035654, 0.0035196
4: 0.0008275, 0.0027592, 0.0008841, 0.0026729, -0.0014967, 0.0015161
5: 0.0009065, 0.0134591, 0.0012745, 0.0128983, -0.0097258, 0.0098522
6: -0.0018752, 0.0013108, -0.0017329, 0.0012173, -0.0025006, 0.0024685
7: -0.0079894, 0.0002537, -0.0076212, 0.0000120, -0.0064698, 0.0063868
8: -0.0037657, 0.0005693, -0.0035720, 0.0004422, -0.0034024, 0.0033588
9: -0.0025239, 0.0025027, -0.0023766, 0.0022781, -0.0038947, 0.0039452

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027171, upper bound: 0.0027220
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027171, upper bound: 0.0027815
time: 1.59 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9887283, 0.9962865, 0.9883941, 0.9957968, -0.0058392, 0.0065729
1: -0.0040725, -0.0021893, -0.0041558, -0.0023113, -0.0014550, 0.0016378
2: 0.0015481, 0.0115284, 0.0021946, 0.0119698, -0.0086794, 0.0077106
3: -0.0065203, -0.0019777, -0.0067212, -0.0022720, -0.0035096, 0.0039505
4: 0.0008275, 0.0027592, 0.0009526, 0.0028446, -0.0016799, 0.0014924
5: 0.0009065, 0.0134591, 0.0017197, 0.0140143, -0.0109164, 0.0096980
6: -0.0018752, 0.0013108, -0.0020161, 0.0011044, -0.0024614, 0.0027707
7: -0.0079894, 0.0002537, -0.0083540, -0.0002803, -0.0063685, 0.0071687
8: -0.0037657, 0.0005693, -0.0039574, 0.0002884, -0.0033491, 0.0037699
9: -0.0025239, 0.0025027, -0.0021983, 0.0027250, -0.0043714, 0.0038835

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027171, upper bound: 0.0027220
time: 1.72 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027171, upper bound: 0.0027815
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9880032, 0.9960642, 0.9890872, 0.9958423, -0.0065480, 0.0057770
1: -0.0042532, -0.0022446, -0.0039831, -0.0022999, -0.0016316, 0.0014395
2: 0.0018414, 0.0124860, 0.0021345, 0.0110546, -0.0076285, 0.0086466
3: -0.0069562, -0.0021113, -0.0063047, -0.0022446, -0.0039356, 0.0034721
4: 0.0008843, 0.0029445, 0.0009410, 0.0026675, -0.0014765, 0.0016735
5: 0.0012755, 0.0146635, 0.0016440, 0.0128632, -0.0095946, 0.0108752
6: -0.0021809, 0.0012171, -0.0017240, 0.0011236, -0.0027602, 0.0024352
7: -0.0087803, 0.0000114, -0.0075981, -0.0002306, -0.0071416, 0.0063006
8: -0.0041816, 0.0004418, -0.0035599, 0.0003146, -0.0037557, 0.0033134
9: -0.0023762, 0.0029850, -0.0022286, 0.0022640, -0.0038421, 0.0043549

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027193
time: 1.77 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027787
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9880158, 0.9959484, 0.9885390, 0.9957395, -0.0065261, 0.0063196
1: -0.0042501, -0.0022735, -0.0041197, -0.0023256, -0.0016261, 0.0015747
2: 0.0019945, 0.0124694, 0.0022702, 0.0117784, -0.0083450, 0.0086176
3: -0.0069487, -0.0021809, -0.0066341, -0.0023064, -0.0039223, 0.0037983
4: 0.0009139, 0.0029413, 0.0009673, 0.0028076, -0.0016152, 0.0016679
5: 0.0014680, 0.0146427, 0.0018148, 0.0137736, -0.0104958, 0.0108387
6: -0.0021756, 0.0011682, -0.0019551, 0.0010802, -0.0027510, 0.0026639
7: -0.0087667, -0.0001150, -0.0081959, -0.0003428, -0.0071176, 0.0068924
8: -0.0041745, 0.0003754, -0.0038743, 0.0002556, -0.0037431, 0.0036247
9: -0.0022991, 0.0029766, -0.0021602, 0.0026286, -0.0042030, 0.0043403

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0027090
time: 1.55 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0027665
time: 1.19 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9887283, 0.9962865, 0.9887283, 0.9962865, -0.0056723, 0.0056723
1: -0.0040725, -0.0021893, -0.0040725, -0.0021893, -0.0014134, 0.0014134
2: 0.0015481, 0.0115284, 0.0015481, 0.0115284, -0.0074902, 0.0074902
3: -0.0065203, -0.0019777, -0.0065203, -0.0019777, -0.0034092, 0.0034092
4: 0.0008275, 0.0027592, 0.0008275, 0.0027592, -0.0014497, 0.0014497
5: 0.0009065, 0.0134591, 0.0009065, 0.0134591, -0.0094207, 0.0094207
6: -0.0018752, 0.0013108, -0.0018752, 0.0013108, -0.0023911, 0.0023911
7: -0.0079894, 0.0002537, -0.0079894, 0.0002537, -0.0061865, 0.0061865
8: -0.0037657, 0.0005693, -0.0037657, 0.0005693, -0.0032534, 0.0032534
9: -0.0025239, 0.0025027, -0.0025239, 0.0025027, -0.0037725, 0.0037725

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027332, upper bound: 0.0027284
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027332, upper bound: 0.0027892
time: 1.66 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9887283, 0.9962865, 0.9879990, 0.9960977, -0.0055679, 0.0064851
1: -0.0040725, -0.0021893, -0.0042543, -0.0022363, -0.0013874, 0.0016159
2: 0.0015481, 0.0115284, 0.0017972, 0.0124915, -0.0085636, 0.0073524
3: -0.0065203, -0.0019777, -0.0069587, -0.0020911, -0.0033465, 0.0038978
4: 0.0008275, 0.0027592, 0.0008757, 0.0029456, -0.0016575, 0.0014230
5: 0.0009065, 0.0134591, 0.0012199, 0.0146704, -0.0107707, 0.0092473
6: -0.0018752, 0.0013108, -0.0021827, 0.0012312, -0.0023471, 0.0027337
7: -0.0079894, 0.0002537, -0.0087849, 0.0000479, -0.0060726, 0.0070730
8: -0.0037657, 0.0005693, -0.0041840, 0.0004610, -0.0031935, 0.0037196
9: -0.0025239, 0.0025027, -0.0023985, 0.0029877, -0.0043131, 0.0037030

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027332, upper bound: 0.0027284
time: 1.78 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027332, upper bound: 0.0027892
time: 1.72 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9880032, 0.9960642, 0.9887581, 0.9961005, -0.0063047, 0.0056693
1: -0.0042532, -0.0022446, -0.0040651, -0.0022356, -0.0015710, 0.0014126
2: 0.0018414, 0.0124860, 0.0017936, 0.0114891, -0.0074862, 0.0083253
3: -0.0069562, -0.0021113, -0.0065025, -0.0020895, -0.0037893, 0.0034074
4: 0.0008843, 0.0029445, 0.0008750, 0.0027516, -0.0014489, 0.0016113
5: 0.0012755, 0.0146635, 0.0012153, 0.0134097, -0.0094157, 0.0104710
6: -0.0021809, 0.0012171, -0.0018627, 0.0012324, -0.0026577, 0.0023898
7: -0.0087803, 0.0000114, -0.0079570, 0.0000509, -0.0068762, 0.0061831
8: -0.0041816, 0.0004418, -0.0037487, 0.0004626, -0.0036161, 0.0032516
9: -0.0023762, 0.0029850, -0.0024003, 0.0024829, -0.0037704, 0.0041931

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026456, upper bound: 0.0027263
time: 1.27 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026456, upper bound: 0.0027867
time: 1.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9880158, 0.9959484, 0.9881898, 0.9960179, -0.0063306, 0.0061750
1: -0.0042501, -0.0022735, -0.0042067, -0.0022562, -0.0015774, 0.0015386
2: 0.0019945, 0.0124694, 0.0019026, 0.0122394, -0.0081540, 0.0083595
3: -0.0069487, -0.0021809, -0.0068440, -0.0021391, -0.0038049, 0.0037113
4: 0.0009139, 0.0029413, 0.0008961, 0.0028968, -0.0015782, 0.0016180
5: 0.0014680, 0.0146427, 0.0013524, 0.0143534, -0.0102556, 0.0105141
6: -0.0021756, 0.0011682, -0.0021022, 0.0011976, -0.0026686, 0.0026030
7: -0.0087667, -0.0001150, -0.0085767, -0.0000391, -0.0069044, 0.0067347
8: -0.0041745, 0.0003754, -0.0040745, 0.0004153, -0.0036310, 0.0035417
9: -0.0022991, 0.0029766, -0.0023454, 0.0028608, -0.0041068, 0.0042103

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026712, upper bound: 0.0027196
time: 1.29 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026714, upper bound: 0.0027787
time: 1.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.50 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027178, upper bound: 0.0026555
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027178, upper bound: 0.0026926
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027178, upper bound: 0.0026555
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027178, upper bound: 0.0026926
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0025992
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0026536
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0025921
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0026448
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0028318, upper bound: 0.0026555
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0028318, upper bound: 0.0026926
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0028318, upper bound: 0.0026555
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0028318, upper bound: 0.0026926
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027294, upper bound: 0.0025992
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027294, upper bound: 0.0026533
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027666, upper bound: 0.0025921
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027665, upper bound: 0.0026448
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027171, upper bound: 0.0027220
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027171, upper bound: 0.0027815
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027171, upper bound: 0.0027220
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027171, upper bound: 0.0027815
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027193
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027787
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0027090
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0027665
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027332, upper bound: 0.0027284
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027332, upper bound: 0.0027892
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027332, upper bound: 0.0027284
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0027332, upper bound: 0.0027892
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0026456, upper bound: 0.0027263
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0026456, upper bound: 0.0027867
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0026712, upper bound: 0.0027196
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.50
Output dim: 0, lower bound: -0.0026714, upper bound: 0.0027787

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9890853, 0.9958748, 0.9890710, 0.9960322, -0.0053532, 0.0052649
1: -0.0039836, -0.0022919, -0.0039872, -0.0022526, -0.0013339, 0.0013119
2: 0.0020916, 0.0110570, 0.0018837, 0.0110761, -0.0069522, 0.0070689
3: -0.0063058, -0.0022251, -0.0063145, -0.0021305, -0.0032175, 0.0031643
4: 0.0009327, 0.0026679, 0.0008925, 0.0026716, -0.0013456, 0.0013682
5: 0.0015902, 0.0128663, 0.0013287, 0.0128902, -0.0087440, 0.0088908
6: -0.0017248, 0.0011372, -0.0017308, 0.0012036, -0.0022566, 0.0022193
7: -0.0076001, -0.0001953, -0.0076158, -0.0000236, -0.0058385, 0.0057421
8: -0.0035610, 0.0003332, -0.0035692, 0.0004235, -0.0030704, 0.0030197
9: -0.0022502, 0.0022653, -0.0023549, 0.0022749, -0.0035015, 0.0035603

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026862, upper bound: 0.0026732
time: 1.27 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027329, upper bound: 0.0026732
time: 1.23 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9885371, 0.9957764, 0.9890821, 0.9959106, -0.0058907, 0.0052733
1: -0.0041202, -0.0023164, -0.0039844, -0.0022829, -0.0014678, 0.0013140
2: 0.0022216, 0.0117809, 0.0020444, 0.0110612, -0.0069633, 0.0077787
3: -0.0066353, -0.0022843, -0.0063077, -0.0022036, -0.0035405, 0.0031694
4: 0.0009579, 0.0028081, 0.0009236, 0.0026688, -0.0013477, 0.0015055
5: 0.0017536, 0.0137768, 0.0015307, 0.0128716, -0.0087580, 0.0097835
6: -0.0019559, 0.0010957, -0.0017261, 0.0011523, -0.0024832, 0.0022229
7: -0.0081980, -0.0003026, -0.0076036, -0.0001562, -0.0064247, 0.0057512
8: -0.0038754, 0.0002767, -0.0035628, 0.0003537, -0.0033787, 0.0030245
9: -0.0021847, 0.0026299, -0.0022740, 0.0022674, -0.0035071, 0.0039177

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026862, upper bound: 0.0027329
time: 1.46 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027329, upper bound: 0.0027329
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9890853, 0.9958748, 0.9883967, 0.9957643, -0.0052499, 0.0060001
1: -0.0039836, -0.0022919, -0.0041552, -0.0023194, -0.0013081, 0.0014951
2: 0.0020916, 0.0110570, 0.0022375, 0.0119662, -0.0079230, 0.0069324
3: -0.0063058, -0.0022251, -0.0067196, -0.0022915, -0.0031553, 0.0036062
4: 0.0009327, 0.0026679, 0.0009610, 0.0028439, -0.0015335, 0.0013418
5: 0.0015902, 0.0128663, 0.0017736, 0.0140098, -0.0099651, 0.0087192
6: -0.0017248, 0.0011372, -0.0020150, 0.0010907, -0.0022130, 0.0025292
7: -0.0076001, -0.0001953, -0.0083511, -0.0003157, -0.0057258, 0.0065439
8: -0.0035610, 0.0003332, -0.0039559, 0.0002698, -0.0030111, 0.0034414
9: -0.0022502, 0.0022653, -0.0021767, 0.0027232, -0.0039905, 0.0034915

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026032, upper bound: 0.0025882
time: 1.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026564, upper bound: 0.0025882
time: 1.33 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9885371, 0.9957764, 0.9884073, 0.9956402, -0.0057882, 0.0060816
1: -0.0041202, -0.0023164, -0.0041526, -0.0023503, -0.0014423, 0.0015154
2: 0.0022216, 0.0117809, 0.0024014, 0.0119524, -0.0080307, 0.0076433
3: -0.0066353, -0.0022843, -0.0067133, -0.0023662, -0.0034789, 0.0036552
4: 0.0009579, 0.0028081, 0.0009927, 0.0028412, -0.0015543, 0.0014793
5: 0.0017536, 0.0137768, 0.0019798, 0.0139924, -0.0101005, 0.0096133
6: -0.0019559, 0.0010957, -0.0020106, 0.0010383, -0.0024400, 0.0025636
7: -0.0081980, -0.0003026, -0.0083396, -0.0004512, -0.0063129, 0.0066328
8: -0.0038754, 0.0002767, -0.0039499, 0.0001986, -0.0033199, 0.0034881
9: -0.0021847, 0.0026299, -0.0020941, 0.0027162, -0.0040447, 0.0038496

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026015, upper bound: 0.0026220
time: 1.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026547, upper bound: 0.0026220
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9882876, 0.9956000, 0.9890923, 0.9957647, -0.0059769, 0.0052488
1: -0.0041824, -0.0023603, -0.0039819, -0.0023193, -0.0014893, 0.0013079
2: 0.0024544, 0.0121104, 0.0022370, 0.0110478, -0.0069310, 0.0078924
3: -0.0067852, -0.0023903, -0.0063016, -0.0022913, -0.0035923, 0.0031547
4: 0.0010029, 0.0028718, 0.0009609, 0.0026662, -0.0013415, 0.0015276
5: 0.0020464, 0.0141911, 0.0017730, 0.0128547, -0.0087174, 0.0099265
6: -0.0020610, 0.0010214, -0.0017218, 0.0010908, -0.0025195, 0.0022126
7: -0.0084701, -0.0004949, -0.0075925, -0.0003153, -0.0065186, 0.0057246
8: -0.0040185, 0.0001756, -0.0035570, 0.0002700, -0.0034281, 0.0030105
9: -0.0020675, 0.0027958, -0.0021770, 0.0022607, -0.0034908, 0.0039750

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026536
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026536
time: 1.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9884210, 0.9953738, 0.9885445, 0.9956250, -0.0059798, 0.0057050
1: -0.0041491, -0.0024167, -0.0041183, -0.0023541, -0.0014900, 0.0014215
2: 0.0027531, 0.0119343, 0.0024214, 0.0117711, -0.0075334, 0.0078963
3: -0.0067051, -0.0025262, -0.0066308, -0.0023752, -0.0035940, 0.0034289
4: 0.0010607, 0.0028377, 0.0009965, 0.0028062, -0.0014581, 0.0015283
5: 0.0024221, 0.0139696, 0.0020049, 0.0137644, -0.0094750, 0.0099314
6: -0.0020048, 0.0009261, -0.0019527, 0.0010320, -0.0025207, 0.0024049
7: -0.0083247, -0.0007416, -0.0081899, -0.0004676, -0.0065218, 0.0062221
8: -0.0039420, 0.0000459, -0.0038711, 0.0001899, -0.0034298, 0.0032721
9: -0.0019170, 0.0027071, -0.0020841, 0.0026249, -0.0037942, 0.0039770

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0025921
time: 1.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0025921
time: 2.00 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9882984, 0.9954761, 0.9885438, 0.9956619, -0.0060610, 0.0057858
1: -0.0041797, -0.0023912, -0.0041185, -0.0023449, -0.0015102, 0.0014417
2: 0.0026181, 0.0120962, 0.0023726, 0.0117720, -0.0076401, 0.0080034
3: -0.0067788, -0.0024648, -0.0066312, -0.0023530, -0.0036428, 0.0034775
4: 0.0010346, 0.0028691, 0.0009871, 0.0028063, -0.0014787, 0.0015490
5: 0.0022523, 0.0141732, 0.0019436, 0.0137656, -0.0096093, 0.0100662
6: -0.0020565, 0.0009692, -0.0019530, 0.0010475, -0.0025549, 0.0024389
7: -0.0084584, -0.0006301, -0.0081907, -0.0004274, -0.0066103, 0.0063103
8: -0.0040123, 0.0001045, -0.0038715, 0.0002111, -0.0034763, 0.0033185
9: -0.0019850, 0.0027886, -0.0021086, 0.0026254, -0.0038480, 0.0040310

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026448
time: 1.31 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026448
time: 1.20 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9890853, 0.9958748, 0.9887328, 0.9962528, -0.0057157, 0.0057193
1: -0.0039836, -0.0022919, -0.0040715, -0.0021977, -0.0014242, 0.0014251
2: 0.0020916, 0.0110570, 0.0015925, 0.0115226, -0.0075523, 0.0075475
3: -0.0063058, -0.0022251, -0.0065177, -0.0019979, -0.0034353, 0.0034375
4: 0.0009327, 0.0026679, 0.0008361, 0.0027581, -0.0014617, 0.0014608
5: 0.0015902, 0.0128663, 0.0009624, 0.0134518, -0.0094988, 0.0094928
6: -0.0017248, 0.0011372, -0.0018734, 0.0012966, -0.0024094, 0.0024109
7: -0.0076001, -0.0001953, -0.0079846, 0.0002170, -0.0062338, 0.0062377
8: -0.0035610, 0.0003332, -0.0037632, 0.0005500, -0.0032783, 0.0032804
9: -0.0022502, 0.0022653, -0.0025016, 0.0024998, -0.0038037, 0.0038013

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027798, upper bound: 0.0026732
time: 1.26 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028246, upper bound: 0.0026732
time: 1.29 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9885371, 0.9957764, 0.9887469, 0.9961403, -0.0062578, 0.0057677
1: -0.0041202, -0.0023164, -0.0040679, -0.0022257, -0.0015593, 0.0014372
2: 0.0022216, 0.0117809, 0.0017410, 0.0115038, -0.0076162, 0.0082634
3: -0.0066353, -0.0022843, -0.0065091, -0.0020656, -0.0037611, 0.0034666
4: 0.0009579, 0.0028081, 0.0008649, 0.0027544, -0.0014741, 0.0015994
5: 0.0017536, 0.0137768, 0.0011492, 0.0134282, -0.0095792, 0.0103932
6: -0.0019559, 0.0010957, -0.0018674, 0.0012492, -0.0026379, 0.0024313
7: -0.0081980, -0.0003026, -0.0079691, 0.0000943, -0.0068251, 0.0062905
8: -0.0038754, 0.0002767, -0.0037550, 0.0004855, -0.0035892, 0.0033081
9: -0.0021847, 0.0026299, -0.0024268, 0.0024903, -0.0038359, 0.0041619

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027798, upper bound: 0.0027299
time: 1.22 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028246, upper bound: 0.0027299
time: 1.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9890853, 0.9958748, 0.9880032, 0.9960642, -0.0056074, 0.0065124
1: -0.0039836, -0.0022919, -0.0042532, -0.0022446, -0.0013972, 0.0016227
2: 0.0020916, 0.0110570, 0.0018414, 0.0124860, -0.0085995, 0.0074045
3: -0.0063058, -0.0022251, -0.0069562, -0.0021113, -0.0033702, 0.0039141
4: 0.0009327, 0.0026679, 0.0008843, 0.0029445, -0.0016644, 0.0014331
5: 0.0015902, 0.0128663, 0.0012755, 0.0146635, -0.0108159, 0.0093129
6: -0.0017248, 0.0011372, -0.0021809, 0.0012171, -0.0023637, 0.0027452
7: -0.0076001, -0.0001953, -0.0087803, 0.0000114, -0.0061157, 0.0071027
8: -0.0035610, 0.0003332, -0.0041816, 0.0004418, -0.0032162, 0.0037352
9: -0.0022502, 0.0022653, -0.0023762, 0.0029850, -0.0043312, 0.0037293

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027233, upper bound: 0.0025882
time: 1.63 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027814, upper bound: 0.0025882
time: 1.56 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9885371, 0.9957764, 0.9880158, 0.9959484, -0.0061504, 0.0064960
1: -0.0041202, -0.0023164, -0.0042501, -0.0022735, -0.0015325, 0.0016186
2: 0.0022216, 0.0117809, 0.0019945, 0.0124694, -0.0085779, 0.0081215
3: -0.0066353, -0.0022843, -0.0069487, -0.0021809, -0.0036966, 0.0039043
4: 0.0009579, 0.0028081, 0.0009139, 0.0029413, -0.0016602, 0.0015719
5: 0.0017536, 0.0137768, 0.0014680, 0.0146427, -0.0107888, 0.0102148
6: -0.0019559, 0.0010957, -0.0021756, 0.0011682, -0.0025926, 0.0027383
7: -0.0081980, -0.0003026, -0.0087667, -0.0001150, -0.0067079, 0.0070848
8: -0.0038754, 0.0002767, -0.0041745, 0.0003754, -0.0035276, 0.0037258
9: -0.0021847, 0.0026299, -0.0022991, 0.0029766, -0.0043203, 0.0040904

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027223, upper bound: 0.0026220
time: 1.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027797, upper bound: 0.0026220
time: 1.57 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9884105, 0.9954953, 0.9887640, 0.9959856, -0.0062547, 0.0056627
1: -0.0041517, -0.0023864, -0.0040637, -0.0022642, -0.0015585, 0.0014110
2: 0.0025927, 0.0119481, 0.0019452, 0.0114814, -0.0074775, 0.0082592
3: -0.0067114, -0.0024532, -0.0064990, -0.0021585, -0.0037592, 0.0034034
4: 0.0010297, 0.0028404, 0.0009044, 0.0027501, -0.0014473, 0.0015986
5: 0.0022204, 0.0139870, 0.0014060, 0.0134000, -0.0094048, 0.0103879
6: -0.0020092, 0.0009773, -0.0018602, 0.0011840, -0.0026366, 0.0023870
7: -0.0083361, -0.0006091, -0.0079506, -0.0000743, -0.0068216, 0.0061760
8: -0.0039480, 0.0001155, -0.0037453, 0.0003968, -0.0035874, 0.0032479
9: -0.0019978, 0.0027141, -0.0023239, 0.0024790, -0.0037661, 0.0041598

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0025992
time: 1.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0025992
time: 1.52 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9882876, 0.9956000, 0.9887629, 0.9960266, -0.0063334, 0.0057455
1: -0.0041824, -0.0023603, -0.0040639, -0.0022540, -0.0015781, 0.0014316
2: 0.0024544, 0.0121104, 0.0018911, 0.0114827, -0.0075868, 0.0083632
3: -0.0067852, -0.0023903, -0.0064996, -0.0021339, -0.0038066, 0.0034532
4: 0.0010029, 0.0028718, 0.0008939, 0.0027503, -0.0014684, 0.0016187
5: 0.0020464, 0.0141911, 0.0013380, 0.0134017, -0.0095422, 0.0105187
6: -0.0020610, 0.0010214, -0.0018607, 0.0012012, -0.0026698, 0.0024219
7: -0.0084701, -0.0004949, -0.0079517, -0.0000296, -0.0069075, 0.0062662
8: -0.0040185, 0.0001756, -0.0037459, 0.0004203, -0.0036326, 0.0032954
9: -0.0020675, 0.0027958, -0.0023512, 0.0024797, -0.0038211, 0.0042122

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026533
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026533
time: 1.65 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9884210, 0.9953738, 0.9881956, 0.9959067, -0.0063222, 0.0061554
1: -0.0041491, -0.0024167, -0.0042053, -0.0022839, -0.0015753, 0.0015338
2: 0.0027531, 0.0119343, 0.0020494, 0.0122319, -0.0081281, 0.0083484
3: -0.0067051, -0.0025262, -0.0068406, -0.0022059, -0.0037999, 0.0036996
4: 0.0010607, 0.0028377, 0.0009245, 0.0028954, -0.0015732, 0.0016158
5: 0.0024221, 0.0139696, 0.0015371, 0.0143440, -0.0102230, 0.0105002
6: -0.0020048, 0.0009261, -0.0020998, 0.0011507, -0.0026651, 0.0025947
7: -0.0083247, -0.0007416, -0.0085705, -0.0001604, -0.0068953, 0.0067133
8: -0.0039420, 0.0000459, -0.0040713, 0.0003515, -0.0036262, 0.0035305
9: -0.0019170, 0.0027071, -0.0022714, 0.0028570, -0.0040938, 0.0042047

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027107, upper bound: 0.0025921
time: 1.60 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027107, upper bound: 0.0025921
time: 1.61 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9882984, 0.9954761, 0.9881945, 0.9959368, -0.0064044, 0.0062369
1: -0.0041797, -0.0023912, -0.0042056, -0.0022764, -0.0015958, 0.0015541
2: 0.0026181, 0.0120962, 0.0020098, 0.0122333, -0.0082357, 0.0084569
3: -0.0067788, -0.0024648, -0.0068412, -0.0021879, -0.0038492, 0.0037486
4: 0.0010346, 0.0028691, 0.0009169, 0.0028956, -0.0015940, 0.0016368
5: 0.0022523, 0.0141732, 0.0014872, 0.0143457, -0.0103584, 0.0106366
6: -0.0020565, 0.0009692, -0.0021003, 0.0011634, -0.0026997, 0.0026291
7: -0.0084584, -0.0006301, -0.0085716, -0.0001276, -0.0069849, 0.0068022
8: -0.0040123, 0.0001045, -0.0040719, 0.0003687, -0.0036733, 0.0035772
9: -0.0019850, 0.0027886, -0.0022914, 0.0028577, -0.0041480, 0.0042594

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027107, upper bound: 0.0026448
time: 1.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027107, upper bound: 0.0026448
time: 1.26 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9887550, 0.9960905, 0.9890710, 0.9960322, -0.0058782, 0.0056355
1: -0.0040659, -0.0022381, -0.0039872, -0.0022526, -0.0014647, 0.0014042
2: 0.0018067, 0.0114933, 0.0018837, 0.0110761, -0.0074417, 0.0077621
3: -0.0065044, -0.0020955, -0.0063145, -0.0021305, -0.0035330, 0.0033871
4: 0.0008776, 0.0027524, 0.0008925, 0.0026716, -0.0014403, 0.0015023
5: 0.0012319, 0.0134150, 0.0013287, 0.0128902, -0.0093597, 0.0097627
6: -0.0018640, 0.0012282, -0.0017308, 0.0012036, -0.0024779, 0.0023756
7: -0.0079605, 0.0000400, -0.0076158, -0.0000236, -0.0064110, 0.0061463
8: -0.0037505, 0.0004569, -0.0035692, 0.0004235, -0.0033715, 0.0032323
9: -0.0023937, 0.0024850, -0.0023549, 0.0022749, -0.0037480, 0.0039094

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026848, upper bound: 0.0027509
time: 1.60 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027299, upper bound: 0.0027509
time: 1.66 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9881873, 0.9960075, 0.9890821, 0.9959106, -0.0064337, 0.0056161
1: -0.0042074, -0.0022588, -0.0039844, -0.0022829, -0.0016031, 0.0013994
2: 0.0019164, 0.0122428, 0.0020444, 0.0110612, -0.0074160, 0.0084956
3: -0.0068455, -0.0021454, -0.0063077, -0.0022036, -0.0038668, 0.0033754
4: 0.0008988, 0.0028975, 0.0009236, 0.0026688, -0.0014353, 0.0016443
5: 0.0013697, 0.0143577, 0.0015307, 0.0128716, -0.0093273, 0.0106852
6: -0.0021033, 0.0011932, -0.0017261, 0.0011523, -0.0027120, 0.0023674
7: -0.0085795, -0.0000505, -0.0076036, -0.0001562, -0.0070168, 0.0061251
8: -0.0040760, 0.0004093, -0.0035628, 0.0003537, -0.0036901, 0.0032211
9: -0.0023384, 0.0028625, -0.0022740, 0.0022674, -0.0037351, 0.0042788

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026848, upper bound: 0.0028246
time: 1.44 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027299, upper bound: 0.0028246
time: 1.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9887550, 0.9960905, 0.9883967, 0.9957643, -0.0057847, 0.0063560
1: -0.0040659, -0.0022381, -0.0041552, -0.0023194, -0.0014414, 0.0015838
2: 0.0018067, 0.0114933, 0.0022375, 0.0119662, -0.0083931, 0.0076386
3: -0.0065044, -0.0020955, -0.0067196, -0.0022915, -0.0034768, 0.0038202
4: 0.0008776, 0.0027524, 0.0009610, 0.0028439, -0.0016245, 0.0014784
5: 0.0012319, 0.0134150, 0.0017736, 0.0140098, -0.0105563, 0.0096074
6: -0.0018640, 0.0012282, -0.0020150, 0.0010907, -0.0024385, 0.0026793
7: -0.0079605, 0.0000400, -0.0083511, -0.0003157, -0.0063090, 0.0069322
8: -0.0037505, 0.0004569, -0.0039559, 0.0002698, -0.0033179, 0.0036456
9: -0.0023937, 0.0024850, -0.0021767, 0.0027232, -0.0042272, 0.0038472

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026032, upper bound: 0.0026556
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026557, upper bound: 0.0026556
time: 1.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9881873, 0.9960075, 0.9884073, 0.9956402, -0.0063428, 0.0064244
1: -0.0042074, -0.0022588, -0.0041526, -0.0023503, -0.0015805, 0.0016008
2: 0.0019164, 0.0122428, 0.0024014, 0.0119524, -0.0084834, 0.0083757
3: -0.0068455, -0.0021454, -0.0067133, -0.0023662, -0.0038122, 0.0038613
4: 0.0008988, 0.0028975, 0.0009927, 0.0028412, -0.0016419, 0.0016211
5: 0.0013697, 0.0143577, 0.0019798, 0.0139924, -0.0106698, 0.0105344
6: -0.0021033, 0.0011932, -0.0020106, 0.0010383, -0.0026737, 0.0027081
7: -0.0085795, -0.0000505, -0.0083396, -0.0004512, -0.0069178, 0.0070067
8: -0.0040760, 0.0004093, -0.0039499, 0.0001986, -0.0036380, 0.0036848
9: -0.0023384, 0.0028625, -0.0020941, 0.0027162, -0.0042727, 0.0042184

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026015, upper bound: 0.0027107
time: 1.30 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026544, upper bound: 0.0027107
time: 1.68 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9880171, 0.9958012, 0.9890929, 0.9957256, -0.0064116, 0.0055379
1: -0.0042498, -0.0023102, -0.0039817, -0.0023290, -0.0015976, 0.0013799
2: 0.0021888, 0.0124676, 0.0022886, 0.0110469, -0.0073128, 0.0084665
3: -0.0069478, -0.0022694, -0.0063012, -0.0023148, -0.0038536, 0.0033285
4: 0.0009515, 0.0029410, 0.0009708, 0.0026660, -0.0014154, 0.0016387
5: 0.0017124, 0.0146404, 0.0018379, 0.0128536, -0.0091976, 0.0106486
6: -0.0021751, 0.0011062, -0.0017215, 0.0010743, -0.0027027, 0.0023344
7: -0.0087652, -0.0002755, -0.0075918, -0.0003580, -0.0069928, 0.0060399
8: -0.0041737, 0.0002910, -0.0035566, 0.0002476, -0.0036774, 0.0031763
9: -0.0022012, 0.0029757, -0.0021510, 0.0022602, -0.0036831, 0.0042642

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027193
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027193
time: 1.27 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9879042, 0.9959075, 0.9890923, 0.9957647, -0.0064980, 0.0056194
1: -0.0042779, -0.0022837, -0.0039819, -0.0023193, -0.0016191, 0.0014002
2: 0.0020484, 0.0126167, 0.0022370, 0.0110478, -0.0074204, 0.0085805
3: -0.0070157, -0.0022054, -0.0063016, -0.0022913, -0.0039055, 0.0033774
4: 0.0009243, 0.0029698, 0.0009609, 0.0026662, -0.0014362, 0.0016607
5: 0.0015358, 0.0148279, 0.0017730, 0.0128547, -0.0093329, 0.0107921
6: -0.0022227, 0.0011510, -0.0017218, 0.0010908, -0.0027391, 0.0023688
7: -0.0088883, -0.0001595, -0.0075925, -0.0003153, -0.0070870, 0.0061288
8: -0.0042384, 0.0003520, -0.0035570, 0.0002700, -0.0037270, 0.0032231
9: -0.0022720, 0.0030508, -0.0021770, 0.0022607, -0.0037373, 0.0043216

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027787
time: 1.28 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027787
time: 1.26 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9880295, 0.9956904, 0.9885445, 0.9956250, -0.0063943, 0.0060783
1: -0.0042467, -0.0023378, -0.0041183, -0.0023541, -0.0015933, 0.0015146
2: 0.0023350, 0.0124512, 0.0024214, 0.0117711, -0.0080264, 0.0084435
3: -0.0069404, -0.0023359, -0.0066308, -0.0023752, -0.0038431, 0.0036533
4: 0.0009798, 0.0029378, 0.0009965, 0.0028062, -0.0015535, 0.0016342
5: 0.0018963, 0.0146197, 0.0020049, 0.0137644, -0.0100951, 0.0106198
6: -0.0021698, 0.0010595, -0.0019527, 0.0010320, -0.0026954, 0.0025622
7: -0.0087516, -0.0003963, -0.0081899, -0.0004676, -0.0069738, 0.0066293
8: -0.0041665, 0.0002274, -0.0038711, 0.0001899, -0.0036675, 0.0034863
9: -0.0021276, 0.0029674, -0.0020841, 0.0026249, -0.0040425, 0.0042526

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027090
time: 1.42 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027090
time: 1.55 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9879171, 0.9957882, 0.9885438, 0.9956619, -0.0064890, 0.0061593
1: -0.0042747, -0.0023134, -0.0041185, -0.0023449, -0.0016169, 0.0015347
2: 0.0022059, 0.0125997, 0.0023726, 0.0117720, -0.0081332, 0.0085687
3: -0.0070080, -0.0022771, -0.0066312, -0.0023530, -0.0039001, 0.0037019
4: 0.0009548, 0.0029665, 0.0009871, 0.0028063, -0.0015742, 0.0016584
5: 0.0017339, 0.0148065, 0.0019436, 0.0137656, -0.0102295, 0.0107771
6: -0.0022172, 0.0011008, -0.0019530, 0.0010475, -0.0027353, 0.0025964
7: -0.0088743, -0.0002896, -0.0081907, -0.0004274, -0.0070772, 0.0067176
8: -0.0042310, 0.0002835, -0.0038715, 0.0002111, -0.0037218, 0.0035327
9: -0.0021926, 0.0030422, -0.0021086, 0.0026254, -0.0040963, 0.0043156

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027665
time: 1.29 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027666
time: 1.30 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9887550, 0.9960905, 0.9887328, 0.9962528, -0.0056175, 0.0054620
1: -0.0040659, -0.0022381, -0.0040715, -0.0021977, -0.0013997, 0.0013610
2: 0.0018067, 0.0114933, 0.0015925, 0.0115226, -0.0072125, 0.0074178
3: -0.0065044, -0.0020955, -0.0065177, -0.0019979, -0.0033763, 0.0032828
4: 0.0008776, 0.0027524, 0.0008361, 0.0027581, -0.0013960, 0.0014357
5: 0.0012319, 0.0134150, 0.0009624, 0.0134518, -0.0090714, 0.0093297
6: -0.0018640, 0.0012282, -0.0018734, 0.0012966, -0.0023680, 0.0023024
7: -0.0079605, 0.0000400, -0.0079846, 0.0002170, -0.0061267, 0.0059571
8: -0.0037505, 0.0004569, -0.0037632, 0.0005500, -0.0032220, 0.0031328
9: -0.0023937, 0.0024850, -0.0025016, 0.0024998, -0.0036326, 0.0037360

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026874, upper bound: 0.0027515
time: 1.82 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027325, upper bound: 0.0027515
time: 1.34 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9881873, 0.9960075, 0.9887469, 0.9961403, -0.0061567, 0.0055217
1: -0.0042074, -0.0022588, -0.0040679, -0.0022257, -0.0015341, 0.0013759
2: 0.0019164, 0.0122428, 0.0017410, 0.0115038, -0.0072913, 0.0081299
3: -0.0068455, -0.0021454, -0.0065091, -0.0020656, -0.0037004, 0.0033187
4: 0.0008988, 0.0028975, 0.0008649, 0.0027544, -0.0014112, 0.0015735
5: 0.0013697, 0.0143577, 0.0011492, 0.0134282, -0.0091705, 0.0102252
6: -0.0021033, 0.0011932, -0.0018674, 0.0012492, -0.0025953, 0.0023276
7: -0.0085795, -0.0000505, -0.0079691, 0.0000943, -0.0067148, 0.0060222
8: -0.0040760, 0.0004093, -0.0037550, 0.0004855, -0.0035312, 0.0031670
9: -0.0023384, 0.0028625, -0.0024268, 0.0024903, -0.0036723, 0.0040946

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026874, upper bound: 0.0028254
time: 1.35 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027325, upper bound: 0.0028253
time: 1.65 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9887550, 0.9960905, 0.9880032, 0.9960642, -0.0055121, 0.0062748
1: -0.0040659, -0.0022381, -0.0042532, -0.0022446, -0.0013735, 0.0015635
2: 0.0018067, 0.0114933, 0.0018414, 0.0124860, -0.0082859, 0.0072786
3: -0.0065044, -0.0020955, -0.0069562, -0.0021113, -0.0033129, 0.0037714
4: 0.0008776, 0.0027524, 0.0008843, 0.0029445, -0.0016037, 0.0014088
5: 0.0012319, 0.0134150, 0.0012755, 0.0146635, -0.0104214, 0.0091546
6: -0.0018640, 0.0012282, -0.0021809, 0.0012171, -0.0023235, 0.0026451
7: -0.0079605, 0.0000400, -0.0087803, 0.0000114, -0.0060117, 0.0068436
8: -0.0037505, 0.0004569, -0.0041816, 0.0004418, -0.0031615, 0.0035990
9: -0.0023937, 0.0024850, -0.0023762, 0.0029850, -0.0041732, 0.0036659

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026231, upper bound: 0.0026607
time: 1.62 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026740, upper bound: 0.0026607
time: 1.22 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9881873, 0.9960075, 0.9880158, 0.9959484, -0.0060555, 0.0063111
1: -0.0042074, -0.0022588, -0.0042501, -0.0022735, -0.0015089, 0.0015725
2: 0.0019164, 0.0122428, 0.0019945, 0.0124694, -0.0083337, 0.0079963
3: -0.0068455, -0.0021454, -0.0069487, -0.0021809, -0.0036396, 0.0037931
4: 0.0008988, 0.0028975, 0.0009139, 0.0029413, -0.0016130, 0.0015477
5: 0.0013697, 0.0143577, 0.0014680, 0.0146427, -0.0104816, 0.0100572
6: -0.0021033, 0.0011932, -0.0021756, 0.0011682, -0.0025526, 0.0026603
7: -0.0085795, -0.0000505, -0.0087667, -0.0001150, -0.0066044, 0.0068831
8: -0.0040760, 0.0004093, -0.0041745, 0.0003754, -0.0034732, 0.0036198
9: -0.0023384, 0.0028625, -0.0022991, 0.0029766, -0.0041973, 0.0040273

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026224, upper bound: 0.0027174
time: 1.30 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026736, upper bound: 0.0027174
time: 1.41 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9880171, 0.9958012, 0.9887640, 0.9959856, -0.0061678, 0.0054219
1: -0.0042498, -0.0023102, -0.0040637, -0.0022642, -0.0015368, 0.0013510
2: 0.0021888, 0.0124676, 0.0019452, 0.0114814, -0.0071595, 0.0081445
3: -0.0069478, -0.0022694, -0.0064990, -0.0021585, -0.0037070, 0.0032587
4: 0.0009515, 0.0029410, 0.0009044, 0.0027501, -0.0013857, 0.0015763
5: 0.0017124, 0.0146404, 0.0014060, 0.0134000, -0.0090048, 0.0102436
6: -0.0021751, 0.0011062, -0.0018602, 0.0011840, -0.0025999, 0.0022855
7: -0.0087652, -0.0002755, -0.0079506, -0.0000743, -0.0067268, 0.0059133
8: -0.0041737, 0.0002910, -0.0037453, 0.0003968, -0.0035376, 0.0031097
9: -0.0022012, 0.0029757, -0.0023239, 0.0024790, -0.0036059, 0.0041020

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027263
time: 1.32 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027263
time: 1.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9879042, 0.9959075, 0.9887629, 0.9960266, -0.0062439, 0.0055001
1: -0.0042779, -0.0022837, -0.0040639, -0.0022540, -0.0015558, 0.0013705
2: 0.0020484, 0.0126167, 0.0018911, 0.0114827, -0.0072629, 0.0082450
3: -0.0070157, -0.0022054, -0.0064996, -0.0021339, -0.0037528, 0.0033057
4: 0.0009243, 0.0029698, 0.0008939, 0.0027503, -0.0014057, 0.0015958
5: 0.0015358, 0.0148279, 0.0013380, 0.0134017, -0.0091348, 0.0103700
6: -0.0022227, 0.0011510, -0.0018607, 0.0012012, -0.0026320, 0.0023185
7: -0.0088883, -0.0001595, -0.0079517, -0.0000296, -0.0068098, 0.0059987
8: -0.0042384, 0.0003520, -0.0037459, 0.0004203, -0.0035812, 0.0031546
9: -0.0022720, 0.0030508, -0.0023512, 0.0024797, -0.0036580, 0.0041526

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027867
time: 1.61 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027867
time: 1.20 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9880295, 0.9956904, 0.9881956, 0.9959067, -0.0061991, 0.0059258
1: -0.0042467, -0.0023378, -0.0042053, -0.0022839, -0.0015447, 0.0014766
2: 0.0023350, 0.0124512, 0.0020494, 0.0122319, -0.0078250, 0.0081859
3: -0.0069404, -0.0023359, -0.0068406, -0.0022059, -0.0037259, 0.0035616
4: 0.0009798, 0.0029378, 0.0009245, 0.0028954, -0.0015145, 0.0015844
5: 0.0018963, 0.0146197, 0.0015371, 0.0143440, -0.0098417, 0.0102957
6: -0.0021698, 0.0010595, -0.0020998, 0.0011507, -0.0026132, 0.0024979
7: -0.0087516, -0.0003963, -0.0085705, -0.0001604, -0.0067610, 0.0064629
8: -0.0041665, 0.0002274, -0.0040713, 0.0003515, -0.0035556, 0.0033988
9: -0.0021276, 0.0029674, -0.0022714, 0.0028570, -0.0039411, 0.0041229

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026404, upper bound: 0.0027196
time: 1.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026404, upper bound: 0.0027196
time: 1.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9879171, 0.9957882, 0.9881945, 0.9959368, -0.0062776, 0.0060032
1: -0.0042747, -0.0023134, -0.0042056, -0.0022764, -0.0015642, 0.0014958
2: 0.0022059, 0.0125997, 0.0020098, 0.0122333, -0.0079272, 0.0082895
3: -0.0070080, -0.0022771, -0.0068412, -0.0021879, -0.0037730, 0.0036081
4: 0.0009548, 0.0029665, 0.0009169, 0.0028956, -0.0015343, 0.0016044
5: 0.0017339, 0.0148065, 0.0014872, 0.0143457, -0.0099703, 0.0104261
6: -0.0022172, 0.0011008, -0.0021003, 0.0011634, -0.0026462, 0.0025306
7: -0.0088743, -0.0002896, -0.0085716, -0.0001276, -0.0068466, 0.0065474
8: -0.0042310, 0.0002835, -0.0040719, 0.0003687, -0.0036006, 0.0034432
9: -0.0021926, 0.0030422, -0.0022914, 0.0028577, -0.0039926, 0.0041751

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026405, upper bound: 0.0027787
time: 1.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026405, upper bound: 0.0027787
time: 1.72 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.95 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026862, upper bound: 0.0026732
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027329, upper bound: 0.0026732
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026862, upper bound: 0.0027329
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027329, upper bound: 0.0027329
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026032, upper bound: 0.0025882
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026564, upper bound: 0.0025882
NS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026015, upper bound: 0.0026220
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026547, upper bound: 0.0026220
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026536
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026536
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0025921
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0025921
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026448
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026448
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027798, upper bound: 0.0026732
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0028246, upper bound: 0.0026732
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027798, upper bound: 0.0027299
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0028246, upper bound: 0.0027299
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027233, upper bound: 0.0025882
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027814, upper bound: 0.0025882
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027223, upper bound: 0.0026220
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027797, upper bound: 0.0026220
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0025992
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0025992
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026533
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026533
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027107, upper bound: 0.0025921
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027107, upper bound: 0.0025921
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027107, upper bound: 0.0026448
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027107, upper bound: 0.0026448
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026848, upper bound: 0.0027509
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027299, upper bound: 0.0027509
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026848, upper bound: 0.0028246
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027299, upper bound: 0.0028246
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026032, upper bound: 0.0026556
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026557, upper bound: 0.0026556
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026015, upper bound: 0.0027107
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026544, upper bound: 0.0027107
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027193
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027193
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027787
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027787
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027090
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027090
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027665
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027666
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026874, upper bound: 0.0027515
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027325, upper bound: 0.0027515
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026874, upper bound: 0.0028254
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0027325, upper bound: 0.0028253
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026231, upper bound: 0.0026607
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026740, upper bound: 0.0026607
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026224, upper bound: 0.0027174
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026736, upper bound: 0.0027174
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027263
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027263
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027867
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027867
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026404, upper bound: 0.0027196
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026404, upper bound: 0.0027196
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026405, upper bound: 0.0027787
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 0, lower bound: -0.0026405, upper bound: 0.0027787

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9890911, 0.9957576, 0.9890842, 0.9957697, -0.0051057, 0.0051283
1: -0.0039822, -0.0023210, -0.0039838, -0.0023180, -0.0012722, 0.0012778
2: 0.0022463, 0.0110494, 0.0022304, 0.0110583, -0.0067719, 0.0067420
3: -0.0063023, -0.0022956, -0.0063064, -0.0022883, -0.0030687, 0.0030823
4: 0.0009627, 0.0026665, 0.0009596, 0.0026682, -0.0013107, 0.0013049
5: 0.0017847, 0.0128567, 0.0017647, 0.0128679, -0.0085172, 0.0084796
6: -0.0017223, 0.0010878, -0.0017252, 0.0010929, -0.0021522, 0.0021618
7: -0.0075939, -0.0003230, -0.0076012, -0.0003099, -0.0055685, 0.0055931
8: -0.0035577, 0.0002660, -0.0035615, 0.0002729, -0.0029284, 0.0029414
9: -0.0021723, 0.0022615, -0.0021803, 0.0022659, -0.0034107, 0.0033956

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025010, upper bound: 0.0023898
time: 1.61 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026033, upper bound: 0.0025940
time: 1.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9890904, 0.9957986, 0.9889615, 0.9958715, -0.0051806, 0.0051930
1: -0.0039823, -0.0023109, -0.0040144, -0.0022927, -0.0012909, 0.0012940
2: 0.0021923, 0.0110503, 0.0020960, 0.0112204, -0.0068573, 0.0068409
3: -0.0063027, -0.0022710, -0.0063802, -0.0022271, -0.0031137, 0.0031211
4: 0.0009522, 0.0026666, 0.0009336, 0.0026996, -0.0013272, 0.0013240
5: 0.0017168, 0.0128578, 0.0015956, 0.0130718, -0.0086246, 0.0086041
6: -0.0017226, 0.0011051, -0.0017769, 0.0011358, -0.0021838, 0.0021890
7: -0.0075945, -0.0002784, -0.0077351, -0.0001989, -0.0056502, 0.0056637
8: -0.0035580, 0.0002894, -0.0036320, 0.0003313, -0.0029714, 0.0029785
9: -0.0021995, 0.0022619, -0.0022480, 0.0023476, -0.0034537, 0.0034455

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025439, upper bound: 0.0023898
time: 1.52 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026507, upper bound: 0.0025940
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9885427, 0.9956633, 0.9890956, 0.9956468, -0.0056422, 0.0051426
1: -0.0041188, -0.0023445, -0.0039810, -0.0023486, -0.0014059, 0.0012814
2: 0.0023709, 0.0117737, 0.0023926, 0.0110434, -0.0067908, 0.0074505
3: -0.0066320, -0.0023522, -0.0062996, -0.0023621, -0.0033911, 0.0030909
4: 0.0009868, 0.0028067, 0.0009910, 0.0026653, -0.0013143, 0.0014420
5: 0.0019414, 0.0137676, 0.0019687, 0.0128491, -0.0085410, 0.0093707
6: -0.0019535, 0.0010481, -0.0017204, 0.0010412, -0.0023784, 0.0021678
7: -0.0081920, -0.0004259, -0.0075889, -0.0004438, -0.0061536, 0.0056088
8: -0.0038723, 0.0002119, -0.0035550, 0.0002024, -0.0032361, 0.0029496
9: -0.0021095, 0.0026262, -0.0020986, 0.0022584, -0.0034202, 0.0037525

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0024950, upper bound: 0.0023765
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026033, upper bound: 0.0026507
time: 1.64 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9885420, 0.9956973, 0.9889740, 0.9957497, -0.0057161, 0.0052200
1: -0.0041190, -0.0023361, -0.0040114, -0.0023230, -0.0014243, 0.0013007
2: 0.0023260, 0.0117746, 0.0022567, 0.0112041, -0.0068930, 0.0075480
3: -0.0066324, -0.0023318, -0.0063727, -0.0023003, -0.0034355, 0.0031374
4: 0.0009781, 0.0028068, 0.0009647, 0.0026964, -0.0013341, 0.0014609
5: 0.0018850, 0.0137687, 0.0017978, 0.0130512, -0.0086696, 0.0094934
6: -0.0019538, 0.0010624, -0.0017717, 0.0010845, -0.0024095, 0.0022004
7: -0.0081928, -0.0003889, -0.0077216, -0.0003316, -0.0062342, 0.0056932
8: -0.0038726, 0.0002314, -0.0036248, 0.0002615, -0.0032785, 0.0029940
9: -0.0021321, 0.0026267, -0.0021670, 0.0023393, -0.0034717, 0.0038016

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025343, upper bound: 0.0023763
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026507, upper bound: 0.0026507
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9890904, 0.9957986, 0.9882876, 0.9956000, -0.0050811, 0.0059473
1: -0.0039823, -0.0023109, -0.0041824, -0.0023603, -0.0012661, 0.0014819
2: 0.0021923, 0.0110503, 0.0024544, 0.0121104, -0.0078533, 0.0067096
3: -0.0063027, -0.0022710, -0.0067852, -0.0023903, -0.0030539, 0.0035745
4: 0.0009522, 0.0026666, 0.0010029, 0.0028718, -0.0015200, 0.0012986
5: 0.0017168, 0.0128578, 0.0020464, 0.0141911, -0.0098774, 0.0084389
6: -0.0017226, 0.0011051, -0.0020610, 0.0010214, -0.0021419, 0.0025070
7: -0.0075945, -0.0002784, -0.0084701, -0.0004949, -0.0055417, 0.0064863
8: -0.0035580, 0.0002894, -0.0040185, 0.0001756, -0.0029143, 0.0034111
9: -0.0021995, 0.0022619, -0.0020675, 0.0027958, -0.0039553, 0.0033793

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026564, upper bound: 0.0025389
time: 1.55 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026564, upper bound: 0.0025882
time: 1.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9885420, 0.9956973, 0.9882984, 0.9954761, -0.0056207, 0.0060342
1: -0.0041190, -0.0023361, -0.0041797, -0.0023912, -0.0014005, 0.0015036
2: 0.0023260, 0.0117746, 0.0026181, 0.0120962, -0.0079681, 0.0074220
3: -0.0066324, -0.0023318, -0.0067788, -0.0024648, -0.0033782, 0.0036267
4: 0.0009781, 0.0028068, 0.0010346, 0.0028691, -0.0015422, 0.0014365
5: 0.0018850, 0.0137687, 0.0022523, 0.0141732, -0.0100218, 0.0093349
6: -0.0019538, 0.0010624, -0.0020565, 0.0009692, -0.0023693, 0.0025436
7: -0.0081928, -0.0003889, -0.0084584, -0.0006301, -0.0061301, 0.0065811
8: -0.0038726, 0.0002314, -0.0040123, 0.0001045, -0.0032238, 0.0034610
9: -0.0021321, 0.0026267, -0.0019850, 0.0027886, -0.0040132, 0.0037381

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026547, upper bound: 0.0025747
time: 1.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026547, upper bound: 0.0026220
time: 1.26 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9882876, 0.9956000, 0.9890904, 0.9957986, -0.0059473, 0.0050811
1: -0.0041824, -0.0023603, -0.0039823, -0.0023109, -0.0014819, 0.0012661
2: 0.0024544, 0.0121104, 0.0021923, 0.0110503, -0.0067096, 0.0078533
3: -0.0067852, -0.0023903, -0.0063027, -0.0022710, -0.0035745, 0.0030539
4: 0.0010029, 0.0028718, 0.0009522, 0.0026666, -0.0012986, 0.0015200
5: 0.0020464, 0.0141911, 0.0017168, 0.0128578, -0.0084389, 0.0098774
6: -0.0020610, 0.0010214, -0.0017226, 0.0011051, -0.0025070, 0.0021419
7: -0.0084701, -0.0004949, -0.0075945, -0.0002784, -0.0064863, 0.0055417
8: -0.0040185, 0.0001756, -0.0035580, 0.0002894, -0.0034111, 0.0029143
9: -0.0020675, 0.0027958, -0.0021995, 0.0022619, -0.0033793, 0.0039553

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026506
time: 1.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026536
time: 1.31 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9882876, 0.9956000, 0.9884154, 0.9955269, -0.0052398, 0.0053031
1: -0.0041824, -0.0023603, -0.0041506, -0.0023786, -0.0013056, 0.0013214
2: 0.0024544, 0.0121104, 0.0025511, 0.0119418, -0.0070026, 0.0069191
3: -0.0067852, -0.0023903, -0.0067085, -0.0024343, -0.0031493, 0.0031873
4: 0.0010029, 0.0028718, 0.0010216, 0.0028392, -0.0013553, 0.0013392
5: 0.0020464, 0.0141911, 0.0021680, 0.0139791, -0.0088075, 0.0087024
6: -0.0020610, 0.0010214, -0.0020072, 0.0009906, -0.0022088, 0.0022354
7: -0.0084701, -0.0004949, -0.0083309, -0.0005747, -0.0057147, 0.0057837
8: -0.0040185, 0.0001756, -0.0039453, 0.0001336, -0.0030053, 0.0030416
9: -0.0020675, 0.0027958, -0.0020188, 0.0027109, -0.0035269, 0.0034848

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026506
time: 1.73 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026536
time: 1.74 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9882984, 0.9954761, 0.9885420, 0.9956973, -0.0060342, 0.0056207
1: -0.0041797, -0.0023912, -0.0041190, -0.0023361, -0.0015036, 0.0014005
2: 0.0026181, 0.0120962, 0.0023260, 0.0117746, -0.0074220, 0.0079681
3: -0.0067788, -0.0024648, -0.0066324, -0.0023318, -0.0036267, 0.0033782
4: 0.0010346, 0.0028691, 0.0009781, 0.0028068, -0.0014365, 0.0015422
5: 0.0022523, 0.0141732, 0.0018850, 0.0137687, -0.0093349, 0.0100218
6: -0.0020565, 0.0009692, -0.0019538, 0.0010624, -0.0025436, 0.0023693
7: -0.0084584, -0.0006301, -0.0081928, -0.0003889, -0.0065811, 0.0061301
8: -0.0040123, 0.0001045, -0.0038726, 0.0002314, -0.0034610, 0.0032238
9: -0.0019850, 0.0027886, -0.0021321, 0.0026267, -0.0037381, 0.0040132

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026251
time: 1.86 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026448
time: 1.29 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9882984, 0.9954761, 0.9878692, 0.9954093, -0.0053105, 0.0058392
1: -0.0041797, -0.0023912, -0.0042866, -0.0024078, -0.0013232, 0.0014550
2: 0.0026181, 0.0120962, 0.0027061, 0.0126629, -0.0077107, 0.0070124
3: -0.0067788, -0.0024648, -0.0070367, -0.0025048, -0.0031917, 0.0035096
4: 0.0010346, 0.0028691, 0.0010517, 0.0029788, -0.0014924, 0.0013572
5: 0.0022523, 0.0141732, 0.0023630, 0.0148861, -0.0096980, 0.0088198
6: -0.0020565, 0.0009692, -0.0022374, 0.0009411, -0.0022386, 0.0024614
7: -0.0084584, -0.0006301, -0.0089265, -0.0007028, -0.0057918, 0.0063685
8: -0.0040123, 0.0001045, -0.0042585, 0.0000663, -0.0030459, 0.0033491
9: -0.0019850, 0.0027886, -0.0019407, 0.0030741, -0.0038835, 0.0035318

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026251
time: 1.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026448
time: 1.78 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9890911, 0.9957576, 0.9887465, 0.9959960, -0.0054754, 0.0055836
1: -0.0039822, -0.0023210, -0.0040680, -0.0022616, -0.0013643, 0.0013913
2: 0.0022463, 0.0110494, 0.0019315, 0.0115044, -0.0073731, 0.0072302
3: -0.0063023, -0.0022956, -0.0065094, -0.0021523, -0.0032909, 0.0033559
4: 0.0009627, 0.0026665, 0.0009017, 0.0027545, -0.0014270, 0.0013994
5: 0.0017847, 0.0128567, 0.0013888, 0.0134290, -0.0092734, 0.0090936
6: -0.0017223, 0.0010878, -0.0018676, 0.0011883, -0.0023081, 0.0023537
7: -0.0075939, -0.0003230, -0.0079696, -0.0000630, -0.0059717, 0.0060897
8: -0.0035577, 0.0002660, -0.0037553, 0.0004027, -0.0031404, 0.0032025
9: -0.0021723, 0.0022615, -0.0023308, 0.0024906, -0.0037135, 0.0036415

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025955, upper bound: 0.0023898
time: 1.61 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026976, upper bound: 0.0025940
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9890904, 0.9957986, 0.9886309, 0.9960932, -0.0055495, 0.0056597
1: -0.0039823, -0.0023109, -0.0040968, -0.0022374, -0.0013828, 0.0014102
2: 0.0021923, 0.0110503, 0.0018032, 0.0116571, -0.0074735, 0.0073280
3: -0.0063027, -0.0022710, -0.0065789, -0.0020939, -0.0033354, 0.0034016
4: 0.0009522, 0.0026666, 0.0008769, 0.0027841, -0.0014465, 0.0014183
5: 0.0017168, 0.0128578, 0.0012274, 0.0136211, -0.0093997, 0.0092167
6: -0.0017226, 0.0011051, -0.0019163, 0.0012293, -0.0023393, 0.0023857
7: -0.0075945, -0.0002784, -0.0080958, 0.0000430, -0.0060525, 0.0061727
8: -0.0035580, 0.0002894, -0.0038216, 0.0004584, -0.0031829, 0.0032461
9: -0.0021995, 0.0022619, -0.0023954, 0.0025675, -0.0037641, 0.0036908

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026445, upper bound: 0.0023898
time: 1.59 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027460, upper bound: 0.0025940
time: 1.31 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9885427, 0.9956633, 0.9887606, 0.9958824, -0.0060156, 0.0056309
1: -0.0041188, -0.0023445, -0.0040645, -0.0022900, -0.0014989, 0.0014031
2: 0.0023709, 0.0117737, 0.0020816, 0.0114857, -0.0074355, 0.0079436
3: -0.0066320, -0.0023522, -0.0065009, -0.0022206, -0.0036156, 0.0033843
4: 0.0009868, 0.0028067, 0.0009308, 0.0027509, -0.0014391, 0.0015375
5: 0.0019414, 0.0137676, 0.0015775, 0.0134055, -0.0093519, 0.0099909
6: -0.0019535, 0.0010481, -0.0018616, 0.0011404, -0.0025358, 0.0023736
7: -0.0081920, -0.0004259, -0.0079542, -0.0001870, -0.0065609, 0.0061412
8: -0.0038723, 0.0002119, -0.0037472, 0.0003375, -0.0034503, 0.0032296
9: -0.0021095, 0.0026262, -0.0022552, 0.0024812, -0.0037449, 0.0040008

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025892, upper bound: 0.0023765
time: 1.32 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026976, upper bound: 0.0026478
time: 1.76 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9885420, 0.9956973, 0.9886454, 0.9959788, -0.0060894, 0.0057105
1: -0.0041190, -0.0023361, -0.0040932, -0.0022659, -0.0015173, 0.0014229
2: 0.0023260, 0.0117746, 0.0019543, 0.0116380, -0.0075406, 0.0080410
3: -0.0066324, -0.0023318, -0.0065702, -0.0021626, -0.0036599, 0.0034322
4: 0.0009781, 0.0028068, 0.0009061, 0.0027804, -0.0014595, 0.0015563
5: 0.0018850, 0.0137687, 0.0014174, 0.0135970, -0.0094841, 0.0101134
6: -0.0019538, 0.0010624, -0.0019102, 0.0011811, -0.0025669, 0.0024072
7: -0.0081928, -0.0003889, -0.0080800, -0.0000818, -0.0066413, 0.0062281
8: -0.0038726, 0.0002314, -0.0038133, 0.0003928, -0.0034926, 0.0032753
9: -0.0021321, 0.0026267, -0.0023194, 0.0025579, -0.0037978, 0.0040499

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026361, upper bound: 0.0023764
time: 1.29 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027460, upper bound: 0.0026478
time: 1.47 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9890911, 0.9957576, 0.9880171, 0.9958012, -0.0053591, 0.0063761
1: -0.0039822, -0.0023210, -0.0042498, -0.0023102, -0.0013353, 0.0015888
2: 0.0022463, 0.0110494, 0.0021888, 0.0124676, -0.0084196, 0.0070766
3: -0.0063023, -0.0022956, -0.0069478, -0.0022694, -0.0032210, 0.0038322
4: 0.0009627, 0.0026665, 0.0009515, 0.0029410, -0.0016296, 0.0013697
5: 0.0017847, 0.0128567, 0.0017124, 0.0146404, -0.0105896, 0.0089005
6: -0.0017223, 0.0010878, -0.0021751, 0.0011062, -0.0022591, 0.0026878
7: -0.0075939, -0.0003230, -0.0087652, -0.0002755, -0.0058449, 0.0069540
8: -0.0035577, 0.0002660, -0.0041737, 0.0002910, -0.0030738, 0.0036571
9: -0.0021723, 0.0022615, -0.0022012, 0.0029757, -0.0042405, 0.0035642

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027233, upper bound: 0.0025389
time: 1.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027233, upper bound: 0.0025882
time: 1.55 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9890904, 0.9957986, 0.9879042, 0.9959075, -0.0054351, 0.0064622
1: -0.0039823, -0.0023109, -0.0042779, -0.0022837, -0.0013543, 0.0016102
2: 0.0021923, 0.0110503, 0.0020484, 0.0126167, -0.0085332, 0.0071770
3: -0.0063027, -0.0022710, -0.0070157, -0.0022054, -0.0032667, 0.0038840
4: 0.0009522, 0.0026666, 0.0009243, 0.0029698, -0.0016516, 0.0013891
5: 0.0017168, 0.0128578, 0.0015358, 0.0148279, -0.0107326, 0.0090268
6: -0.0017226, 0.0011051, -0.0022227, 0.0011510, -0.0022911, 0.0027240
7: -0.0075945, -0.0002784, -0.0088883, -0.0001595, -0.0059278, 0.0070479
8: -0.0035580, 0.0002894, -0.0042384, 0.0003520, -0.0031174, 0.0037064
9: -0.0021995, 0.0022619, -0.0022720, 0.0030508, -0.0042978, 0.0036147

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027814, upper bound: 0.0025389
time: 1.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027814, upper bound: 0.0025882
time: 1.59 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9885427, 0.9956633, 0.9880295, 0.9956904, -0.0059001, 0.0063648
1: -0.0041188, -0.0023445, -0.0042467, -0.0023378, -0.0014702, 0.0015859
2: 0.0023709, 0.0117737, 0.0023350, 0.0124512, -0.0084047, 0.0077910
3: -0.0066320, -0.0023522, -0.0069404, -0.0023359, -0.0035461, 0.0038255
4: 0.0009868, 0.0028067, 0.0009798, 0.0029378, -0.0016267, 0.0015079
5: 0.0019414, 0.0137676, 0.0018963, 0.0146197, -0.0105709, 0.0097991
6: -0.0019535, 0.0010481, -0.0021698, 0.0010595, -0.0024871, 0.0026830
7: -0.0081920, -0.0004259, -0.0087516, -0.0003963, -0.0064349, 0.0069418
8: -0.0038723, 0.0002119, -0.0041665, 0.0002274, -0.0033841, 0.0036506
9: -0.0021095, 0.0026262, -0.0021276, 0.0029674, -0.0042331, 0.0039240

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027223, upper bound: 0.0025747
time: 1.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027223, upper bound: 0.0026220
time: 1.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9885420, 0.9956973, 0.9879171, 0.9957882, -0.0059757, 0.0064622
1: -0.0041190, -0.0023361, -0.0042747, -0.0023134, -0.0014890, 0.0016102
2: 0.0023260, 0.0117746, 0.0022059, 0.0125997, -0.0085333, 0.0078909
3: -0.0066324, -0.0023318, -0.0070080, -0.0022771, -0.0035916, 0.0038840
4: 0.0009781, 0.0028068, 0.0009548, 0.0029665, -0.0016516, 0.0015273
5: 0.0018850, 0.0137687, 0.0017339, 0.0148065, -0.0107327, 0.0099247
6: -0.0019538, 0.0010624, -0.0022172, 0.0011008, -0.0025190, 0.0027241
7: -0.0081928, -0.0003889, -0.0088743, -0.0002896, -0.0065174, 0.0070480
8: -0.0038726, 0.0002314, -0.0042310, 0.0002835, -0.0034274, 0.0037065
9: -0.0021321, 0.0026267, -0.0021926, 0.0030422, -0.0042978, 0.0039743

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027797, upper bound: 0.0025747
time: 1.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027797, upper bound: 0.0026220
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9884105, 0.9954953, 0.9887607, 0.9959766, -0.0062273, 0.0055365
1: -0.0041517, -0.0023864, -0.0040645, -0.0022665, -0.0015517, 0.0013795
2: 0.0025927, 0.0119481, 0.0019570, 0.0114857, -0.0073108, 0.0082230
3: -0.0067114, -0.0024532, -0.0065009, -0.0021639, -0.0037428, 0.0033276
4: 0.0010297, 0.0028404, 0.0009067, 0.0027509, -0.0014150, 0.0015915
5: 0.0022204, 0.0139870, 0.0014209, 0.0134054, -0.0091951, 0.0103424
6: -0.0020092, 0.0009773, -0.0018616, 0.0011802, -0.0026250, 0.0023338
7: -0.0083361, -0.0006091, -0.0079542, -0.0000841, -0.0067917, 0.0060383
8: -0.0039480, 0.0001155, -0.0037472, 0.0003916, -0.0035717, 0.0031755
9: -0.0019978, 0.0027141, -0.0023180, 0.0024812, -0.0036821, 0.0041415

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0025958
time: 1.28 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0025992
time: 1.27 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9884105, 0.9954953, 0.9880242, 0.9957840, -0.0055326, 0.0056566
1: -0.0041517, -0.0023864, -0.0042480, -0.0023145, -0.0013786, 0.0014095
2: 0.0025927, 0.0119481, 0.0022114, 0.0124582, -0.0074695, 0.0073058
3: -0.0067114, -0.0024532, -0.0069436, -0.0022797, -0.0033253, 0.0033998
4: 0.0010297, 0.0028404, 0.0009559, 0.0029391, -0.0014457, 0.0014140
5: 0.0022204, 0.0139870, 0.0017408, 0.0146286, -0.0093947, 0.0091888
6: -0.0020092, 0.0009773, -0.0021721, 0.0010990, -0.0023322, 0.0023845
7: -0.0083361, -0.0006091, -0.0087574, -0.0002942, -0.0060341, 0.0061693
8: -0.0039480, 0.0001155, -0.0041696, 0.0002811, -0.0031733, 0.0032444
9: -0.0019978, 0.0027141, -0.0021898, 0.0029710, -0.0037620, 0.0036796

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0025958
time: 1.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0025992
time: 1.45 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9882876, 0.9956000, 0.9887598, 0.9960143, -0.0063029, 0.0056145
1: -0.0041824, -0.0023603, -0.0040647, -0.0022571, -0.0015705, 0.0013990
2: 0.0024544, 0.0121104, 0.0019074, 0.0114869, -0.0074138, 0.0083229
3: -0.0067852, -0.0023903, -0.0065014, -0.0021413, -0.0037882, 0.0033745
4: 0.0010029, 0.0028718, 0.0008971, 0.0027511, -0.0014349, 0.0016109
5: 0.0020464, 0.0141911, 0.0013585, 0.0134069, -0.0093246, 0.0104680
6: -0.0020610, 0.0010214, -0.0018620, 0.0011960, -0.0026569, 0.0023667
7: -0.0084701, -0.0004949, -0.0079551, -0.0000431, -0.0068742, 0.0061234
8: -0.0040185, 0.0001756, -0.0037477, 0.0004132, -0.0036151, 0.0032202
9: -0.0020675, 0.0027958, -0.0023430, 0.0024818, -0.0037340, 0.0041918

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026500
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026533
time: 1.50 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9882876, 0.9956000, 0.9880232, 0.9958220, -0.0056000, 0.0057377
1: -0.0041824, -0.0023603, -0.0042483, -0.0023050, -0.0013954, 0.0014297
2: 0.0024544, 0.0121104, 0.0021614, 0.0124596, -0.0075766, 0.0073948
3: -0.0067852, -0.0023903, -0.0069442, -0.0022569, -0.0033658, 0.0034485
4: 0.0010029, 0.0028718, 0.0009462, 0.0029394, -0.0014664, 0.0014312
5: 0.0020464, 0.0141911, 0.0016779, 0.0146303, -0.0095293, 0.0093007
6: -0.0020610, 0.0010214, -0.0021725, 0.0011150, -0.0023606, 0.0024186
7: -0.0084701, -0.0004949, -0.0087585, -0.0002529, -0.0061076, 0.0062578
8: -0.0040185, 0.0001756, -0.0041702, 0.0003029, -0.0032119, 0.0032909
9: -0.0020675, 0.0027958, -0.0022151, 0.0029717, -0.0038160, 0.0037244

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026500
time: 1.96 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026533
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9884210, 0.9953738, 0.9881930, 0.9958961, -0.0062974, 0.0059965
1: -0.0041491, -0.0024167, -0.0042059, -0.0022865, -0.0015692, 0.0014942
2: 0.0027531, 0.0119343, 0.0020634, 0.0122353, -0.0079183, 0.0083157
3: -0.0067051, -0.0025262, -0.0068421, -0.0022123, -0.0037850, 0.0036041
4: 0.0010607, 0.0028377, 0.0009273, 0.0028960, -0.0015326, 0.0016095
5: 0.0024221, 0.0139696, 0.0015546, 0.0143483, -0.0099591, 0.0104590
6: -0.0020048, 0.0009261, -0.0021009, 0.0011462, -0.0026546, 0.0025277
7: -0.0083247, -0.0007416, -0.0085733, -0.0001719, -0.0068683, 0.0065400
8: -0.0039420, 0.0000459, -0.0040728, 0.0003454, -0.0036120, 0.0034393
9: -0.0019170, 0.0027071, -0.0022644, 0.0028587, -0.0039881, 0.0041882

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0025667
time: 1.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0025921
time: 1.74 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9884210, 0.9953738, 0.9874700, 0.9956874, -0.0055832, 0.0062043
1: -0.0041491, -0.0024167, -0.0043861, -0.0023385, -0.0013912, 0.0015460
2: 0.0027531, 0.0119343, 0.0023390, 0.0131901, -0.0081928, 0.0073726
3: -0.0067051, -0.0025262, -0.0072767, -0.0023377, -0.0033557, 0.0037290
4: 0.0010607, 0.0028377, 0.0009806, 0.0030808, -0.0015857, 0.0014270
5: 0.0024221, 0.0139696, 0.0019013, 0.0155491, -0.0103043, 0.0092728
6: -0.0020048, 0.0009261, -0.0024057, 0.0010583, -0.0023535, 0.0026154
7: -0.0083247, -0.0007416, -0.0093619, -0.0003996, -0.0060893, 0.0067667
8: -0.0039420, 0.0000459, -0.0044875, 0.0002257, -0.0032023, 0.0035585
9: -0.0019170, 0.0027071, -0.0021256, 0.0033396, -0.0041263, 0.0037132

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0025667
time: 1.86 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0025921
time: 1.24 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9882984, 0.9954761, 0.9881920, 0.9959261, -0.0063734, 0.0060723
1: -0.0041797, -0.0023912, -0.0042062, -0.0022791, -0.0015881, 0.0015131
2: 0.0026181, 0.0120962, 0.0020238, 0.0122366, -0.0080185, 0.0084160
3: -0.0067788, -0.0024648, -0.0068427, -0.0021943, -0.0038306, 0.0036497
4: 0.0010346, 0.0028691, 0.0009196, 0.0028963, -0.0015520, 0.0016289
5: 0.0022523, 0.0141732, 0.0015049, 0.0143499, -0.0100851, 0.0105851
6: -0.0020565, 0.0009692, -0.0021013, 0.0011589, -0.0026866, 0.0025597
7: -0.0084584, -0.0006301, -0.0085744, -0.0001393, -0.0069511, 0.0066228
8: -0.0040123, 0.0001045, -0.0040733, 0.0003626, -0.0036555, 0.0034828
9: -0.0019850, 0.0027886, -0.0022843, 0.0028594, -0.0040385, 0.0042387

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026251
time: 1.76 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026448
time: 1.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9882984, 0.9954761, 0.9874691, 0.9957181, -0.0056542, 0.0062857
1: -0.0041797, -0.0023912, -0.0043863, -0.0023309, -0.0014089, 0.0015662
2: 0.0026181, 0.0120962, 0.0022986, 0.0131913, -0.0083002, 0.0074663
3: -0.0067788, -0.0024648, -0.0072772, -0.0023193, -0.0033984, 0.0037779
4: 0.0010346, 0.0028691, 0.0009728, 0.0030810, -0.0016065, 0.0014451
5: 0.0022523, 0.0141732, 0.0018504, 0.0155507, -0.0104395, 0.0093907
6: -0.0020565, 0.0009692, -0.0024061, 0.0010712, -0.0023835, 0.0026497
7: -0.0084584, -0.0006301, -0.0093629, -0.0003662, -0.0061667, 0.0068555
8: -0.0040123, 0.0001045, -0.0044880, 0.0002433, -0.0032430, 0.0036052
9: -0.0019850, 0.0027886, -0.0021459, 0.0033402, -0.0041804, 0.0037604

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026251
time: 1.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026448
time: 1.91 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9887607, 0.9959766, 0.9890842, 0.9957697, -0.0056300, 0.0055015
1: -0.0040645, -0.0022665, -0.0039838, -0.0023180, -0.0014028, 0.0013708
2: 0.0019570, 0.0114857, 0.0022304, 0.0110583, -0.0072647, 0.0074343
3: -0.0065009, -0.0021639, -0.0063064, -0.0022883, -0.0033838, 0.0033066
4: 0.0009067, 0.0027509, 0.0009596, 0.0026682, -0.0014061, 0.0014389
5: 0.0014209, 0.0134054, 0.0017647, 0.0128679, -0.0091371, 0.0093504
6: -0.0018616, 0.0011802, -0.0017252, 0.0010929, -0.0023732, 0.0023191
7: -0.0079542, -0.0000841, -0.0076012, -0.0003099, -0.0061403, 0.0060002
8: -0.0037472, 0.0003916, -0.0035615, 0.0002729, -0.0032291, 0.0031554
9: -0.0023180, 0.0024812, -0.0021803, 0.0022659, -0.0036589, 0.0037443

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025010, upper bound: 0.0024646
time: 1.29 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026021, upper bound: 0.0026717
time: 1.73 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9887598, 0.9960143, 0.9889615, 0.9958715, -0.0057068, 0.0055660
1: -0.0040647, -0.0022571, -0.0040144, -0.0022927, -0.0014220, 0.0013869
2: 0.0019074, 0.0114869, 0.0020960, 0.0112204, -0.0073498, 0.0075357
3: -0.0065014, -0.0021413, -0.0063802, -0.0022271, -0.0034299, 0.0033453
4: 0.0008971, 0.0027511, 0.0009336, 0.0026996, -0.0014225, 0.0014585
5: 0.0013585, 0.0134069, 0.0015956, 0.0130718, -0.0092441, 0.0094780
6: -0.0018620, 0.0011960, -0.0017769, 0.0011358, -0.0024056, 0.0023463
7: -0.0079551, -0.0000431, -0.0077351, -0.0001989, -0.0062240, 0.0060705
8: -0.0037477, 0.0004132, -0.0036320, 0.0003313, -0.0032732, 0.0031924
9: -0.0023430, 0.0024818, -0.0022480, 0.0023476, -0.0037017, 0.0037954

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025435, upper bound: 0.0024646
time: 1.67 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026478, upper bound: 0.0026717
time: 1.26 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9881929, 0.9958961, 0.9890956, 0.9956468, -0.0061870, 0.0054897
1: -0.0042060, -0.0022865, -0.0039810, -0.0023486, -0.0015416, 0.0013679
2: 0.0020634, 0.0122354, 0.0023926, 0.0110434, -0.0072491, 0.0081699
3: -0.0068422, -0.0022123, -0.0062996, -0.0023621, -0.0037186, 0.0032995
4: 0.0009273, 0.0028960, 0.0009910, 0.0026653, -0.0014030, 0.0015813
5: 0.0015546, 0.0143484, 0.0019687, 0.0128491, -0.0091174, 0.0102756
6: -0.0021009, 0.0011462, -0.0017204, 0.0010412, -0.0026080, 0.0023141
7: -0.0085734, -0.0001719, -0.0075889, -0.0004438, -0.0067478, 0.0059873
8: -0.0040728, 0.0003454, -0.0035550, 0.0002024, -0.0035486, 0.0031486
9: -0.0022644, 0.0028588, -0.0020986, 0.0022584, -0.0036510, 0.0041148

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0024950, upper bound: 0.0024659
time: 1.26 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026021, upper bound: 0.0027460
time: 1.41 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9881920, 0.9959261, 0.9889740, 0.9957497, -0.0062645, 0.0055592
1: -0.0042062, -0.0022791, -0.0040114, -0.0023230, -0.0015609, 0.0013852
2: 0.0020238, 0.0122367, 0.0022567, 0.0112041, -0.0073409, 0.0082722
3: -0.0068427, -0.0021943, -0.0063727, -0.0023003, -0.0037651, 0.0033412
4: 0.0009196, 0.0028963, 0.0009647, 0.0026964, -0.0014208, 0.0016011
5: 0.0015049, 0.0143500, 0.0017978, 0.0130512, -0.0092329, 0.0104042
6: -0.0021014, 0.0011589, -0.0017717, 0.0010845, -0.0026407, 0.0023434
7: -0.0085745, -0.0001393, -0.0077216, -0.0003316, -0.0068323, 0.0060631
8: -0.0040734, 0.0003626, -0.0036248, 0.0002615, -0.0035930, 0.0031885
9: -0.0022843, 0.0028594, -0.0021670, 0.0023393, -0.0036972, 0.0041663

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025343, upper bound: 0.0024659
time: 1.30 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026478, upper bound: 0.0027460
time: 1.25 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9887607, 0.9959766, 0.9884105, 0.9954953, -0.0055365, 0.0062273
1: -0.0040645, -0.0022665, -0.0041517, -0.0023864, -0.0013795, 0.0015517
2: 0.0019570, 0.0114857, 0.0025927, 0.0119481, -0.0082230, 0.0073108
3: -0.0065009, -0.0021639, -0.0067114, -0.0024532, -0.0033276, 0.0037428
4: 0.0009067, 0.0027509, 0.0010297, 0.0028404, -0.0015915, 0.0014150
5: 0.0014209, 0.0134054, 0.0022204, 0.0139870, -0.0103424, 0.0091951
6: -0.0018616, 0.0011802, -0.0020092, 0.0009773, -0.0023338, 0.0026250
7: -0.0079542, -0.0000841, -0.0083361, -0.0006091, -0.0060383, 0.0067917
8: -0.0037472, 0.0003916, -0.0039480, 0.0001155, -0.0031755, 0.0035717
9: -0.0023180, 0.0024812, -0.0019978, 0.0027141, -0.0041415, 0.0036821

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026032, upper bound: 0.0026061
time: 1.57 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026032, upper bound: 0.0026556
time: 1.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9887598, 0.9960143, 0.9882876, 0.9956000, -0.0056145, 0.0063029
1: -0.0040647, -0.0022571, -0.0041824, -0.0023603, -0.0013990, 0.0015705
2: 0.0019074, 0.0114869, 0.0024544, 0.0121104, -0.0083229, 0.0074138
3: -0.0065014, -0.0021413, -0.0067852, -0.0023903, -0.0033745, 0.0037882
4: 0.0008971, 0.0027511, 0.0010029, 0.0028718, -0.0016109, 0.0014349
5: 0.0013585, 0.0134069, 0.0020464, 0.0141911, -0.0104680, 0.0093246
6: -0.0018620, 0.0011960, -0.0020610, 0.0010214, -0.0023667, 0.0026569
7: -0.0079551, -0.0000431, -0.0084701, -0.0004949, -0.0061234, 0.0068742
8: -0.0037477, 0.0004132, -0.0040185, 0.0001756, -0.0032202, 0.0036151
9: -0.0023430, 0.0024818, -0.0020675, 0.0027958, -0.0041918, 0.0037340

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026557, upper bound: 0.0026061
time: 1.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026557, upper bound: 0.0026556
time: 1.23 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9881929, 0.9958961, 0.9884210, 0.9953738, -0.0060949, 0.0062974
1: -0.0042060, -0.0022865, -0.0041491, -0.0024167, -0.0015187, 0.0015692
2: 0.0020634, 0.0122354, 0.0027531, 0.0119343, -0.0083157, 0.0080483
3: -0.0068422, -0.0022123, -0.0067051, -0.0025262, -0.0036632, 0.0037850
4: 0.0009273, 0.0028960, 0.0010607, 0.0028377, -0.0016095, 0.0015577
5: 0.0015546, 0.0143484, 0.0024221, 0.0139696, -0.0104590, 0.0101226
6: -0.0021009, 0.0011462, -0.0020048, 0.0009261, -0.0025692, 0.0026546
7: -0.0085734, -0.0001719, -0.0083247, -0.0007416, -0.0066474, 0.0068683
8: -0.0040728, 0.0003454, -0.0039420, 0.0000459, -0.0034958, 0.0036120
9: -0.0022644, 0.0028588, -0.0019170, 0.0027071, -0.0041882, 0.0040535

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026015, upper bound: 0.0026648
time: 1.58 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026015, upper bound: 0.0027107
time: 1.26 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9881920, 0.9959261, 0.9882984, 0.9954761, -0.0061736, 0.0063734
1: -0.0042062, -0.0022791, -0.0041797, -0.0023912, -0.0015383, 0.0015881
2: 0.0020238, 0.0122367, 0.0026181, 0.0120962, -0.0084160, 0.0081522
3: -0.0068427, -0.0021943, -0.0067788, -0.0024648, -0.0037105, 0.0038306
4: 0.0009196, 0.0028963, 0.0010346, 0.0028691, -0.0016289, 0.0015778
5: 0.0015049, 0.0143500, 0.0022523, 0.0141732, -0.0105851, 0.0102533
6: -0.0021014, 0.0011589, -0.0020565, 0.0009692, -0.0026024, 0.0026866
7: -0.0085745, -0.0001393, -0.0084584, -0.0006301, -0.0067332, 0.0069511
8: -0.0040734, 0.0003626, -0.0040123, 0.0001045, -0.0035409, 0.0036555
9: -0.0022843, 0.0028594, -0.0019850, 0.0027886, -0.0042387, 0.0041059

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026544, upper bound: 0.0026648
time: 1.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026544, upper bound: 0.0027107
time: 1.59 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9880171, 0.9958012, 0.9890911, 0.9957576, -0.0063761, 0.0053591
1: -0.0042498, -0.0023102, -0.0039822, -0.0023210, -0.0015888, 0.0013353
2: 0.0021888, 0.0124676, 0.0022463, 0.0110494, -0.0070766, 0.0084196
3: -0.0069478, -0.0022694, -0.0063023, -0.0022956, -0.0038322, 0.0032210
4: 0.0009515, 0.0029410, 0.0009627, 0.0026665, -0.0013697, 0.0016296
5: 0.0017124, 0.0146404, 0.0017847, 0.0128567, -0.0089005, 0.0105896
6: -0.0021751, 0.0011062, -0.0017223, 0.0010878, -0.0026878, 0.0022591
7: -0.0087652, -0.0002755, -0.0075939, -0.0003230, -0.0069540, 0.0058449
8: -0.0041737, 0.0002910, -0.0035577, 0.0002660, -0.0036571, 0.0030738
9: -0.0022012, 0.0029757, -0.0021723, 0.0022615, -0.0035642, 0.0042405

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027123
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027193
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9880171, 0.9958012, 0.9884161, 0.9954856, -0.0057063, 0.0055920
1: -0.0042498, -0.0023102, -0.0041503, -0.0023888, -0.0014219, 0.0013934
2: 0.0021888, 0.0124676, 0.0026055, 0.0119407, -0.0073842, 0.0075352
3: -0.0069478, -0.0022694, -0.0067080, -0.0024590, -0.0034297, 0.0033610
4: 0.0009515, 0.0029410, 0.0010322, 0.0028390, -0.0014292, 0.0014584
5: 0.0017124, 0.0146404, 0.0022364, 0.0139777, -0.0092874, 0.0094773
6: -0.0021751, 0.0011062, -0.0020069, 0.0009732, -0.0024054, 0.0023572
7: -0.0087652, -0.0002755, -0.0083300, -0.0006196, -0.0062236, 0.0060989
8: -0.0041737, 0.0002910, -0.0039448, 0.0001100, -0.0032729, 0.0032074
9: -0.0022012, 0.0029757, -0.0019914, 0.0027103, -0.0037191, 0.0037951

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027123
time: 1.85 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027193
time: 1.90 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9879042, 0.9959075, 0.9890904, 0.9957986, -0.0064622, 0.0054351
1: -0.0042779, -0.0022837, -0.0039823, -0.0023109, -0.0016102, 0.0013543
2: 0.0020484, 0.0126167, 0.0021923, 0.0110503, -0.0071770, 0.0085332
3: -0.0070157, -0.0022054, -0.0063027, -0.0022710, -0.0038840, 0.0032667
4: 0.0009243, 0.0029698, 0.0009522, 0.0026666, -0.0013891, 0.0016516
5: 0.0015358, 0.0148279, 0.0017168, 0.0128578, -0.0090268, 0.0107326
6: -0.0022227, 0.0011510, -0.0017226, 0.0011051, -0.0027240, 0.0022911
7: -0.0088883, -0.0001595, -0.0075945, -0.0002784, -0.0070479, 0.0059278
8: -0.0042384, 0.0003520, -0.0035580, 0.0002894, -0.0037064, 0.0031174
9: -0.0022720, 0.0030508, -0.0021995, 0.0022619, -0.0036147, 0.0042978

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027732
time: 1.46 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027787
time: 1.42 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9879042, 0.9959075, 0.9884154, 0.9955269, -0.0057854, 0.0056736
1: -0.0042779, -0.0022837, -0.0041506, -0.0023786, -0.0014416, 0.0014137
2: 0.0020484, 0.0126167, 0.0025511, 0.0119418, -0.0074920, 0.0076396
3: -0.0070157, -0.0022054, -0.0067085, -0.0024343, -0.0034772, 0.0034100
4: 0.0009243, 0.0029698, 0.0010216, 0.0028392, -0.0014501, 0.0014786
5: 0.0015358, 0.0148279, 0.0021680, 0.0139791, -0.0094229, 0.0096086
6: -0.0022227, 0.0011510, -0.0020072, 0.0009906, -0.0024388, 0.0023916
7: -0.0088883, -0.0001595, -0.0083309, -0.0005747, -0.0063098, 0.0061879
8: -0.0042384, 0.0003520, -0.0039453, 0.0001336, -0.0033183, 0.0032542
9: -0.0022720, 0.0030508, -0.0020188, 0.0027109, -0.0037734, 0.0038477

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027732
time: 1.93 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027787
time: 1.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9880295, 0.9956904, 0.9885427, 0.9956633, -0.0063648, 0.0059001
1: -0.0042467, -0.0023378, -0.0041188, -0.0023445, -0.0015859, 0.0014702
2: 0.0023350, 0.0124512, 0.0023709, 0.0117737, -0.0077910, 0.0084047
3: -0.0069404, -0.0023359, -0.0066320, -0.0023522, -0.0038255, 0.0035461
4: 0.0009798, 0.0029378, 0.0009868, 0.0028067, -0.0015079, 0.0016267
5: 0.0018963, 0.0146197, 0.0019414, 0.0137676, -0.0097991, 0.0105709
6: -0.0021698, 0.0010595, -0.0019535, 0.0010481, -0.0026830, 0.0024871
7: -0.0087516, -0.0003963, -0.0081920, -0.0004259, -0.0069418, 0.0064349
8: -0.0041665, 0.0002274, -0.0038723, 0.0002119, -0.0036506, 0.0033841
9: -0.0021276, 0.0029674, -0.0021095, 0.0026262, -0.0039240, 0.0042331

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026678
time: 1.94 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027090
time: 1.44 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9880295, 0.9956904, 0.9878703, 0.9953730, -0.0056658, 0.0061315
1: -0.0042467, -0.0023378, -0.0042863, -0.0024169, -0.0014118, 0.0015278
2: 0.0023350, 0.0124512, 0.0027543, 0.0126613, -0.0080966, 0.0074816
3: -0.0069404, -0.0023359, -0.0070360, -0.0025268, -0.0034053, 0.0036852
4: 0.0009798, 0.0029378, 0.0010610, 0.0029785, -0.0015671, 0.0014481
5: 0.0018963, 0.0146197, 0.0024236, 0.0148841, -0.0101833, 0.0094099
6: -0.0021698, 0.0010595, -0.0022369, 0.0009257, -0.0023883, 0.0025846
7: -0.0087516, -0.0003963, -0.0089252, -0.0007426, -0.0061794, 0.0066872
8: -0.0041665, 0.0002274, -0.0042578, 0.0000453, -0.0032497, 0.0035168
9: -0.0021276, 0.0029674, -0.0019164, 0.0030733, -0.0040779, 0.0037682

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026678
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027090
time: 1.98 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9879171, 0.9957882, 0.9885420, 0.9956973, -0.0064622, 0.0059757
1: -0.0042747, -0.0023134, -0.0041190, -0.0023361, -0.0016102, 0.0014890
2: 0.0022059, 0.0125997, 0.0023260, 0.0117746, -0.0078909, 0.0085333
3: -0.0070080, -0.0022771, -0.0066324, -0.0023318, -0.0038840, 0.0035916
4: 0.0009548, 0.0029665, 0.0009781, 0.0028068, -0.0015273, 0.0016516
5: 0.0017339, 0.0148065, 0.0018850, 0.0137687, -0.0099247, 0.0107327
6: -0.0022172, 0.0011008, -0.0019538, 0.0010624, -0.0027241, 0.0025190
7: -0.0088743, -0.0002896, -0.0081928, -0.0003889, -0.0070480, 0.0065174
8: -0.0042310, 0.0002835, -0.0038726, 0.0002314, -0.0037065, 0.0034274
9: -0.0021926, 0.0030422, -0.0021321, 0.0026267, -0.0039743, 0.0042978

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027294
time: 1.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027665
time: 1.42 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9879171, 0.9957882, 0.9878692, 0.9954093, -0.0057497, 0.0062127
1: -0.0042747, -0.0023134, -0.0042866, -0.0024078, -0.0014327, 0.0015480
2: 0.0022059, 0.0125997, 0.0027061, 0.0126629, -0.0082038, 0.0075924
3: -0.0070080, -0.0022771, -0.0070367, -0.0025048, -0.0034558, 0.0037340
4: 0.0009548, 0.0029665, 0.0010517, 0.0029788, -0.0015878, 0.0014695
5: 0.0017339, 0.0148065, 0.0023630, 0.0148861, -0.0103182, 0.0095493
6: -0.0022172, 0.0011008, -0.0022374, 0.0009411, -0.0024237, 0.0026189
7: -0.0088743, -0.0002896, -0.0089265, -0.0007028, -0.0062709, 0.0067758
8: -0.0042310, 0.0002835, -0.0042585, 0.0000663, -0.0032978, 0.0035633
9: -0.0021926, 0.0030422, -0.0019407, 0.0030741, -0.0041319, 0.0038240

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027294
time: 1.86 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027666
time: 1.86 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9887607, 0.9959766, 0.9887465, 0.9959960, -0.0053679, 0.0053248
1: -0.0040645, -0.0022665, -0.0040680, -0.0022616, -0.0013375, 0.0013268
2: 0.0019570, 0.0114857, 0.0019315, 0.0115044, -0.0070313, 0.0070883
3: -0.0065009, -0.0021639, -0.0065094, -0.0021523, -0.0032263, 0.0032003
4: 0.0009067, 0.0027509, 0.0009017, 0.0027545, -0.0013609, 0.0013719
5: 0.0014209, 0.0134054, 0.0013888, 0.0134290, -0.0088435, 0.0089152
6: -0.0018616, 0.0011802, -0.0018676, 0.0011883, -0.0022628, 0.0022446
7: -0.0079542, -0.0000841, -0.0079696, -0.0000630, -0.0058545, 0.0058074
8: -0.0037472, 0.0003916, -0.0037553, 0.0004027, -0.0030788, 0.0030541
9: -0.0023180, 0.0024812, -0.0023308, 0.0024906, -0.0035413, 0.0035700

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025119, upper bound: 0.0024653
time: 1.58 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026047, upper bound: 0.0026725
time: 1.83 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9887598, 0.9960143, 0.9886309, 0.9960932, -0.0054402, 0.0053922
1: -0.0040647, -0.0022571, -0.0040968, -0.0022374, -0.0013555, 0.0013436
2: 0.0019074, 0.0114869, 0.0018032, 0.0116571, -0.0071203, 0.0071837
3: -0.0065014, -0.0021413, -0.0065789, -0.0020939, -0.0032697, 0.0032408
4: 0.0008971, 0.0027511, 0.0008769, 0.0027841, -0.0013781, 0.0013904
5: 0.0013585, 0.0134069, 0.0012274, 0.0136211, -0.0089554, 0.0090352
6: -0.0018620, 0.0011960, -0.0019163, 0.0012293, -0.0022932, 0.0022730
7: -0.0079551, -0.0000431, -0.0080958, 0.0000430, -0.0059333, 0.0058809
8: -0.0037477, 0.0004132, -0.0038216, 0.0004584, -0.0031203, 0.0030927
9: -0.0023430, 0.0024818, -0.0023954, 0.0025675, -0.0035861, 0.0036181

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025545, upper bound: 0.0024653
time: 1.32 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026505, upper bound: 0.0026725
time: 1.31 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9881929, 0.9958961, 0.9887606, 0.9958824, -0.0059098, 0.0053834
1: -0.0042060, -0.0022865, -0.0040645, -0.0022900, -0.0014726, 0.0013414
2: 0.0020634, 0.0122354, 0.0020816, 0.0114857, -0.0071088, 0.0078038
3: -0.0068422, -0.0022123, -0.0065009, -0.0022206, -0.0035520, 0.0032356
4: 0.0009273, 0.0028960, 0.0009308, 0.0027509, -0.0013759, 0.0015104
5: 0.0015546, 0.0143484, 0.0015775, 0.0134055, -0.0089409, 0.0098151
6: -0.0021009, 0.0011462, -0.0018616, 0.0011404, -0.0024912, 0.0022693
7: -0.0085734, -0.0001719, -0.0079542, -0.0001870, -0.0064455, 0.0058714
8: -0.0040728, 0.0003454, -0.0037472, 0.0003375, -0.0033896, 0.0030877
9: -0.0022644, 0.0028588, -0.0022552, 0.0024812, -0.0035803, 0.0039304

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 177

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025099, upper bound: 0.0024670
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026047, upper bound: 0.0027467
time: 1.43 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9881920, 0.9959261, 0.9886454, 0.9959788, -0.0059809, 0.0054599
1: -0.0042062, -0.0022791, -0.0040932, -0.0022659, -0.0014903, 0.0013605
2: 0.0020238, 0.0122367, 0.0019543, 0.0116380, -0.0072097, 0.0078978
3: -0.0068427, -0.0021943, -0.0065702, -0.0021626, -0.0035947, 0.0032816
4: 0.0009196, 0.0028963, 0.0009061, 0.0027804, -0.0013954, 0.0015286
5: 0.0015049, 0.0143500, 0.0014174, 0.0135970, -0.0090679, 0.0099333
6: -0.0021014, 0.0011589, -0.0019102, 0.0011811, -0.0025212, 0.0023015
7: -0.0085745, -0.0001393, -0.0080800, -0.0000818, -0.0065231, 0.0059548
8: -0.0040734, 0.0003626, -0.0038133, 0.0003928, -0.0034304, 0.0031316
9: -0.0022843, 0.0028594, -0.0023194, 0.0025579, -0.0036312, 0.0039777

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025502, upper bound: 0.0024670
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026505, upper bound: 0.0027467
time: 1.62 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9887607, 0.9959766, 0.9880171, 0.9958012, -0.0052612, 0.0061374
1: -0.0040645, -0.0022665, -0.0042498, -0.0023102, -0.0013110, 0.0015293
2: 0.0019570, 0.0114857, 0.0021888, 0.0124676, -0.0081043, 0.0069474
3: -0.0065009, -0.0021639, -0.0069478, -0.0022694, -0.0031622, 0.0036887
4: 0.0009067, 0.0027509, 0.0009515, 0.0029410, -0.0015686, 0.0013447
5: 0.0014209, 0.0134054, 0.0017124, 0.0146404, -0.0101931, 0.0087380
6: -0.0018616, 0.0011802, -0.0021751, 0.0011062, -0.0022178, 0.0025871
7: -0.0079542, -0.0000841, -0.0087652, -0.0002755, -0.0057381, 0.0066937
8: -0.0037472, 0.0003916, -0.0041737, 0.0002910, -0.0030176, 0.0035201
9: -0.0023180, 0.0024812, -0.0022012, 0.0029757, -0.0040818, 0.0034991

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026231, upper bound: 0.0026110
time: 1.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026231, upper bound: 0.0026607
time: 1.32 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9887598, 0.9960143, 0.9879042, 0.9959075, -0.0053349, 0.0062146
1: -0.0040647, -0.0022571, -0.0042779, -0.0022837, -0.0013293, 0.0015485
2: 0.0019074, 0.0114869, 0.0020484, 0.0126167, -0.0082063, 0.0070446
3: -0.0065014, -0.0021413, -0.0070157, -0.0022054, -0.0032064, 0.0037352
4: 0.0008971, 0.0027511, 0.0009243, 0.0029698, -0.0015883, 0.0013635
5: 0.0013585, 0.0134069, 0.0015358, 0.0148279, -0.0103214, 0.0088603
6: -0.0018620, 0.0011960, -0.0022227, 0.0011510, -0.0022488, 0.0026197
7: -0.0079551, -0.0000431, -0.0088883, -0.0001595, -0.0058184, 0.0067779
8: -0.0037477, 0.0004132, -0.0042384, 0.0003520, -0.0030599, 0.0035644
9: -0.0023430, 0.0024818, -0.0022720, 0.0030508, -0.0041331, 0.0035480

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026739, upper bound: 0.0026110
time: 1.28 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026739, upper bound: 0.0026607
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9881929, 0.9958961, 0.9880295, 0.9956904, -0.0058062, 0.0061759
1: -0.0042060, -0.0022865, -0.0042467, -0.0023378, -0.0014468, 0.0015389
2: 0.0020634, 0.0122354, 0.0023350, 0.0124512, -0.0081553, 0.0076671
3: -0.0068422, -0.0022123, -0.0069404, -0.0023359, -0.0034897, 0.0037119
4: 0.0009273, 0.0028960, 0.0009798, 0.0029378, -0.0015784, 0.0014839
5: 0.0015546, 0.0143484, 0.0018963, 0.0146197, -0.0102572, 0.0096431
6: -0.0021009, 0.0011462, -0.0021698, 0.0010595, -0.0024475, 0.0026034
7: -0.0085734, -0.0001719, -0.0087516, -0.0003963, -0.0063325, 0.0067357
8: -0.0040728, 0.0003454, -0.0041665, 0.0002274, -0.0033302, 0.0035423
9: -0.0022644, 0.0028588, -0.0021276, 0.0029674, -0.0041074, 0.0038615

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 177

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026224, upper bound: 0.0026719
time: 1.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026224, upper bound: 0.0027173
time: 1.65 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9881920, 0.9959261, 0.9879171, 0.9957882, -0.0058788, 0.0062624
1: -0.0042062, -0.0022791, -0.0042747, -0.0023134, -0.0014648, 0.0015604
2: 0.0020238, 0.0122367, 0.0022059, 0.0125997, -0.0082694, 0.0077629
3: -0.0068427, -0.0021943, -0.0070080, -0.0022771, -0.0035333, 0.0037639
4: 0.0009196, 0.0028963, 0.0009548, 0.0029665, -0.0016005, 0.0015025
5: 0.0015049, 0.0143500, 0.0017339, 0.0148065, -0.0104007, 0.0097636
6: -0.0021014, 0.0011589, -0.0022172, 0.0011008, -0.0024781, 0.0026398
7: -0.0085745, -0.0001393, -0.0088743, -0.0002896, -0.0064116, 0.0068300
8: -0.0040734, 0.0003626, -0.0042310, 0.0002835, -0.0033718, 0.0035918
9: -0.0022843, 0.0028594, -0.0021926, 0.0030422, -0.0041649, 0.0039098

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 177

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026736, upper bound: 0.0026719
time: 1.26 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026736, upper bound: 0.0027174
time: 1.77 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9880171, 0.9958012, 0.9887607, 0.9959766, -0.0061374, 0.0052612
1: -0.0042498, -0.0023102, -0.0040645, -0.0022665, -0.0015293, 0.0013110
2: 0.0021888, 0.0124676, 0.0019570, 0.0114857, -0.0069474, 0.0081043
3: -0.0069478, -0.0022694, -0.0065009, -0.0021639, -0.0036887, 0.0031622
4: 0.0009515, 0.0029410, 0.0009067, 0.0027509, -0.0013447, 0.0015686
5: 0.0017124, 0.0146404, 0.0014209, 0.0134054, -0.0087380, 0.0101931
6: -0.0021751, 0.0011062, -0.0018616, 0.0011802, -0.0025871, 0.0022178
7: -0.0087652, -0.0002755, -0.0079542, -0.0000841, -0.0066937, 0.0057381
8: -0.0041737, 0.0002910, -0.0037472, 0.0003916, -0.0035201, 0.0030176
9: -0.0022012, 0.0029757, -0.0023180, 0.0024812, -0.0034991, 0.0040818

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027181
time: 1.39 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027263
time: 1.37 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9880171, 0.9958012, 0.9880242, 0.9957840, -0.0054687, 0.0054513
1: -0.0042498, -0.0023102, -0.0042480, -0.0023145, -0.0013626, 0.0013583
2: 0.0021888, 0.0124676, 0.0022114, 0.0124582, -0.0071983, 0.0072213
3: -0.0069478, -0.0022694, -0.0069436, -0.0022797, -0.0032868, 0.0032764
4: 0.0009515, 0.0029410, 0.0009559, 0.0029391, -0.0013932, 0.0013977
5: 0.0017124, 0.0146404, 0.0017408, 0.0146286, -0.0090536, 0.0090825
6: -0.0021751, 0.0011062, -0.0021721, 0.0010990, -0.0023052, 0.0022979
7: -0.0087652, -0.0002755, -0.0087574, -0.0002942, -0.0059644, 0.0059454
8: -0.0041737, 0.0002910, -0.0041696, 0.0002811, -0.0031366, 0.0031266
9: -0.0022012, 0.0029757, -0.0021898, 0.0029710, -0.0036255, 0.0036370

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027181
time: 1.85 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027263
time: 1.85 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9879042, 0.9959075, 0.9887598, 0.9960143, -0.0062146, 0.0053349
1: -0.0042779, -0.0022837, -0.0040647, -0.0022571, -0.0015485, 0.0013293
2: 0.0020484, 0.0126167, 0.0019074, 0.0114869, -0.0070446, 0.0082063
3: -0.0070157, -0.0022054, -0.0065014, -0.0021413, -0.0037352, 0.0032064
4: 0.0009243, 0.0029698, 0.0008971, 0.0027511, -0.0013635, 0.0015883
5: 0.0015358, 0.0148279, 0.0013585, 0.0134069, -0.0088603, 0.0103214
6: -0.0022227, 0.0011510, -0.0018620, 0.0011960, -0.0026197, 0.0022488
7: -0.0088883, -0.0001595, -0.0079551, -0.0000431, -0.0067779, 0.0058184
8: -0.0042384, 0.0003520, -0.0037477, 0.0004132, -0.0035644, 0.0030599
9: -0.0022720, 0.0030508, -0.0023430, 0.0024818, -0.0035480, 0.0041331

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027793
time: 1.25 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027867
time: 1.24 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9879042, 0.9959075, 0.9880232, 0.9958220, -0.0055382, 0.0055297
1: -0.0042779, -0.0022837, -0.0042483, -0.0023050, -0.0013800, 0.0013779
2: 0.0020484, 0.0126167, 0.0021614, 0.0124596, -0.0073019, 0.0073131
3: -0.0070157, -0.0022054, -0.0069442, -0.0022569, -0.0033286, 0.0033235
4: 0.0009243, 0.0029698, 0.0009462, 0.0029394, -0.0014133, 0.0014154
5: 0.0015358, 0.0148279, 0.0016779, 0.0146303, -0.0091839, 0.0091980
6: -0.0022227, 0.0011510, -0.0021725, 0.0011150, -0.0023346, 0.0023310
7: -0.0088883, -0.0001595, -0.0087585, -0.0002529, -0.0060402, 0.0060309
8: -0.0042384, 0.0003520, -0.0041702, 0.0003029, -0.0031765, 0.0031716
9: -0.0022720, 0.0030508, -0.0022151, 0.0029717, -0.0036776, 0.0036833

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027793
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027867
time: 1.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9880295, 0.9956904, 0.9881930, 0.9958961, -0.0061760, 0.0057263
1: -0.0042467, -0.0023378, -0.0042059, -0.0022865, -0.0015389, 0.0014268
2: 0.0023350, 0.0124512, 0.0020634, 0.0122353, -0.0075615, 0.0081553
3: -0.0069404, -0.0023359, -0.0068421, -0.0022123, -0.0037119, 0.0034417
4: 0.0009798, 0.0029378, 0.0009273, 0.0028960, -0.0014635, 0.0015784
5: 0.0018963, 0.0146197, 0.0015546, 0.0143483, -0.0095103, 0.0102572
6: -0.0021698, 0.0010595, -0.0021009, 0.0011462, -0.0026034, 0.0024138
7: -0.0087516, -0.0003963, -0.0085733, -0.0001719, -0.0067357, 0.0062453
8: -0.0041665, 0.0002274, -0.0040728, 0.0003454, -0.0035423, 0.0032843
9: -0.0021276, 0.0029674, -0.0022644, 0.0028587, -0.0038084, 0.0041074

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.10 + 596.35 = 600.45 seconds
