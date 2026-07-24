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
execution time: IAR + RelationalAnalysis = 1.66 + 2.23 = 3.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0032832, upper bound: 0.0032832

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0031766, upper bound: 0.0030743
time: 1.22 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0031766, upper bound: 0.0031766
time: 1.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.63 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.63
Output dim: 0, lower bound: -0.0031766, upper bound: 0.0030743
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.63
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

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0031127, upper bound: 0.0029591
time: 1.30 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0031127, upper bound: 0.0030106
time: 1.23 seconds

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

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0031127, upper bound: 0.0030550
time: 1.90 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0031127, upper bound: 0.0031127
time: 1.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.67 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 5.67
Output dim: 0, lower bound: -0.0031127, upper bound: 0.0029591
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 5.67
Output dim: 0, lower bound: -0.0031127, upper bound: 0.0030106
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 5.67
Output dim: 0, lower bound: -0.0031127, upper bound: 0.0030550
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 5.67
Output dim: 0, lower bound: -0.0031127, upper bound: 0.0031127

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: 0.9890695, 0.9959663, 0.9890010, 0.9963839, -0.0058674, 0.0054850
1: -0.0039876, -0.0022690, -0.0040046, -0.0021650, -0.0014620, 0.0013667
2: 0.0019707, 0.0110780, 0.0014193, 0.0111683, -0.0072428, 0.0077479
3: -0.0063153, -0.0021701, -0.0063564, -0.0019191, -0.0035265, 0.0032966
4: 0.0009093, 0.0026720, 0.0008026, 0.0026895, -0.0014018, 0.0014996
5: 0.0014381, 0.0128926, 0.0007446, 0.0130062, -0.0091096, 0.0097448
6: -0.0017314, 0.0011758, -0.0017603, 0.0013519, -0.0024733, 0.0023121
7: -0.0076174, -0.0000954, -0.0076920, 0.0003600, -0.0063993, 0.0059821
8: -0.0035701, 0.0003857, -0.0036093, 0.0006252, -0.0033653, 0.0031459
9: -0.0023111, 0.0022758, -0.0025888, 0.0023213, -0.0036479, 0.0039022

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029968, upper bound: 0.0027807
time: 1.46 seconds

## Relational analysis of NS_A1_A1_A2

### Relational analysis result of NS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029968, upper bound: 0.0028442
time: 1.43 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: 0.9889466, 0.9960691, 0.9890003, 0.9964190, -0.0059322, 0.0055628
1: -0.0040182, -0.0022434, -0.0040048, -0.0021562, -0.0014781, 0.0013861
2: 0.0018349, 0.0112403, 0.0013729, 0.0111693, -0.0073456, 0.0078334
3: -0.0063892, -0.0021083, -0.0063569, -0.0018980, -0.0035654, 0.0033434
4: 0.0008830, 0.0027034, 0.0007936, 0.0026897, -0.0014217, 0.0015161
5: 0.0012673, 0.0130968, 0.0006862, 0.0130075, -0.0092389, 0.0098524
6: -0.0017833, 0.0012192, -0.0017606, 0.0013667, -0.0025006, 0.0023449
7: -0.0077515, 0.0000168, -0.0076929, 0.0003983, -0.0064699, 0.0060670
8: -0.0036406, 0.0004447, -0.0036097, 0.0006453, -0.0034025, 0.0031906
9: -0.0023795, 0.0023576, -0.0026122, 0.0023218, -0.0036997, 0.0039453

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A1

### Relational analysis result of NS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029968, upper bound: 0.0028360
time: 1.62 seconds

## Relational analysis of NS_A1_A2_A2

### Relational analysis result of NS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029968, upper bound: 0.0028928
time: 1.86 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: 0.9887329, 0.9962082, 0.9889781, 0.9964823, -0.0065254, 0.0056919
1: -0.0040714, -0.0022088, -0.0040103, -0.0021405, -0.0016260, 0.0014183
2: 0.0016514, 0.0115223, 0.0012894, 0.0111986, -0.0075161, 0.0086167
3: -0.0065176, -0.0020248, -0.0063702, -0.0018600, -0.0039220, 0.0034210
4: 0.0008475, 0.0027580, 0.0007774, 0.0026954, -0.0014547, 0.0016677
5: 0.0010365, 0.0134515, 0.0005811, 0.0130443, -0.0094532, 0.0108376
6: -0.0018733, 0.0012778, -0.0017700, 0.0013933, -0.0027507, 0.0023993
7: -0.0079844, 0.0001684, -0.0077170, 0.0004674, -0.0071169, 0.0062078
8: -0.0037631, 0.0005244, -0.0036225, 0.0006816, -0.0037427, 0.0032646
9: -0.0024719, 0.0024996, -0.0026542, 0.0023366, -0.0037855, 0.0043398

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029247, upper bound: 0.0029489
time: 1.72 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029986, upper bound: 0.0029489
time: 1.66 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: 0.9886171, 0.9963088, 0.9889771, 0.9965158, -0.0066032, 0.0057567
1: -0.0041003, -0.0021837, -0.0040106, -0.0021321, -0.0016453, 0.0014344
2: 0.0015184, 0.0116754, 0.0012452, 0.0111999, -0.0076017, 0.0087195
3: -0.0065872, -0.0019642, -0.0063708, -0.0018399, -0.0039687, 0.0034599
4: 0.0008218, 0.0027876, 0.0007689, 0.0026956, -0.0014713, 0.0016876
5: 0.0008692, 0.0136440, 0.0005256, 0.0130460, -0.0095609, 0.0109668
6: -0.0019222, 0.0013202, -0.0017704, 0.0014074, -0.0027835, 0.0024267
7: -0.0081108, 0.0002782, -0.0077181, 0.0005038, -0.0072017, 0.0062785
8: -0.0038296, 0.0005821, -0.0036230, 0.0007008, -0.0037873, 0.0033018
9: -0.0025389, 0.0025767, -0.0026765, 0.0023372, -0.0038286, 0.0043916

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029247, upper bound: 0.0029986
time: 1.77 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029986, upper bound: 0.0029986
time: 1.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.13 seconds
NS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 5.13
Output dim: 0, lower bound: -0.0029968, upper bound: 0.0027807
NS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 5.13
Output dim: 0, lower bound: -0.0029968, upper bound: 0.0028442
NS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 5.13
Output dim: 0, lower bound: -0.0029968, upper bound: 0.0028360
NS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 5.13
Output dim: 0, lower bound: -0.0029968, upper bound: 0.0028928
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 5.13
Output dim: 0, lower bound: -0.0029247, upper bound: 0.0029489
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 5.13
Output dim: 0, lower bound: -0.0029986, upper bound: 0.0029489
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.13
Output dim: 0, lower bound: -0.0029247, upper bound: 0.0029986
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 5.13
Output dim: 0, lower bound: -0.0029986, upper bound: 0.0029986

## BFS NS instance: NS_A1_A1_A1

### Backsubstitution after applying NS history:
0: 0.9890941, 0.9957717, 0.9890057, 0.9963506, -0.0057522, 0.0052726
1: -0.0039814, -0.0023175, -0.0040034, -0.0021733, -0.0014333, 0.0013138
2: 0.0022277, 0.0110454, 0.0014632, 0.0111620, -0.0069624, 0.0075957
3: -0.0063005, -0.0022871, -0.0063536, -0.0019391, -0.0034572, 0.0031690
4: 0.0009591, 0.0026657, 0.0008111, 0.0026883, -0.0013476, 0.0014701
5: 0.0017613, 0.0128517, 0.0007998, 0.0129984, -0.0087568, 0.0095534
6: -0.0017211, 0.0010938, -0.0017583, 0.0013378, -0.0024248, 0.0022226
7: -0.0075906, -0.0003077, -0.0076869, 0.0003238, -0.0062736, 0.0057505
8: -0.0035559, 0.0002741, -0.0036066, 0.0006061, -0.0032992, 0.0030241
9: -0.0021816, 0.0022594, -0.0025667, 0.0023182, -0.0035066, 0.0038256

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027814, upper bound: 0.0025389
time: 1.21 seconds

## Relational analysis of NS_A1_A1_A1_A2

### Relational analysis result of NS_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027787, upper bound: 0.0025667
time: 1.26 seconds

## BFS NS instance: NS_A1_A1_A2

### Backsubstitution after applying NS history:
0: 0.9885452, 0.9956813, 0.9890209, 0.9962338, -0.0062997, 0.0053195
1: -0.0041182, -0.0023401, -0.0039996, -0.0022024, -0.0015697, 0.0013255
2: 0.0023471, 0.0117701, 0.0016176, 0.0111420, -0.0070243, 0.0083187
3: -0.0066303, -0.0023414, -0.0063445, -0.0020094, -0.0037863, 0.0031972
4: 0.0009822, 0.0028060, 0.0008410, 0.0026844, -0.0013595, 0.0016101
5: 0.0019115, 0.0137631, 0.0009940, 0.0129732, -0.0088348, 0.0104628
6: -0.0019524, 0.0010557, -0.0017519, 0.0012886, -0.0026556, 0.0022424
7: -0.0081891, -0.0004062, -0.0076703, 0.0001963, -0.0068707, 0.0058017
8: -0.0038707, 0.0002222, -0.0035979, 0.0005391, -0.0036133, 0.0030510
9: -0.0021215, 0.0026244, -0.0024889, 0.0023081, -0.0035378, 0.0041897

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_A1_A2_A1

### Relational analysis result of NS_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027797, upper bound: 0.0025747
time: 1.25 seconds

## Relational analysis of NS_A1_A1_A2_A2

### Relational analysis result of NS_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027665, upper bound: 0.0025921
time: 1.59 seconds

## BFS NS instance: NS_A1_A2_A1

### Backsubstitution after applying NS history:
0: 0.9889723, 0.9958820, 0.9890051, 0.9963866, -0.0058202, 0.0053503
1: -0.0040118, -0.0022901, -0.0040036, -0.0021643, -0.0014502, 0.0013332
2: 0.0020821, 0.0112063, 0.0014156, 0.0111631, -0.0070650, 0.0076855
3: -0.0063737, -0.0022208, -0.0063541, -0.0019175, -0.0034981, 0.0032157
4: 0.0009309, 0.0026968, 0.0008019, 0.0026885, -0.0013674, 0.0014875
5: 0.0015782, 0.0130540, 0.0007399, 0.0129996, -0.0088859, 0.0096663
6: -0.0017724, 0.0011403, -0.0017586, 0.0013530, -0.0024534, 0.0022553
7: -0.0077234, -0.0001874, -0.0076877, 0.0003631, -0.0063477, 0.0058353
8: -0.0036258, 0.0003373, -0.0036070, 0.0006268, -0.0033382, 0.0030687
9: -0.0022550, 0.0023404, -0.0025906, 0.0023187, -0.0035583, 0.0038708

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_A2_A1_A1

### Relational analysis result of NS_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027814, upper bound: 0.0025882
time: 1.63 seconds

## Relational analysis of NS_A1_A2_A1_A2

### Relational analysis result of NS_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027787, upper bound: 0.0026251
time: 1.35 seconds

## BFS NS instance: NS_A1_A2_A2

### Backsubstitution after applying NS history:
0: 0.9884464, 0.9957835, 0.9890200, 0.9962678, -0.0063834, 0.0053943
1: -0.0041428, -0.0023146, -0.0039999, -0.0021939, -0.0015906, 0.0013441
2: 0.0022121, 0.0119008, 0.0015726, 0.0111431, -0.0071231, 0.0084293
3: -0.0066898, -0.0022800, -0.0063450, -0.0019889, -0.0038366, 0.0032421
4: 0.0009560, 0.0028313, 0.0008323, 0.0026846, -0.0013787, 0.0016315
5: 0.0017417, 0.0139275, 0.0009374, 0.0129745, -0.0089589, 0.0106018
6: -0.0019941, 0.0010988, -0.0017522, 0.0013029, -0.0026908, 0.0022739
7: -0.0082970, -0.0002948, -0.0076712, 0.0002334, -0.0069620, 0.0058832
8: -0.0039275, 0.0002808, -0.0035984, 0.0005586, -0.0036613, 0.0030939
9: -0.0021895, 0.0026902, -0.0025116, 0.0023086, -0.0035876, 0.0042454

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_A2_A2_A1

### Relational analysis result of NS_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027797, upper bound: 0.0026220
time: 1.29 seconds

## Relational analysis of NS_A1_A2_A2_A2

### Relational analysis result of NS_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027665, upper bound: 0.0026448
time: 1.31 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9887372, 0.9961742, 0.9890091, 0.9962838, -0.0063048, 0.0055860
1: -0.0040703, -0.0022173, -0.0040026, -0.0021899, -0.0015710, 0.0013919
2: 0.0016963, 0.0115166, 0.0015515, 0.0111576, -0.0073762, 0.0083254
3: -0.0065150, -0.0020452, -0.0063516, -0.0019793, -0.0037894, 0.0033573
4: 0.0008562, 0.0027569, 0.0008282, 0.0026874, -0.0014277, 0.0016114
5: 0.0010929, 0.0134443, 0.0009108, 0.0129928, -0.0092774, 0.0104712
6: -0.0018715, 0.0012634, -0.0017569, 0.0013097, -0.0026577, 0.0023547
7: -0.0079797, 0.0001313, -0.0076832, 0.0002509, -0.0068763, 0.0060923
8: -0.0037606, 0.0005049, -0.0036047, 0.0005678, -0.0036162, 0.0032039
9: -0.0024493, 0.0024967, -0.0025222, 0.0023159, -0.0037151, 0.0041931

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026607, upper bound: 0.0027289
time: 1.25 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027386, upper bound: 0.0027263
time: 1.26 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9887514, 0.9960642, 0.9884681, 0.9962019, -0.0063563, 0.0061286
1: -0.0040668, -0.0022447, -0.0041374, -0.0022103, -0.0015838, 0.0015271
2: 0.0018415, 0.0114979, 0.0016596, 0.0118720, -0.0080928, 0.0083935
3: -0.0065065, -0.0021113, -0.0066767, -0.0020285, -0.0038203, 0.0036835
4: 0.0008843, 0.0027533, 0.0008491, 0.0028257, -0.0015663, 0.0016245
5: 0.0012756, 0.0134208, 0.0010468, 0.0138913, -0.0101786, 0.0105568
6: -0.0018655, 0.0012171, -0.0019849, 0.0012751, -0.0026794, 0.0025834
7: -0.0079643, 0.0000113, -0.0082732, 0.0001615, -0.0069325, 0.0066841
8: -0.0037525, 0.0004418, -0.0039149, 0.0005208, -0.0036457, 0.0035151
9: -0.0023762, 0.0024873, -0.0024678, 0.0026757, -0.0040760, 0.0042274

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027174, upper bound: 0.0027281
time: 1.22 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027786, upper bound: 0.0027196
time: 1.31 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9886214, 0.9962758, 0.9890083, 0.9963205, -0.0063818, 0.0056583
1: -0.0040992, -0.0021919, -0.0040028, -0.0021808, -0.0015902, 0.0014099
2: 0.0015621, 0.0116695, 0.0015031, 0.0111587, -0.0074717, 0.0084271
3: -0.0065846, -0.0019841, -0.0063521, -0.0019573, -0.0038357, 0.0034008
4: 0.0008302, 0.0027865, 0.0008188, 0.0026876, -0.0014461, 0.0016311
5: 0.0009241, 0.0136366, 0.0008499, 0.0129942, -0.0093975, 0.0105991
6: -0.0019203, 0.0013063, -0.0017572, 0.0013251, -0.0026902, 0.0023852
7: -0.0081060, 0.0002421, -0.0076841, 0.0002909, -0.0069603, 0.0061712
8: -0.0038270, 0.0005632, -0.0036051, 0.0005888, -0.0036603, 0.0032454
9: -0.0025169, 0.0025738, -0.0025466, 0.0023165, -0.0037632, 0.0042443

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026607, upper bound: 0.0027880
time: 1.61 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027385, upper bound: 0.0027867
time: 1.59 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9886359, 0.9961605, 0.9884673, 0.9962333, -0.0064344, 0.0061997
1: -0.0040956, -0.0022206, -0.0041376, -0.0022025, -0.0016033, 0.0015448
2: 0.0017143, 0.0116504, 0.0016182, 0.0118731, -0.0081867, 0.0084965
3: -0.0065759, -0.0020534, -0.0066772, -0.0020097, -0.0038672, 0.0037262
4: 0.0008597, 0.0027828, 0.0008411, 0.0028259, -0.0015845, 0.0016445
5: 0.0011155, 0.0136126, 0.0009948, 0.0138927, -0.0102967, 0.0106864
6: -0.0019142, 0.0012577, -0.0019853, 0.0012884, -0.0027123, 0.0026134
7: -0.0080902, 0.0001164, -0.0082742, 0.0001957, -0.0070176, 0.0067617
8: -0.0038187, 0.0004971, -0.0039154, 0.0005388, -0.0036905, 0.0035559
9: -0.0024402, 0.0025641, -0.0024886, 0.0026763, -0.0041232, 0.0042793

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027174, upper bound: 0.0027871
time: 1.56 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027787, upper bound: 0.0027787
time: 1.17 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.42 seconds
NS_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0027814, upper bound: 0.0025389
NS_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0027787, upper bound: 0.0025667
NS_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0027797, upper bound: 0.0025747
NS_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0027665, upper bound: 0.0025921
NS_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0027814, upper bound: 0.0025882
NS_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0027787, upper bound: 0.0026251
NS_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0027797, upper bound: 0.0026220
NS_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0027665, upper bound: 0.0026448
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0026607, upper bound: 0.0027289
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0027386, upper bound: 0.0027263
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0027174, upper bound: 0.0027281
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0027786, upper bound: 0.0027196
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0026607, upper bound: 0.0027880
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0027385, upper bound: 0.0027867
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0027174, upper bound: 0.0027871
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 0, lower bound: -0.0027787, upper bound: 0.0027787

## BFS NS instance: NS_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: 0.9890988, 0.9956085, 0.9890057, 0.9963506, -0.0057473, 0.0050888
1: -0.0039802, -0.0023582, -0.0040034, -0.0021733, -0.0014321, 0.0012680
2: 0.0024432, 0.0110391, 0.0014632, 0.0111620, -0.0067197, 0.0075892
3: -0.0062976, -0.0023852, -0.0063536, -0.0019391, -0.0034543, 0.0030585
4: 0.0010008, 0.0026645, 0.0008111, 0.0026883, -0.0013006, 0.0014689
5: 0.0020324, 0.0128438, 0.0007998, 0.0129984, -0.0084516, 0.0095453
6: -0.0017191, 0.0010250, -0.0017583, 0.0013378, -0.0024227, 0.0021451
7: -0.0075853, -0.0004857, -0.0076869, 0.0003238, -0.0062682, 0.0055500
8: -0.0035532, 0.0001804, -0.0036066, 0.0006061, -0.0032964, 0.0029187
9: -0.0020731, 0.0022563, -0.0025667, 0.0023182, -0.0033844, 0.0038223

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_A1_A1_A1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026564, upper bound: 0.0025389
time: 1.56 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026564, upper bound: 0.0025389
time: 1.64 seconds

## BFS NS instance: NS_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: 0.9884241, 0.9953319, 0.9890186, 0.9961642, -0.0064116, 0.0051111
1: -0.0041484, -0.0024271, -0.0040002, -0.0022197, -0.0015976, 0.0012735
2: 0.0028085, 0.0119302, 0.0017094, 0.0111452, -0.0067491, 0.0084665
3: -0.0067032, -0.0025514, -0.0063459, -0.0020512, -0.0038536, 0.0030719
4: 0.0010715, 0.0028369, 0.0008587, 0.0026850, -0.0013063, 0.0016387
5: 0.0024918, 0.0139644, 0.0011095, 0.0129771, -0.0084886, 0.0106486
6: -0.0020035, 0.0009084, -0.0017529, 0.0012592, -0.0027027, 0.0021545
7: -0.0083213, -0.0007874, -0.0076729, 0.0001204, -0.0069928, 0.0055743
8: -0.0039402, 0.0000218, -0.0035993, 0.0004992, -0.0036775, 0.0029315
9: -0.0018891, 0.0027050, -0.0024427, 0.0023097, -0.0033992, 0.0042642

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_A1_A1_A2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026536, upper bound: 0.0025667
time: 1.22 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026536, upper bound: 0.0025667
time: 1.65 seconds

## BFS NS instance: NS_A1_A1_A2_A1

### Backsubstitution after applying NS history:
0: 0.9885501, 0.9955158, 0.9890209, 0.9962338, -0.0062948, 0.0051364
1: -0.0041170, -0.0023813, -0.0039996, -0.0022024, -0.0015685, 0.0012799
2: 0.0025657, 0.0117637, 0.0016176, 0.0111420, -0.0067826, 0.0083122
3: -0.0066275, -0.0024409, -0.0063445, -0.0020094, -0.0037834, 0.0030871
4: 0.0010245, 0.0028047, 0.0008410, 0.0026844, -0.0013127, 0.0016088
5: 0.0021864, 0.0137551, 0.0009940, 0.0129732, -0.0085307, 0.0104546
6: -0.0019504, 0.0009859, -0.0017519, 0.0012886, -0.0026535, 0.0021652
7: -0.0081838, -0.0005868, -0.0076703, 0.0001963, -0.0068654, 0.0056020
8: -0.0038679, 0.0001272, -0.0035979, 0.0005391, -0.0036104, 0.0029460
9: -0.0020114, 0.0026212, -0.0024889, 0.0023081, -0.0034161, 0.0041865

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_A1_A2_A1_B1

### Relational analysis result of NS_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026547, upper bound: 0.0025747
time: 1.73 seconds

## Relational analysis of NS_A1_A1_A2_A1_B2

### Relational analysis result of NS_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026547, upper bound: 0.0025747
time: 1.68 seconds

## BFS NS instance: NS_A1_A1_A2_A2

### Backsubstitution after applying NS history:
0: 0.9878775, 0.9952232, 0.9890339, 0.9960473, -0.0069414, 0.0051631
1: -0.0042845, -0.0024542, -0.0039964, -0.0022489, -0.0017296, 0.0012865
2: 0.0029521, 0.0126518, 0.0018638, 0.0111248, -0.0068178, 0.0091661
3: -0.0070317, -0.0026168, -0.0063367, -0.0021214, -0.0041720, 0.0031032
4: 0.0010993, 0.0029766, 0.0008886, 0.0026811, -0.0013196, 0.0017741
5: 0.0026724, 0.0148721, 0.0013036, 0.0129516, -0.0085750, 0.0115285
6: -0.0022339, 0.0008625, -0.0017464, 0.0012100, -0.0029261, 0.0021764
7: -0.0089173, -0.0009060, -0.0076561, -0.0000071, -0.0075706, 0.0056311
8: -0.0042537, -0.0000406, -0.0035904, 0.0004321, -0.0039813, 0.0029613
9: -0.0018168, 0.0030685, -0.0023649, 0.0022994, -0.0034338, 0.0046165

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_A1_A2_A2_B1

### Relational analysis result of NS_A1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0025921
time: 1.25 seconds

## Relational analysis of NS_A1_A1_A2_A2_B2

### Relational analysis result of NS_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0025921
time: 1.69 seconds

## BFS NS instance: NS_A1_A2_A1_A1

### Backsubstitution after applying NS history:
0: 0.9889771, 0.9957175, 0.9890051, 0.9963866, -0.0058151, 0.0051664
1: -0.0040106, -0.0023311, -0.0040036, -0.0021643, -0.0014490, 0.0012873
2: 0.0022994, 0.0111999, 0.0014156, 0.0111631, -0.0068222, 0.0076788
3: -0.0063708, -0.0023197, -0.0063541, -0.0019175, -0.0034951, 0.0031052
4: 0.0009729, 0.0026956, 0.0008019, 0.0026885, -0.0013204, 0.0014862
5: 0.0018514, 0.0130459, 0.0007399, 0.0129996, -0.0085806, 0.0096579
6: -0.0017704, 0.0010709, -0.0017586, 0.0013530, -0.0024513, 0.0021778
7: -0.0077181, -0.0003668, -0.0076877, 0.0003631, -0.0063422, 0.0056347
8: -0.0036230, 0.0002429, -0.0036070, 0.0006268, -0.0033353, 0.0029633
9: -0.0021455, 0.0023372, -0.0025906, 0.0023187, -0.0034360, 0.0038675

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_A2_A1_A1_B1

### Relational analysis result of NS_A1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026564, upper bound: 0.0025882
time: 1.20 seconds

## Relational analysis of NS_A1_A2_A1_A1_B2

### Relational analysis result of NS_A1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026564, upper bound: 0.0025882
time: 1.79 seconds

## BFS NS instance: NS_A1_A2_A1_A2

### Backsubstitution after applying NS history:
0: 0.9883012, 0.9954404, 0.9890177, 0.9961994, -0.0064880, 0.0051949
1: -0.0041790, -0.0024001, -0.0040004, -0.0022110, -0.0016166, 0.0012944
2: 0.0026652, 0.0120923, 0.0016629, 0.0111463, -0.0068598, 0.0085674
3: -0.0067770, -0.0024862, -0.0063464, -0.0020300, -0.0038995, 0.0031223
4: 0.0010437, 0.0028683, 0.0008497, 0.0026852, -0.0013277, 0.0016582
5: 0.0023116, 0.0141684, 0.0010510, 0.0129785, -0.0086278, 0.0107755
6: -0.0020553, 0.0009541, -0.0017533, 0.0012741, -0.0027349, 0.0021898
7: -0.0084552, -0.0006690, -0.0076738, 0.0001588, -0.0070761, 0.0056658
8: -0.0040107, 0.0000840, -0.0035997, 0.0005194, -0.0037213, 0.0029796
9: -0.0019613, 0.0027867, -0.0024661, 0.0023102, -0.0034549, 0.0043150

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_A2_A1_A2_B1

### Relational analysis result of NS_A1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026536, upper bound: 0.0026251
time: 1.22 seconds

## Relational analysis of NS_A1_A2_A1_A2_B2

### Relational analysis result of NS_A1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026536, upper bound: 0.0026251
time: 1.67 seconds

## BFS NS instance: NS_A1_A2_A2_A1

### Backsubstitution after applying NS history:
0: 0.9884502, 0.9956167, 0.9890200, 0.9962678, -0.0063791, 0.0052109
1: -0.0041419, -0.0023562, -0.0039999, -0.0021939, -0.0015895, 0.0012984
2: 0.0024324, 0.0118957, 0.0015726, 0.0111431, -0.0068809, 0.0084236
3: -0.0066875, -0.0023803, -0.0063450, -0.0019889, -0.0038340, 0.0031319
4: 0.0009987, 0.0028303, 0.0008323, 0.0026846, -0.0013318, 0.0016304
5: 0.0020188, 0.0139211, 0.0009374, 0.0129745, -0.0086544, 0.0105947
6: -0.0019925, 0.0010284, -0.0017522, 0.0013029, -0.0026890, 0.0021966
7: -0.0082928, -0.0004767, -0.0076712, 0.0002334, -0.0069574, 0.0056832
8: -0.0039252, 0.0001851, -0.0035984, 0.0005586, -0.0036588, 0.0029887
9: -0.0020785, 0.0026877, -0.0025116, 0.0023086, -0.0034656, 0.0042426

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_A2_A2_A1_B1

### Relational analysis result of NS_A1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026547, upper bound: 0.0026220
time: 1.25 seconds

## Relational analysis of NS_A1_A2_A2_A1_B2

### Relational analysis result of NS_A1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026547, upper bound: 0.0026220
time: 1.72 seconds

## BFS NS instance: NS_A1_A2_A2_A2

### Backsubstitution after applying NS history:
0: 0.9877682, 0.9953254, 0.9890330, 0.9960812, -0.0070270, 0.0052474
1: -0.0043118, -0.0024287, -0.0039966, -0.0022404, -0.0017509, 0.0013075
2: 0.0028169, 0.0127963, 0.0018190, 0.0111260, -0.0069291, 0.0092791
3: -0.0070974, -0.0025553, -0.0063372, -0.0021011, -0.0042234, 0.0031538
4: 0.0010731, 0.0030046, 0.0008800, 0.0026813, -0.0013411, 0.0017959
5: 0.0025024, 0.0150539, 0.0012473, 0.0129530, -0.0087150, 0.0116707
6: -0.0022800, 0.0009057, -0.0017468, 0.0012243, -0.0029621, 0.0022120
7: -0.0090367, -0.0007943, -0.0076571, 0.0000299, -0.0076640, 0.0057230
8: -0.0043164, 0.0000181, -0.0035909, 0.0004516, -0.0040304, 0.0030097
9: -0.0018849, 0.0031413, -0.0023875, 0.0023000, -0.0034899, 0.0046734

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_A2_A2_A2_B1

### Relational analysis result of NS_A1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0026448
time: 1.27 seconds

## Relational analysis of NS_A1_A2_A2_A2_B2

### Relational analysis result of NS_A1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0026448
time: 1.74 seconds

## BFS NS instance: NS_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.9887372, 0.9961742, 0.9890191, 0.9961036, -0.0061164, 0.0055765
1: -0.0040703, -0.0022173, -0.0040001, -0.0022348, -0.0015240, 0.0013895
2: 0.0016963, 0.0115166, 0.0017895, 0.0111445, -0.0073637, 0.0080767
3: -0.0065150, -0.0020452, -0.0063456, -0.0020876, -0.0036761, 0.0033516
4: 0.0008562, 0.0027569, 0.0008742, 0.0026849, -0.0014252, 0.0015632
5: 0.0010929, 0.0134443, 0.0012101, 0.0129763, -0.0092616, 0.0101583
6: -0.0018715, 0.0012634, -0.0017527, 0.0012337, -0.0025783, 0.0023507
7: -0.0079797, 0.0001313, -0.0076724, 0.0000543, -0.0066708, 0.0060819
8: -0.0037606, 0.0005049, -0.0035990, 0.0004644, -0.0035081, 0.0031984
9: -0.0024493, 0.0024967, -0.0024024, 0.0023093, -0.0037087, 0.0040678

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_A1_B1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027233
time: 1.58 seconds

## Relational analysis of NS_A2_A1_B1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027289
time: 1.62 seconds

## BFS NS instance: NS_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.9887499, 0.9960033, 0.9883525, 0.9958968, -0.0061419, 0.0061863
1: -0.0040672, -0.0022598, -0.0041662, -0.0022864, -0.0015304, 0.0015415
2: 0.0019219, 0.0115001, 0.0020625, 0.0120247, -0.0081689, 0.0081103
3: -0.0065074, -0.0021479, -0.0067462, -0.0022119, -0.0036915, 0.0037181
4: 0.0008999, 0.0027537, 0.0009271, 0.0028552, -0.0015811, 0.0015697
5: 0.0013767, 0.0134235, 0.0015536, 0.0140833, -0.0102743, 0.0102006
6: -0.0018662, 0.0011914, -0.0020337, 0.0011465, -0.0025890, 0.0026077
7: -0.0079660, -0.0000551, -0.0083993, -0.0001712, -0.0066986, 0.0067470
8: -0.0037534, 0.0004069, -0.0039813, 0.0003458, -0.0035227, 0.0035482
9: -0.0023357, 0.0024884, -0.0022648, 0.0027526, -0.0041143, 0.0040848

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_A1_B1_B2_B1

### Relational analysis result of NS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027193
time: 1.53 seconds

## Relational analysis of NS_A2_A1_B1_B2_B2

### Relational analysis result of NS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027263
time: 1.75 seconds

## BFS NS instance: NS_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.9887514, 0.9960642, 0.9884772, 0.9960240, -0.0061696, 0.0061204
1: -0.0040668, -0.0022447, -0.0041351, -0.0022547, -0.0015373, 0.0015250
2: 0.0018415, 0.0114979, 0.0018946, 0.0118600, -0.0080819, 0.0081469
3: -0.0065065, -0.0021113, -0.0066713, -0.0021354, -0.0037081, 0.0036785
4: 0.0008843, 0.0027533, 0.0008946, 0.0028234, -0.0015642, 0.0015768
5: 0.0012756, 0.0134208, 0.0013423, 0.0138762, -0.0101649, 0.0102467
6: -0.0018655, 0.0012171, -0.0019811, 0.0012001, -0.0026007, 0.0025800
7: -0.0079643, 0.0000113, -0.0082633, -0.0000325, -0.0067288, 0.0066751
8: -0.0037525, 0.0004418, -0.0039097, 0.0004188, -0.0035386, 0.0035104
9: -0.0023762, 0.0024873, -0.0023494, 0.0026697, -0.0040705, 0.0041032

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_A1_B2_B1_B1

### Relational analysis result of NS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027223
time: 1.67 seconds

## Relational analysis of NS_A2_A1_B2_B1_B2

### Relational analysis result of NS_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027281
time: 1.66 seconds

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.9887642, 0.9958942, 0.9878110, 0.9958028, -0.0061979, 0.0066949
1: -0.0040636, -0.0022870, -0.0043011, -0.0023098, -0.0015444, 0.0016682
2: 0.0020659, 0.0114811, 0.0021867, 0.0127397, -0.0088406, 0.0081843
3: -0.0064988, -0.0022134, -0.0070717, -0.0022684, -0.0037251, 0.0040238
4: 0.0009277, 0.0027500, 0.0009511, 0.0029936, -0.0017111, 0.0015841
5: 0.0015579, 0.0133997, 0.0017097, 0.0149826, -0.0111191, 0.0102937
6: -0.0018601, 0.0011454, -0.0022619, 0.0011069, -0.0026127, 0.0028221
7: -0.0079504, -0.0001740, -0.0089899, -0.0002738, -0.0067597, 0.0073017
8: -0.0037452, 0.0003443, -0.0042918, 0.0002919, -0.0035549, 0.0038399
9: -0.0022631, 0.0024789, -0.0022023, 0.0031127, -0.0044526, 0.0041221

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_A1_B2_B2_B1

### Relational analysis result of NS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0027090
time: 1.55 seconds

## Relational analysis of NS_A2_A1_B2_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0027196
time: 1.63 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.9886214, 0.9962758, 0.9890183, 0.9961410, -0.0061933, 0.0056488
1: -0.0040992, -0.0021919, -0.0040003, -0.0022255, -0.0015432, 0.0014075
2: 0.0015621, 0.0116695, 0.0017402, 0.0111455, -0.0074592, 0.0081781
3: -0.0065846, -0.0019841, -0.0063461, -0.0020652, -0.0037223, 0.0033951
4: 0.0008302, 0.0027865, 0.0008647, 0.0026851, -0.0014437, 0.0015829
5: 0.0009241, 0.0136366, 0.0011482, 0.0129776, -0.0093817, 0.0102860
6: -0.0019203, 0.0013063, -0.0017530, 0.0012494, -0.0026107, 0.0023812
7: -0.0081060, 0.0002421, -0.0076732, 0.0000950, -0.0067546, 0.0061608
8: -0.0038270, 0.0005632, -0.0035994, 0.0004858, -0.0035522, 0.0032399
9: -0.0025169, 0.0025738, -0.0024272, 0.0023099, -0.0037568, 0.0041189

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_A2_B1_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027814
time: 1.59 seconds

## Relational analysis of NS_A2_A2_B1_B1_B2

### Relational analysis result of NS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027880
time: 1.61 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9886341, 0.9961085, 0.9883515, 0.9959383, -0.0062179, 0.0062562
1: -0.0040960, -0.0022336, -0.0041665, -0.0022760, -0.0015493, 0.0015589
2: 0.0017830, 0.0116528, 0.0020078, 0.0120260, -0.0082613, 0.0082107
3: -0.0065770, -0.0020846, -0.0067468, -0.0021870, -0.0037371, 0.0037602
4: 0.0008730, 0.0027833, 0.0009165, 0.0028555, -0.0015990, 0.0015892
5: 0.0012020, 0.0136156, 0.0014847, 0.0140850, -0.0103905, 0.0103269
6: -0.0019150, 0.0012358, -0.0020341, 0.0011640, -0.0026211, 0.0026372
7: -0.0080922, 0.0000597, -0.0084004, -0.0001260, -0.0067815, 0.0068233
8: -0.0038198, 0.0004672, -0.0039819, 0.0003696, -0.0035663, 0.0035883
9: -0.0024056, 0.0025653, -0.0022924, 0.0027533, -0.0041608, 0.0041353

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_A2_B1_B2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027787
time: 1.51 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027867
time: 1.64 seconds

## BFS NS instance: NS_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9886359, 0.9961605, 0.9884764, 0.9960553, -0.0062469, 0.0061915
1: -0.0040956, -0.0022206, -0.0041353, -0.0022469, -0.0015566, 0.0015428
2: 0.0017143, 0.0116504, 0.0018533, 0.0118611, -0.0081758, 0.0082489
3: -0.0065759, -0.0020534, -0.0066718, -0.0021167, -0.0037546, 0.0037213
4: 0.0008597, 0.0027828, 0.0008866, 0.0028236, -0.0015824, 0.0015966
5: 0.0011155, 0.0136126, 0.0012904, 0.0138776, -0.0102830, 0.0103750
6: -0.0019142, 0.0012577, -0.0019815, 0.0012133, -0.0026333, 0.0026099
7: -0.0080902, 0.0001164, -0.0082643, 0.0000016, -0.0068131, 0.0067527
8: -0.0038187, 0.0004971, -0.0039102, 0.0004367, -0.0035829, 0.0035512
9: -0.0024402, 0.0025641, -0.0023702, 0.0026703, -0.0041178, 0.0041546

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_A2_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027797
time: 1.56 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027871
time: 1.66 seconds

## BFS NS instance: NS_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9886487, 0.9959905, 0.9878101, 0.9958359, -0.0062752, 0.0067645
1: -0.0040924, -0.0022630, -0.0043014, -0.0023015, -0.0015636, 0.0016855
2: 0.0019388, 0.0116336, 0.0021429, 0.0127409, -0.0089324, 0.0082863
3: -0.0065682, -0.0021556, -0.0070722, -0.0022485, -0.0037716, 0.0040656
4: 0.0009031, 0.0027795, 0.0009426, 0.0029939, -0.0017289, 0.0016038
5: 0.0013979, 0.0135914, 0.0016547, 0.0149842, -0.0112346, 0.0104220
6: -0.0019088, 0.0011860, -0.0022623, 0.0011209, -0.0026452, 0.0028515
7: -0.0080763, -0.0000690, -0.0089909, -0.0002376, -0.0068440, 0.0073776
8: -0.0038114, 0.0003996, -0.0042924, 0.0003109, -0.0035992, 0.0038798
9: -0.0023272, 0.0025557, -0.0022243, 0.0031134, -0.0044988, 0.0041734

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_A2_B2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0027666
time: 1.27 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0027787
time: 1.22 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.04 seconds
NS_A1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026564, upper bound: 0.0025389
NS_A1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026564, upper bound: 0.0025389
NS_A1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026536, upper bound: 0.0025667
NS_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026536, upper bound: 0.0025667
NS_A1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026547, upper bound: 0.0025747
NS_A1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026547, upper bound: 0.0025747
NS_A1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0025921
NS_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0025921
NS_A1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026564, upper bound: 0.0025882
NS_A1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026564, upper bound: 0.0025882
NS_A1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026536, upper bound: 0.0026251
NS_A1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026536, upper bound: 0.0026251
NS_A1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026547, upper bound: 0.0026220
NS_A1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026547, upper bound: 0.0026220
NS_A1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0026448
NS_A1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0026448
NS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027233
NS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027289
NS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027193
NS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027263
NS_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027223
NS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027281
NS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0027090
NS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0027196
NS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027814
NS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027880
NS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027787
NS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027867
NS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027797
NS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027871
NS_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0027666
NS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.0026448, upper bound: 0.0027787

## BFS NS instance: NS_A1_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9890988, 0.9956085, 0.9890664, 0.9960821, -0.0054053, 0.0050261
1: -0.0039802, -0.0023582, -0.0039883, -0.0022402, -0.0013468, 0.0012524
2: 0.0024432, 0.0110391, 0.0018179, 0.0110820, -0.0066369, 0.0071376
3: -0.0062976, -0.0023852, -0.0063172, -0.0021006, -0.0032487, 0.0030208
4: 0.0010008, 0.0026645, 0.0008797, 0.0026728, -0.0012846, 0.0013815
5: 0.0020324, 0.0128438, 0.0012459, 0.0128977, -0.0083475, 0.0089772
6: -0.0017191, 0.0010250, -0.0017327, 0.0012246, -0.0022785, 0.0021187
7: -0.0075853, -0.0004857, -0.0076208, 0.0000308, -0.0058952, 0.0054817
8: -0.0035532, 0.0001804, -0.0035718, 0.0004521, -0.0031002, 0.0028828
9: -0.0020731, 0.0022563, -0.0023880, 0.0022779, -0.0033427, 0.0035949

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A1_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025389
time: 1.18 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025389
time: 1.67 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9890988, 0.9956085, 0.9887292, 0.9963228, -0.0057702, 0.0054804
1: -0.0039802, -0.0023582, -0.0040723, -0.0021802, -0.0014378, 0.0013656
2: 0.0024432, 0.0110391, 0.0015000, 0.0115272, -0.0072368, 0.0076195
3: -0.0062976, -0.0023852, -0.0065198, -0.0019559, -0.0034681, 0.0032939
4: 0.0010008, 0.0026645, 0.0008182, 0.0027589, -0.0014007, 0.0014747
5: 0.0020324, 0.0128438, 0.0008461, 0.0134576, -0.0091020, 0.0095833
6: -0.0017191, 0.0010250, -0.0018749, 0.0013261, -0.0024323, 0.0023102
7: -0.0075853, -0.0004857, -0.0079884, 0.0002934, -0.0062932, 0.0059771
8: -0.0035532, 0.0001804, -0.0037652, 0.0005901, -0.0033095, 0.0031433
9: -0.0020731, 0.0022563, -0.0025481, 0.0025021, -0.0036448, 0.0038376

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A1_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025389
time: 1.68 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025389
time: 1.65 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9884241, 0.9953319, 0.9890792, 0.9958846, -0.0060632, 0.0050030
1: -0.0041484, -0.0024271, -0.0039851, -0.0022894, -0.0015108, 0.0012466
2: 0.0028085, 0.0119302, 0.0020786, 0.0110651, -0.0066064, 0.0080064
3: -0.0067032, -0.0025514, -0.0063095, -0.0022192, -0.0036442, 0.0030069
4: 0.0010715, 0.0028369, 0.0009302, 0.0026695, -0.0012786, 0.0015496
5: 0.0024918, 0.0139644, 0.0015738, 0.0128764, -0.0083091, 0.0100699
6: -0.0020035, 0.0009084, -0.0017273, 0.0011414, -0.0025559, 0.0021089
7: -0.0083213, -0.0007874, -0.0076068, -0.0001845, -0.0066128, 0.0054564
8: -0.0039402, 0.0000218, -0.0035645, 0.0003388, -0.0034776, 0.0028695
9: -0.0018891, 0.0027050, -0.0022567, 0.0022693, -0.0033273, 0.0040324

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A1_A2_B1_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0025667
time: 1.24 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_B2

### Relational analysis result of NS_A1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0025667
time: 1.23 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9884241, 0.9953319, 0.9887418, 0.9961532, -0.0064237, 0.0055032
1: -0.0041484, -0.0024271, -0.0040692, -0.0022225, -0.0016006, 0.0013712
2: 0.0028085, 0.0119302, 0.0017239, 0.0115106, -0.0072669, 0.0084824
3: -0.0067032, -0.0025514, -0.0065122, -0.0020578, -0.0038608, 0.0033076
4: 0.0010715, 0.0028369, 0.0008615, 0.0027557, -0.0014065, 0.0016418
5: 0.0024918, 0.0139644, 0.0011277, 0.0134367, -0.0091398, 0.0106687
6: -0.0020035, 0.0009084, -0.0018696, 0.0012546, -0.0027078, 0.0023198
7: -0.0083213, -0.0007874, -0.0079747, 0.0001085, -0.0070060, 0.0060020
8: -0.0039402, 0.0000218, -0.0037580, 0.0004929, -0.0036844, 0.0031564
9: -0.0018891, 0.0027050, -0.0024354, 0.0024937, -0.0036600, 0.0042722

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A1_A2_B2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0025667
time: 1.67 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0025667
time: 1.68 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9885501, 0.9955158, 0.9890816, 0.9959620, -0.0059431, 0.0050733
1: -0.0041170, -0.0023813, -0.0039845, -0.0022701, -0.0014809, 0.0012641
2: 0.0025657, 0.0117637, 0.0019765, 0.0110620, -0.0066993, 0.0078477
3: -0.0066275, -0.0024409, -0.0063080, -0.0021727, -0.0035720, 0.0030492
4: 0.0010245, 0.0028047, 0.0009104, 0.0026689, -0.0012966, 0.0015189
5: 0.0021864, 0.0137551, 0.0014453, 0.0128725, -0.0084259, 0.0098704
6: -0.0019504, 0.0009859, -0.0017263, 0.0011740, -0.0025052, 0.0021386
7: -0.0081838, -0.0005868, -0.0076042, -0.0001001, -0.0064818, 0.0055332
8: -0.0038679, 0.0001272, -0.0035631, 0.0003832, -0.0034087, 0.0029099
9: -0.0020114, 0.0026212, -0.0023082, 0.0022678, -0.0033741, 0.0039525

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A1_A2_A1_B1_B1

### Relational analysis result of NS_A1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025747
time: 1.34 seconds

## Relational analysis of NS_A1_A1_A2_A1_B1_B2

### Relational analysis result of NS_A1_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025747
time: 1.95 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9885501, 0.9955158, 0.9887434, 0.9962119, -0.0063113, 0.0055257
1: -0.0041170, -0.0023813, -0.0040688, -0.0022078, -0.0015726, 0.0013769
2: 0.0025657, 0.0117637, 0.0016464, 0.0115084, -0.0072966, 0.0083340
3: -0.0066275, -0.0024409, -0.0065112, -0.0020225, -0.0037933, 0.0033211
4: 0.0010245, 0.0028047, 0.0008465, 0.0027553, -0.0014122, 0.0016130
5: 0.0021864, 0.0137551, 0.0010302, 0.0134340, -0.0091772, 0.0104820
6: -0.0019504, 0.0009859, -0.0018689, 0.0012794, -0.0026604, 0.0023293
7: -0.0081838, -0.0005868, -0.0079729, 0.0001725, -0.0068834, 0.0060265
8: -0.0038679, 0.0001272, -0.0037570, 0.0005266, -0.0036199, 0.0031693
9: -0.0020114, 0.0026212, -0.0024744, 0.0024926, -0.0036749, 0.0041975

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A1_A2_A1_B2_B1

### Relational analysis result of NS_A1_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025747
time: 1.26 seconds

## Relational analysis of NS_A1_A1_A2_A1_B2_B2

### Relational analysis result of NS_A1_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025747
time: 1.40 seconds

## BFS NS instance: NS_A1_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9878775, 0.9952232, 0.9890898, 0.9957614, -0.0065809, 0.0050894
1: -0.0042845, -0.0024542, -0.0039825, -0.0023201, -0.0016398, 0.0012682
2: 0.0029521, 0.0126518, 0.0022413, 0.0110512, -0.0067205, 0.0086900
3: -0.0070317, -0.0026168, -0.0063031, -0.0022933, -0.0039553, 0.0030589
4: 0.0010993, 0.0029766, 0.0009617, 0.0026668, -0.0013007, 0.0016819
5: 0.0026724, 0.0148721, 0.0017784, 0.0128589, -0.0084527, 0.0109297
6: -0.0022339, 0.0008625, -0.0017229, 0.0010894, -0.0027741, 0.0021454
7: -0.0089173, -0.0009060, -0.0075953, -0.0003189, -0.0071774, 0.0055507
8: -0.0042537, -0.0000406, -0.0035584, 0.0002681, -0.0037745, 0.0029191
9: -0.0018168, 0.0030685, -0.0021748, 0.0022623, -0.0033848, 0.0043767

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A1_A2_A2_B1_B1

### Relational analysis result of NS_A1_A1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0025921
time: 1.25 seconds

## Relational analysis of NS_A1_A1_A2_A2_B1_B2

### Relational analysis result of NS_A1_A1_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0025921
time: 1.83 seconds

## BFS NS instance: NS_A1_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9878775, 0.9952232, 0.9887562, 0.9960411, -0.0069495, 0.0055529
1: -0.0042845, -0.0024542, -0.0040656, -0.0022504, -0.0017316, 0.0013836
2: 0.0029521, 0.0126518, 0.0018720, 0.0114916, -0.0073326, 0.0091767
3: -0.0070317, -0.0026168, -0.0065036, -0.0021252, -0.0041768, 0.0033375
4: 0.0010993, 0.0029766, 0.0008902, 0.0027521, -0.0014192, 0.0017761
5: 0.0026724, 0.0148721, 0.0013139, 0.0134128, -0.0092225, 0.0115419
6: -0.0022339, 0.0008625, -0.0018635, 0.0012073, -0.0029295, 0.0023408
7: -0.0089173, -0.0009060, -0.0079590, -0.0000138, -0.0075794, 0.0060563
8: -0.0042537, -0.0000406, -0.0037497, 0.0004286, -0.0039859, 0.0031849
9: -0.0018168, 0.0030685, -0.0023608, 0.0024841, -0.0036931, 0.0046219

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A1_A2_A2_B2_B1

### Relational analysis result of NS_A1_A1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0025921
time: 1.70 seconds

## Relational analysis of NS_A1_A1_A2_A2_B2_B2

### Relational analysis result of NS_A1_A1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0025921
time: 1.84 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9889771, 0.9957175, 0.9890658, 0.9961199, -0.0054745, 0.0051039
1: -0.0040106, -0.0023311, -0.0039885, -0.0022308, -0.0013641, 0.0012718
2: 0.0022994, 0.0111999, 0.0017681, 0.0110829, -0.0067397, 0.0072291
3: -0.0063708, -0.0023197, -0.0063176, -0.0020779, -0.0032904, 0.0030676
4: 0.0009729, 0.0026956, 0.0008701, 0.0026730, -0.0013045, 0.0013992
5: 0.0018514, 0.0130459, 0.0011832, 0.0128988, -0.0084767, 0.0090922
6: -0.0017704, 0.0010709, -0.0017330, 0.0012405, -0.0023077, 0.0021515
7: -0.0077181, -0.0003668, -0.0076215, 0.0000720, -0.0059707, 0.0055666
8: -0.0036230, 0.0002429, -0.0035722, 0.0004737, -0.0031400, 0.0029274
9: -0.0021455, 0.0023372, -0.0024131, 0.0022783, -0.0033945, 0.0036409

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A2_A1_A1_B1_B1

### Relational analysis result of NS_A1_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025882
time: 1.28 seconds

## Relational analysis of NS_A1_A2_A1_A1_B1_B2

### Relational analysis result of NS_A1_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025882
time: 1.79 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9889771, 0.9957175, 0.9887283, 0.9963576, -0.0058386, 0.0055572
1: -0.0040106, -0.0023311, -0.0040726, -0.0021716, -0.0014548, 0.0013847
2: 0.0022994, 0.0111999, 0.0014541, 0.0115284, -0.0073382, 0.0077097
3: -0.0063708, -0.0023197, -0.0065204, -0.0019350, -0.0035091, 0.0033400
4: 0.0009729, 0.0026956, 0.0008093, 0.0027592, -0.0014203, 0.0014922
5: 0.0018514, 0.0130459, 0.0007884, 0.0134592, -0.0092296, 0.0096968
6: -0.0017704, 0.0010709, -0.0018753, 0.0013407, -0.0024612, 0.0023426
7: -0.0077181, -0.0003668, -0.0079895, 0.0003313, -0.0063678, 0.0060609
8: -0.0036230, 0.0002429, -0.0037657, 0.0006101, -0.0033487, 0.0031874
9: -0.0021455, 0.0023372, -0.0025712, 0.0025027, -0.0036959, 0.0038830

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A2_A1_A1_B2_B1

### Relational analysis result of NS_A1_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025882
time: 1.19 seconds

## Relational analysis of NS_A1_A2_A1_A1_B2_B2

### Relational analysis result of NS_A1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025882
time: 1.65 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9883012, 0.9954404, 0.9890784, 0.9959230, -0.0061416, 0.0050845
1: -0.0041790, -0.0024001, -0.0039853, -0.0022798, -0.0015303, 0.0012669
2: 0.0026652, 0.0120923, 0.0020279, 0.0110660, -0.0067140, 0.0081099
3: -0.0067770, -0.0024862, -0.0063099, -0.0021961, -0.0036913, 0.0030559
4: 0.0010437, 0.0028683, 0.0009204, 0.0026697, -0.0012995, 0.0015697
5: 0.0023116, 0.0141684, 0.0015100, 0.0128776, -0.0084445, 0.0102001
6: -0.0020553, 0.0009541, -0.0017276, 0.0011576, -0.0025889, 0.0021433
7: -0.0084552, -0.0006690, -0.0076076, -0.0001426, -0.0066983, 0.0055454
8: -0.0040107, 0.0000840, -0.0035649, 0.0003608, -0.0035226, 0.0029163
9: -0.0019613, 0.0027867, -0.0022823, 0.0022698, -0.0033815, 0.0040846

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0026251
time: 1.23 seconds

## Relational analysis of NS_A1_A2_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0026251
time: 1.76 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9883012, 0.9954404, 0.9887408, 0.9961904, -0.0065035, 0.0055862
1: -0.0041790, -0.0024001, -0.0040695, -0.0022132, -0.0016205, 0.0013919
2: 0.0026652, 0.0120923, 0.0016748, 0.0115120, -0.0073765, 0.0085878
3: -0.0067770, -0.0024862, -0.0065129, -0.0020354, -0.0039088, 0.0033575
4: 0.0010437, 0.0028683, 0.0008520, 0.0027560, -0.0014277, 0.0016621
5: 0.0023116, 0.0141684, 0.0010659, 0.0134385, -0.0092777, 0.0108011
6: -0.0020553, 0.0009541, -0.0018700, 0.0012703, -0.0027414, 0.0023548
7: -0.0084552, -0.0006690, -0.0079759, 0.0001490, -0.0070929, 0.0060925
8: -0.0040107, 0.0000840, -0.0037586, 0.0005142, -0.0037301, 0.0032040
9: -0.0019613, 0.0027867, -0.0024601, 0.0024944, -0.0037152, 0.0043252

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0026251
time: 1.49 seconds

## Relational analysis of NS_A1_A2_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0026251
time: 1.65 seconds

## BFS NS instance: NS_A1_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9884502, 0.9956167, 0.9890809, 0.9960003, -0.0060317, 0.0051480
1: -0.0041419, -0.0023562, -0.0039847, -0.0022606, -0.0015029, 0.0012827
2: 0.0024324, 0.0118957, 0.0019259, 0.0110628, -0.0067979, 0.0079648
3: -0.0066875, -0.0023803, -0.0063084, -0.0021497, -0.0036252, 0.0030941
4: 0.0009987, 0.0028303, 0.0009006, 0.0026691, -0.0013157, 0.0015416
5: 0.0020188, 0.0139211, 0.0013817, 0.0128736, -0.0085499, 0.0100176
6: -0.0019925, 0.0010284, -0.0017266, 0.0011901, -0.0025426, 0.0021701
7: -0.0082928, -0.0004767, -0.0076049, -0.0000584, -0.0065784, 0.0056146
8: -0.0039252, 0.0001851, -0.0035635, 0.0004052, -0.0034595, 0.0029527
9: -0.0020785, 0.0026877, -0.0023336, 0.0022682, -0.0034238, 0.0040115

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A2_A2_A1_B1_B1

### Relational analysis result of NS_A1_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026514, upper bound: 0.0026220
time: 1.34 seconds

## Relational analysis of NS_A1_A2_A2_A1_B1_B2

### Relational analysis result of NS_A1_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026514, upper bound: 0.0026220
time: 1.89 seconds

## BFS NS instance: NS_A1_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9884502, 0.9956167, 0.9887426, 0.9962443, -0.0063983, 0.0055998
1: -0.0041419, -0.0023562, -0.0040690, -0.0021998, -0.0015943, 0.0013953
2: 0.0024324, 0.0118957, 0.0016038, 0.0115096, -0.0073944, 0.0084489
3: -0.0066875, -0.0023803, -0.0065118, -0.0020031, -0.0038456, 0.0033656
4: 0.0009987, 0.0028303, 0.0008383, 0.0027556, -0.0014312, 0.0016353
5: 0.0020188, 0.0139211, 0.0009766, 0.0134356, -0.0093002, 0.0106265
6: -0.0019925, 0.0010284, -0.0018693, 0.0012930, -0.0026971, 0.0023605
7: -0.0082928, -0.0004767, -0.0079740, 0.0002077, -0.0069782, 0.0061073
8: -0.0039252, 0.0001851, -0.0037576, 0.0005451, -0.0036698, 0.0032118
9: -0.0020785, 0.0026877, -0.0024959, 0.0024932, -0.0037242, 0.0042553

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A2_A2_A1_B2_B1

### Relational analysis result of NS_A1_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026514, upper bound: 0.0026220
time: 1.30 seconds

## Relational analysis of NS_A1_A2_A2_A1_B2_B2

### Relational analysis result of NS_A1_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026514, upper bound: 0.0026220
time: 1.78 seconds

## BFS NS instance: NS_A1_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9877682, 0.9953254, 0.9890890, 0.9958006, -0.0066708, 0.0051698
1: -0.0043118, -0.0024287, -0.0039827, -0.0023103, -0.0016622, 0.0012882
2: 0.0028169, 0.0127963, 0.0021896, 0.0110521, -0.0068267, 0.0088087
3: -0.0070974, -0.0025553, -0.0063036, -0.0022697, -0.0040094, 0.0031072
4: 0.0010731, 0.0030046, 0.0009517, 0.0026670, -0.0013213, 0.0017049
5: 0.0025024, 0.0150539, 0.0017134, 0.0128601, -0.0085862, 0.0110791
6: -0.0022800, 0.0009057, -0.0017232, 0.0011059, -0.0028120, 0.0021793
7: -0.0090367, -0.0007943, -0.0075961, -0.0002762, -0.0072755, 0.0056384
8: -0.0043164, 0.0000181, -0.0035589, 0.0002906, -0.0038261, 0.0029652
9: -0.0018849, 0.0031413, -0.0022008, 0.0022628, -0.0034383, 0.0044366

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A2_A2_A2_B1_B1

### Relational analysis result of NS_A1_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026448
time: 1.25 seconds

## Relational analysis of NS_A1_A2_A2_A2_B1_B2

### Relational analysis result of NS_A1_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026448
time: 1.67 seconds

## BFS NS instance: NS_A1_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9877682, 0.9953254, 0.9887552, 0.9960740, -0.0070415, 0.0056368
1: -0.0043118, -0.0024287, -0.0040659, -0.0022422, -0.0017546, 0.0014045
2: 0.0028169, 0.0127963, 0.0018285, 0.0114929, -0.0074434, 0.0092983
3: -0.0070974, -0.0025553, -0.0065042, -0.0021054, -0.0042322, 0.0033879
4: 0.0010731, 0.0030046, 0.0008818, 0.0027523, -0.0014406, 0.0017997
5: 0.0025024, 0.0150539, 0.0012592, 0.0134145, -0.0093618, 0.0116948
6: -0.0022800, 0.0009057, -0.0018639, 0.0012212, -0.0029683, 0.0023761
7: -0.0090367, -0.0007943, -0.0079601, 0.0000221, -0.0076798, 0.0061478
8: -0.0043164, 0.0000181, -0.0037503, 0.0004475, -0.0040387, 0.0032330
9: -0.0018849, 0.0031413, -0.0023827, 0.0024848, -0.0037489, 0.0046831

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A2_A2_A2_B2_B1

### Relational analysis result of NS_A1_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026448
time: 1.73 seconds

## Relational analysis of NS_A1_A2_A2_A2_B2_B2

### Relational analysis result of NS_A1_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026448
time: 1.83 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: 0.9887372, 0.9961742, 0.9890911, 0.9957576, -0.0055922, 0.0056545
1: -0.0040703, -0.0022173, -0.0039822, -0.0023210, -0.0013934, 0.0014090
2: 0.0016963, 0.0115166, 0.0022463, 0.0110494, -0.0074667, 0.0073845
3: -0.0065150, -0.0020452, -0.0063023, -0.0022956, -0.0033611, 0.0033985
4: 0.0008562, 0.0027569, 0.0009627, 0.0026665, -0.0014452, 0.0014293
5: 0.0010929, 0.0134443, 0.0017847, 0.0128567, -0.0093911, 0.0092877
6: -0.0018715, 0.0012634, -0.0017223, 0.0010878, -0.0023573, 0.0023836
7: -0.0079797, 0.0001313, -0.0075939, -0.0003230, -0.0060991, 0.0061670
8: -0.0037606, 0.0005049, -0.0035577, 0.0002660, -0.0032075, 0.0032432
9: -0.0024493, 0.0024967, -0.0021723, 0.0022615, -0.0037606, 0.0037192

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A1_B1_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027112
time: 1.28 seconds

## Relational analysis of NS_A2_A1_B1_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027233
time: 1.59 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: 0.9887372, 0.9961742, 0.9887607, 0.9959766, -0.0053342, 0.0055489
1: -0.0040703, -0.0022173, -0.0040645, -0.0022665, -0.0013291, 0.0013826
2: 0.0016963, 0.0115166, 0.0019570, 0.0114857, -0.0073272, 0.0070437
3: -0.0065150, -0.0020452, -0.0065009, -0.0021639, -0.0032060, 0.0033350
4: 0.0008562, 0.0027569, 0.0009067, 0.0027509, -0.0014182, 0.0013633
5: 0.0010929, 0.0134443, 0.0014209, 0.0134054, -0.0092157, 0.0088591
6: -0.0018715, 0.0012634, -0.0018616, 0.0011802, -0.0022485, 0.0023390
7: -0.0079797, 0.0001313, -0.0079542, -0.0000841, -0.0058177, 0.0060518
8: -0.0037606, 0.0005049, -0.0037472, 0.0003916, -0.0030595, 0.0031826
9: -0.0024493, 0.0024967, -0.0023180, 0.0024812, -0.0036904, 0.0035476

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A1_B1_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027157
time: 1.66 seconds

## Relational analysis of NS_A2_A1_B1_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027289
time: 1.72 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: 0.9887499, 0.9960033, 0.9884161, 0.9954856, -0.0056134, 0.0063079
1: -0.0040672, -0.0022598, -0.0041503, -0.0023888, -0.0013987, 0.0015718
2: 0.0019219, 0.0115001, 0.0026055, 0.0119407, -0.0083296, 0.0074125
3: -0.0065074, -0.0021479, -0.0067080, -0.0024590, -0.0033738, 0.0037913
4: 0.0008999, 0.0027537, 0.0010322, 0.0028390, -0.0016122, 0.0014347
5: 0.0013767, 0.0134235, 0.0022364, 0.0139777, -0.0104764, 0.0093229
6: -0.0018662, 0.0011914, -0.0020069, 0.0009732, -0.0023663, 0.0026590
7: -0.0079660, -0.0000551, -0.0083300, -0.0006196, -0.0061222, 0.0068797
8: -0.0037534, 0.0004069, -0.0039448, 0.0001100, -0.0032196, 0.0036180
9: -0.0023357, 0.0024884, -0.0019914, 0.0027103, -0.0041952, 0.0037333

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027123
time: 1.61 seconds

## Relational analysis of NS_A2_A1_B1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027193
time: 1.46 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: 0.9887499, 0.9960033, 0.9880242, 0.9957840, -0.0053763, 0.0061458
1: -0.0040672, -0.0022598, -0.0042480, -0.0023145, -0.0013396, 0.0015314
2: 0.0019219, 0.0115001, 0.0022114, 0.0124582, -0.0081155, 0.0070993
3: -0.0065074, -0.0021479, -0.0069436, -0.0022797, -0.0032313, 0.0036938
4: 0.0008999, 0.0027537, 0.0009559, 0.0029391, -0.0015707, 0.0013741
5: 0.0013767, 0.0134235, 0.0017408, 0.0146286, -0.0102071, 0.0089291
6: -0.0018662, 0.0011914, -0.0021721, 0.0010990, -0.0022663, 0.0025907
7: -0.0079660, -0.0000551, -0.0087574, -0.0002942, -0.0058636, 0.0067029
8: -0.0037534, 0.0004069, -0.0041696, 0.0002811, -0.0030836, 0.0035250
9: -0.0023357, 0.0024884, -0.0021898, 0.0029710, -0.0040874, 0.0035756

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027181
time: 1.70 seconds

## Relational analysis of NS_A2_A1_B1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027263
time: 1.73 seconds

## BFS NS instance: NS_A2_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: 0.9887514, 0.9960642, 0.9885427, 0.9956633, -0.0056395, 0.0061948
1: -0.0040668, -0.0022447, -0.0041188, -0.0023445, -0.0014052, 0.0015436
2: 0.0018415, 0.0114979, 0.0023709, 0.0117737, -0.0081802, 0.0074469
3: -0.0065065, -0.0021113, -0.0066320, -0.0023522, -0.0033895, 0.0037233
4: 0.0008843, 0.0027533, 0.0009868, 0.0028067, -0.0015833, 0.0014413
5: 0.0012756, 0.0134208, 0.0019414, 0.0137676, -0.0102885, 0.0093663
6: -0.0018655, 0.0012171, -0.0019535, 0.0010481, -0.0023773, 0.0026113
7: -0.0079643, 0.0000113, -0.0081920, -0.0004259, -0.0061507, 0.0067563
8: -0.0037525, 0.0004418, -0.0038723, 0.0002119, -0.0032346, 0.0035531
9: -0.0023762, 0.0024873, -0.0021095, 0.0026262, -0.0041200, 0.0037507

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A1_B2_B1_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027112
time: 1.19 seconds

## Relational analysis of NS_A2_A1_B2_B1_B1_A2

### Relational analysis result of NS_A2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027223
time: 1.65 seconds

## BFS NS instance: NS_A2_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9887514, 0.9960642, 0.9881929, 0.9958961, -0.0053929, 0.0060890
1: -0.0040668, -0.0022447, -0.0042060, -0.0022865, -0.0013438, 0.0015172
2: 0.0018415, 0.0114979, 0.0020634, 0.0122354, -0.0080404, 0.0071212
3: -0.0065065, -0.0021113, -0.0068422, -0.0022123, -0.0032413, 0.0036596
4: 0.0008843, 0.0027533, 0.0009273, 0.0028960, -0.0015562, 0.0013783
5: 0.0012756, 0.0134208, 0.0015546, 0.0143484, -0.0101127, 0.0089566
6: -0.0018655, 0.0012171, -0.0021009, 0.0011462, -0.0022733, 0.0025667
7: -0.0079643, 0.0000113, -0.0085734, -0.0001719, -0.0058817, 0.0066409
8: -0.0037525, 0.0004418, -0.0040728, 0.0003454, -0.0030931, 0.0034924
9: -0.0023762, 0.0024873, -0.0022644, 0.0028588, -0.0040496, 0.0035866

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A1_B2_B1_B2_A1

### Relational analysis result of NS_A2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027157
time: 1.74 seconds

## Relational analysis of NS_A2_A1_B2_B1_B2_A2

### Relational analysis result of NS_A2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027281
time: 1.74 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9887642, 0.9958942, 0.9878703, 0.9953730, -0.0056663, 0.0068331
1: -0.0040636, -0.0022870, -0.0042863, -0.0024169, -0.0014119, 0.0017026
2: 0.0020659, 0.0114811, 0.0027543, 0.0126613, -0.0090231, 0.0074823
3: -0.0064988, -0.0022134, -0.0070360, -0.0025268, -0.0034056, 0.0041069
4: 0.0009277, 0.0027500, 0.0010610, 0.0029785, -0.0017464, 0.0014482
5: 0.0015579, 0.0133997, 0.0024236, 0.0148841, -0.0113487, 0.0094107
6: -0.0018601, 0.0011454, -0.0022369, 0.0009257, -0.0023885, 0.0028804
7: -0.0079504, -0.0001740, -0.0089252, -0.0007426, -0.0061799, 0.0074525
8: -0.0037452, 0.0003443, -0.0042578, 0.0000453, -0.0032499, 0.0039192
9: -0.0022631, 0.0024789, -0.0019164, 0.0030733, -0.0045445, 0.0037685

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A1_B2_B2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026648
time: 1.70 seconds

## Relational analysis of NS_A2_A1_B2_B2_B1_A2

### Relational analysis result of NS_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027090
time: 1.27 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9887642, 0.9958942, 0.9874700, 0.9956874, -0.0054316, 0.0066482
1: -0.0040636, -0.0022870, -0.0043861, -0.0023385, -0.0013534, 0.0016566
2: 0.0020659, 0.0114811, 0.0023390, 0.0131901, -0.0087789, 0.0071723
3: -0.0064988, -0.0022134, -0.0072767, -0.0023377, -0.0032645, 0.0039958
4: 0.0009277, 0.0027500, 0.0009806, 0.0030808, -0.0016991, 0.0013882
5: 0.0015579, 0.0133997, 0.0019013, 0.0155491, -0.0110415, 0.0090209
6: -0.0018601, 0.0011454, -0.0024057, 0.0010583, -0.0022896, 0.0028025
7: -0.0079504, -0.0001740, -0.0093619, -0.0003996, -0.0059239, 0.0072508
8: -0.0037452, 0.0003443, -0.0044875, 0.0002257, -0.0031153, 0.0038131
9: -0.0022631, 0.0024789, -0.0021256, 0.0033396, -0.0044215, 0.0036124

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A1_B2_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026719
time: 1.91 seconds

## Relational analysis of NS_A2_A1_B2_B2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027196
time: 1.80 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: 0.9886214, 0.9962758, 0.9890904, 0.9957986, -0.0056684, 0.0057296
1: -0.0040992, -0.0021919, -0.0039823, -0.0023109, -0.0014124, 0.0014277
2: 0.0015621, 0.0116695, 0.0021923, 0.0110503, -0.0075659, 0.0074851
3: -0.0065846, -0.0019841, -0.0063027, -0.0022710, -0.0034069, 0.0034437
4: 0.0008302, 0.0027865, 0.0009522, 0.0026666, -0.0014644, 0.0014487
5: 0.0009241, 0.0136366, 0.0017168, 0.0128578, -0.0095159, 0.0094143
6: -0.0019203, 0.0013063, -0.0017226, 0.0011051, -0.0023894, 0.0024152
7: -0.0081060, 0.0002421, -0.0075945, -0.0002784, -0.0061822, 0.0062490
8: -0.0038270, 0.0005632, -0.0035580, 0.0002894, -0.0032512, 0.0032863
9: -0.0025169, 0.0025738, -0.0021995, 0.0022619, -0.0038106, 0.0037699

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027652
time: 1.60 seconds

## Relational analysis of NS_A2_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027814
time: 1.25 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: 0.9886214, 0.9962758, 0.9887598, 0.9960143, -0.0054017, 0.0056213
1: -0.0040992, -0.0021919, -0.0040647, -0.0022571, -0.0013459, 0.0014007
2: 0.0015621, 0.0116695, 0.0019074, 0.0114869, -0.0074229, 0.0071328
3: -0.0065846, -0.0019841, -0.0065014, -0.0021413, -0.0032466, 0.0033786
4: 0.0008302, 0.0027865, 0.0008971, 0.0027511, -0.0014367, 0.0013805
5: 0.0009241, 0.0136366, 0.0013585, 0.0134069, -0.0093361, 0.0089712
6: -0.0019203, 0.0013063, -0.0018620, 0.0011960, -0.0022770, 0.0023696
7: -0.0081060, 0.0002421, -0.0079551, -0.0000431, -0.0058913, 0.0061309
8: -0.0038270, 0.0005632, -0.0037477, 0.0004132, -0.0030982, 0.0032242
9: -0.0025169, 0.0025738, -0.0023430, 0.0024818, -0.0037386, 0.0035925

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027692
time: 1.65 seconds

## Relational analysis of NS_A2_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027880
time: 1.70 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: 0.9886341, 0.9961085, 0.9884154, 0.9955269, -0.0056909, 0.0063833
1: -0.0040960, -0.0022336, -0.0041506, -0.0023786, -0.0014180, 0.0015906
2: 0.0017830, 0.0116528, 0.0025511, 0.0119418, -0.0084291, 0.0075147
3: -0.0065770, -0.0020846, -0.0067085, -0.0024343, -0.0034204, 0.0038366
4: 0.0008730, 0.0027833, 0.0010216, 0.0028392, -0.0016314, 0.0014545
5: 0.0012020, 0.0136156, 0.0021680, 0.0139791, -0.0106016, 0.0094516
6: -0.0019150, 0.0012358, -0.0020072, 0.0009906, -0.0023989, 0.0026908
7: -0.0080922, 0.0000597, -0.0083309, -0.0005747, -0.0062067, 0.0069619
8: -0.0038198, 0.0004672, -0.0039453, 0.0001336, -0.0032640, 0.0036612
9: -0.0024056, 0.0025653, -0.0020188, 0.0027109, -0.0042454, 0.0037848

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027732
time: 1.24 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027787
time: 1.23 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: 0.9886341, 0.9961085, 0.9880232, 0.9958220, -0.0054449, 0.0062156
1: -0.0040960, -0.0022336, -0.0042483, -0.0023050, -0.0013567, 0.0015488
2: 0.0017830, 0.0116528, 0.0021614, 0.0124596, -0.0082077, 0.0071899
3: -0.0065770, -0.0020846, -0.0069442, -0.0022569, -0.0032725, 0.0037358
4: 0.0008730, 0.0027833, 0.0009462, 0.0029394, -0.0015886, 0.0013916
5: 0.0012020, 0.0136156, 0.0016779, 0.0146303, -0.0103231, 0.0090430
6: -0.0019150, 0.0012358, -0.0021725, 0.0011150, -0.0022952, 0.0026201
7: -0.0080922, 0.0000597, -0.0087585, -0.0002529, -0.0059384, 0.0067790
8: -0.0038198, 0.0004672, -0.0041702, 0.0003029, -0.0031230, 0.0035650
9: -0.0024056, 0.0025653, -0.0022151, 0.0029717, -0.0041338, 0.0036212

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027793
time: 1.87 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027867
time: 1.77 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: 0.9886359, 0.9961605, 0.9885420, 0.9956973, -0.0057193, 0.0062692
1: -0.0040956, -0.0022206, -0.0041190, -0.0023361, -0.0014251, 0.0015621
2: 0.0017143, 0.0116504, 0.0023260, 0.0117746, -0.0082784, 0.0075523
3: -0.0065759, -0.0020534, -0.0066324, -0.0023318, -0.0034375, 0.0037679
4: 0.0008597, 0.0027828, 0.0009781, 0.0028068, -0.0016023, 0.0014617
5: 0.0011155, 0.0136126, 0.0018850, 0.0137687, -0.0104120, 0.0094988
6: -0.0019142, 0.0012577, -0.0019538, 0.0010624, -0.0024109, 0.0026427
7: -0.0080902, 0.0001164, -0.0081928, -0.0003889, -0.0062378, 0.0068374
8: -0.0038187, 0.0004971, -0.0038726, 0.0002314, -0.0032804, 0.0035957
9: -0.0024402, 0.0025641, -0.0021321, 0.0026267, -0.0041694, 0.0038038

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027652
time: 1.56 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027797
time: 1.53 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9886359, 0.9961605, 0.9881920, 0.9959261, -0.0054695, 0.0061604
1: -0.0040956, -0.0022206, -0.0042062, -0.0022791, -0.0013628, 0.0015350
2: 0.0017143, 0.0116504, 0.0020238, 0.0122367, -0.0081347, 0.0072224
3: -0.0065759, -0.0020534, -0.0068427, -0.0021943, -0.0032873, 0.0037026
4: 0.0008597, 0.0027828, 0.0009196, 0.0028963, -0.0015745, 0.0013979
5: 0.0011155, 0.0136126, 0.0015049, 0.0143500, -0.0102314, 0.0090839
6: -0.0019142, 0.0012577, -0.0021014, 0.0011589, -0.0023056, 0.0025968
7: -0.0080902, 0.0001164, -0.0085745, -0.0001393, -0.0059652, 0.0067188
8: -0.0038187, 0.0004971, -0.0040734, 0.0003626, -0.0031371, 0.0035333
9: -0.0024402, 0.0025641, -0.0022843, 0.0028594, -0.0040971, 0.0036376

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027692
time: 1.65 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027871
time: 1.75 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9886487, 0.9959905, 0.9878692, 0.9954093, -0.0057463, 0.0069080
1: -0.0040924, -0.0022630, -0.0042866, -0.0024078, -0.0014318, 0.0017213
2: 0.0019388, 0.0116336, 0.0027061, 0.0126629, -0.0091220, 0.0075879
3: -0.0065682, -0.0021556, -0.0070367, -0.0025048, -0.0034537, 0.0041519
4: 0.0009031, 0.0027795, 0.0010517, 0.0029788, -0.0017655, 0.0014686
5: 0.0013979, 0.0135914, 0.0023630, 0.0148861, -0.0114730, 0.0095436
6: -0.0019088, 0.0011860, -0.0022374, 0.0009411, -0.0024223, 0.0029120
7: -0.0080763, -0.0000690, -0.0089265, -0.0007028, -0.0062671, 0.0075342
8: -0.0038114, 0.0003996, -0.0042585, 0.0000663, -0.0032958, 0.0039622
9: -0.0023272, 0.0025557, -0.0019407, 0.0030741, -0.0045943, 0.0038217

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A2_B2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027107
time: 1.75 seconds

## Relational analysis of NS_A2_A2_B2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027666
time: 1.29 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9886487, 0.9959905, 0.9874691, 0.9957181, -0.0055067, 0.0067177
1: -0.0040924, -0.0022630, -0.0043863, -0.0023309, -0.0013721, 0.0016739
2: 0.0019388, 0.0116336, 0.0022986, 0.0131913, -0.0088707, 0.0072715
3: -0.0065682, -0.0021556, -0.0072772, -0.0023193, -0.0033097, 0.0040375
4: 0.0009031, 0.0027795, 0.0009728, 0.0030810, -0.0017169, 0.0014074
5: 0.0013979, 0.0135914, 0.0018504, 0.0155507, -0.0111570, 0.0091456
6: -0.0019088, 0.0011860, -0.0024061, 0.0010712, -0.0023213, 0.0028318
7: -0.0080763, -0.0000690, -0.0093629, -0.0003662, -0.0060058, 0.0073266
8: -0.0038114, 0.0003996, -0.0044880, 0.0002433, -0.0031584, 0.0038530
9: -0.0023272, 0.0025557, -0.0021459, 0.0033402, -0.0044677, 0.0036623

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A2_B2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027174
time: 1.23 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027787
time: 1.96 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.73 seconds
NS_A1_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025389
NS_A1_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025389
NS_A1_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025389
NS_A1_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025389
NS_A1_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0025667
NS_A1_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0025667
NS_A1_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0025667
NS_A1_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0025667
NS_A1_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025747
NS_A1_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025747
NS_A1_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025747
NS_A1_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025747
NS_A1_A1_A2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0025921
NS_A1_A1_A2_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0025921
NS_A1_A1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0025921
NS_A1_A1_A2_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0025921
NS_A1_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025882
NS_A1_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025882
NS_A1_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025882
NS_A1_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026513, upper bound: 0.0025882
NS_A1_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0026251
NS_A1_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0026251
NS_A1_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0026251
NS_A1_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026506, upper bound: 0.0026251
NS_A1_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026514, upper bound: 0.0026220
NS_A1_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026514, upper bound: 0.0026220
NS_A1_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026514, upper bound: 0.0026220
NS_A1_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026514, upper bound: 0.0026220
NS_A1_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026448
NS_A1_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026448
NS_A1_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026448
NS_A1_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026448
NS_A2_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027112
NS_A2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027233
NS_A2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027157
NS_A2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027289
NS_A2_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027123
NS_A2_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027193
NS_A2_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027181
NS_A2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027263
NS_A2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027112
NS_A2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027223
NS_A2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027157
NS_A2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027281
NS_A2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026648
NS_A2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027090
NS_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0026719
NS_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027196
NS_A2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027652
NS_A2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027814
NS_A2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027692
NS_A2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0027880
NS_A2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027732
NS_A2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027787
NS_A2_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027793
NS_A2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026251, upper bound: 0.0027867
NS_A2_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027652
NS_A2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027797
NS_A2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027692
NS_A2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027871
NS_A2_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027107
NS_A2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027666
NS_A2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027174
NS_A2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -0.0026220, upper bound: 0.0027787

## BFS NS instance: NS_A1_A1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.9890988, 0.9956085, 0.9890766, 0.9959178, -0.0052235, 0.0050169
1: -0.0039802, -0.0023582, -0.0039858, -0.0022811, -0.0013015, 0.0012501
2: 0.0024432, 0.0110391, 0.0020348, 0.0110686, -0.0066247, 0.0068975
3: -0.0062976, -0.0023852, -0.0063111, -0.0021993, -0.0031394, 0.0030153
4: 0.0010008, 0.0026645, 0.0009217, 0.0026702, -0.0012822, 0.0013350
5: 0.0020324, 0.0128438, 0.0015187, 0.0128808, -0.0083322, 0.0086753
6: -0.0017191, 0.0010250, -0.0017285, 0.0011554, -0.0022019, 0.0021148
7: -0.0075853, -0.0004857, -0.0076097, -0.0001483, -0.0056969, 0.0054716
8: -0.0035532, 0.0001804, -0.0035660, 0.0003579, -0.0029960, 0.0028775
9: -0.0020731, 0.0022563, -0.0022788, 0.0022711, -0.0033366, 0.0034740

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026284, upper bound: 0.0025389
time: 1.76 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B2

### Relational analysis result of NS_A1_A1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026284, upper bound: 0.0025389
time: 1.75 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.9890988, 0.9956085, 0.9884026, 0.9956475, -0.0051200, 0.0057533
1: -0.0039802, -0.0023582, -0.0041537, -0.0023485, -0.0012758, 0.0014336
2: 0.0024432, 0.0110391, 0.0023918, 0.0119586, -0.0075971, 0.0067610
3: -0.0062976, -0.0023852, -0.0067161, -0.0023617, -0.0030773, 0.0034579
4: 0.0010008, 0.0026645, 0.0009908, 0.0028424, -0.0014704, 0.0013086
5: 0.0020324, 0.0128438, 0.0019677, 0.0140002, -0.0095552, 0.0085035
6: -0.0017191, 0.0010250, -0.0020126, 0.0010414, -0.0021583, 0.0024252
7: -0.0075853, -0.0004857, -0.0083447, -0.0004432, -0.0055841, 0.0062748
8: -0.0035532, 0.0001804, -0.0039526, 0.0002028, -0.0029366, 0.0032998
9: -0.0020731, 0.0022563, -0.0020990, 0.0027193, -0.0038263, 0.0034052

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026284, upper bound: 0.0025389
time: 1.70 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B2

### Relational analysis result of NS_A1_A1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026284, upper bound: 0.0025389
time: 1.34 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.9890988, 0.9956085, 0.9887385, 0.9961405, -0.0055874, 0.0054717
1: -0.0039802, -0.0023582, -0.0040700, -0.0022256, -0.0013922, 0.0013634
2: 0.0024432, 0.0110391, 0.0017406, 0.0115149, -0.0072253, 0.0073781
3: -0.0062976, -0.0023852, -0.0065142, -0.0020654, -0.0033582, 0.0032887
4: 0.0010008, 0.0026645, 0.0008648, 0.0027566, -0.0013984, 0.0014280
5: 0.0020324, 0.0128438, 0.0011486, 0.0134421, -0.0090876, 0.0092797
6: -0.0017191, 0.0010250, -0.0018709, 0.0012493, -0.0023553, 0.0023065
7: -0.0075853, -0.0004857, -0.0079783, 0.0000947, -0.0060938, 0.0059677
8: -0.0035532, 0.0001804, -0.0037598, 0.0004856, -0.0032047, 0.0031383
9: -0.0020731, 0.0022563, -0.0024270, 0.0024959, -0.0036391, 0.0037160

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027302, upper bound: 0.0025389
time: 1.26 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027302, upper bound: 0.0025389
time: 1.67 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.9890988, 0.9956085, 0.9880091, 0.9959509, -0.0054762, 0.0062646
1: -0.0039802, -0.0023582, -0.0042518, -0.0022729, -0.0013645, 0.0015610
2: 0.0024432, 0.0110391, 0.0019910, 0.0124782, -0.0082723, 0.0072312
3: -0.0062976, -0.0023852, -0.0069527, -0.0021794, -0.0032913, 0.0037652
4: 0.0010008, 0.0026645, 0.0009133, 0.0029430, -0.0016011, 0.0013996
5: 0.0020324, 0.0128438, 0.0014637, 0.0146537, -0.0104044, 0.0090950
6: -0.0017191, 0.0010250, -0.0021784, 0.0011693, -0.0023084, 0.0026407
7: -0.0075853, -0.0004857, -0.0087739, -0.0001122, -0.0059725, 0.0068324
8: -0.0035532, 0.0001804, -0.0041783, 0.0003769, -0.0031409, 0.0035931
9: -0.0020731, 0.0022563, -0.0023008, 0.0029810, -0.0041664, 0.0036420

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027302, upper bound: 0.0025389
time: 1.28 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027302, upper bound: 0.0025389
time: 1.69 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.9884241, 0.9953319, 0.9890929, 0.9957256, -0.0058895, 0.0049950
1: -0.0041484, -0.0024271, -0.0039817, -0.0023290, -0.0014675, 0.0012446
2: 0.0028085, 0.0119302, 0.0022886, 0.0110469, -0.0065959, 0.0077770
3: -0.0067032, -0.0025514, -0.0063012, -0.0023148, -0.0035398, 0.0030021
4: 0.0010715, 0.0028369, 0.0009708, 0.0026660, -0.0012766, 0.0015052
5: 0.0024918, 0.0139644, 0.0018379, 0.0128536, -0.0082958, 0.0097815
6: -0.0020035, 0.0009084, -0.0017215, 0.0010743, -0.0024826, 0.0021056
7: -0.0083213, -0.0007874, -0.0075918, -0.0003580, -0.0064233, 0.0054478
8: -0.0039402, 0.0000218, -0.0035566, 0.0002476, -0.0033780, 0.0028649
9: -0.0018891, 0.0027050, -0.0021510, 0.0022602, -0.0033220, 0.0039169

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026104, upper bound: 0.0025667
time: 1.29 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_A1_A1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026104, upper bound: 0.0025667
time: 1.94 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9884241, 0.9953319, 0.9885445, 0.9956250, -0.0058931, 0.0056295
1: -0.0041484, -0.0024271, -0.0041183, -0.0023541, -0.0014684, 0.0014027
2: 0.0028085, 0.0119302, 0.0024214, 0.0117711, -0.0074337, 0.0077818
3: -0.0067032, -0.0025514, -0.0066308, -0.0023752, -0.0035420, 0.0033835
4: 0.0010715, 0.0028369, 0.0009965, 0.0028062, -0.0014388, 0.0015062
5: 0.0024918, 0.0139644, 0.0020049, 0.0137644, -0.0093496, 0.0097875
6: -0.0020035, 0.0009084, -0.0019527, 0.0010320, -0.0024842, 0.0023730
7: -0.0083213, -0.0007874, -0.0081899, -0.0004676, -0.0064273, 0.0061397
8: -0.0039402, 0.0000218, -0.0038711, 0.0001899, -0.0033801, 0.0032288
9: -0.0018891, 0.0027050, -0.0020841, 0.0026249, -0.0037440, 0.0039193

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026104, upper bound: 0.0025667
time: 1.74 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026104, upper bound: 0.0025667
time: 1.37 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9884241, 0.9953319, 0.9887640, 0.9959856, -0.0062450, 0.0054882
1: -0.0041484, -0.0024271, -0.0040637, -0.0022642, -0.0015561, 0.0013675
2: 0.0028085, 0.0119302, 0.0019452, 0.0114814, -0.0072471, 0.0082464
3: -0.0067032, -0.0025514, -0.0064990, -0.0021585, -0.0037534, 0.0032986
4: 0.0010715, 0.0028369, 0.0009044, 0.0027501, -0.0014027, 0.0015961
5: 0.0024918, 0.0139644, 0.0014060, 0.0134000, -0.0091149, 0.0103719
6: -0.0020035, 0.0009084, -0.0018602, 0.0011840, -0.0026325, 0.0023135
7: -0.0083213, -0.0007874, -0.0079506, -0.0000743, -0.0068111, 0.0059856
8: -0.0039402, 0.0000218, -0.0037453, 0.0003968, -0.0035819, 0.0031478
9: -0.0018891, 0.0027050, -0.0023239, 0.0024790, -0.0036500, 0.0041533

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026879, upper bound: 0.0025667
time: 1.67 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026879, upper bound: 0.0025667
time: 1.78 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9884241, 0.9953319, 0.9881956, 0.9959067, -0.0062569, 0.0060799
1: -0.0041484, -0.0024271, -0.0042053, -0.0022839, -0.0015591, 0.0015149
2: 0.0028085, 0.0119302, 0.0020494, 0.0122319, -0.0080284, 0.0082622
3: -0.0067032, -0.0025514, -0.0068406, -0.0022059, -0.0037606, 0.0036542
4: 0.0010715, 0.0028369, 0.0009245, 0.0028954, -0.0015539, 0.0015991
5: 0.0024918, 0.0139644, 0.0015371, 0.0143440, -0.0100976, 0.0103916
6: -0.0020035, 0.0009084, -0.0020998, 0.0011507, -0.0026375, 0.0025629
7: -0.0083213, -0.0007874, -0.0085705, -0.0001604, -0.0068240, 0.0066310
8: -0.0039402, 0.0000218, -0.0040713, 0.0003515, -0.0035887, 0.0034872
9: -0.0018891, 0.0027050, -0.0022714, 0.0028570, -0.0040435, 0.0041613

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026879, upper bound: 0.0025667
time: 1.78 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026879, upper bound: 0.0025667
time: 1.82 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.9885501, 0.9955158, 0.9890879, 0.9957963, -0.0057601, 0.0050254
1: -0.0041170, -0.0023813, -0.0039830, -0.0023114, -0.0014353, 0.0012522
2: 0.0025657, 0.0117637, 0.0021953, 0.0110537, -0.0066359, 0.0076061
3: -0.0066275, -0.0024409, -0.0063043, -0.0022723, -0.0034620, 0.0030204
4: 0.0010245, 0.0028047, 0.0009528, 0.0026673, -0.0012844, 0.0014721
5: 0.0021864, 0.0137551, 0.0017205, 0.0128621, -0.0083462, 0.0095665
6: -0.0019504, 0.0009859, -0.0017237, 0.0011042, -0.0024281, 0.0021184
7: -0.0081838, -0.0005868, -0.0075974, -0.0002809, -0.0062822, 0.0054809
8: -0.0038679, 0.0001272, -0.0035595, 0.0002882, -0.0033037, 0.0028823
9: -0.0020114, 0.0026212, -0.0021980, 0.0022636, -0.0033422, 0.0038308

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A2_A1_B1_B1_B1

### Relational analysis result of NS_A1_A1_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026111, upper bound: 0.0025747
time: 1.38 seconds

## Relational analysis of NS_A1_A1_A2_A1_B1_B1_B2

### Relational analysis result of NS_A1_A1_A2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026111, upper bound: 0.0025747
time: 1.23 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.9885501, 0.9955158, 0.9884132, 0.9955243, -0.0056615, 0.0058334
1: -0.0041170, -0.0023813, -0.0041511, -0.0023792, -0.0014107, 0.0014535
2: 0.0025657, 0.0117637, 0.0025543, 0.0119447, -0.0077030, 0.0074759
3: -0.0066275, -0.0024409, -0.0067098, -0.0024357, -0.0034027, 0.0035061
4: 0.0010245, 0.0028047, 0.0010223, 0.0028398, -0.0014909, 0.0014470
5: 0.0021864, 0.0137551, 0.0021720, 0.0139827, -0.0096883, 0.0094028
6: -0.0019504, 0.0009859, -0.0020081, 0.0009895, -0.0023865, 0.0024590
7: -0.0081838, -0.0005868, -0.0083333, -0.0005774, -0.0061747, 0.0063622
8: -0.0038679, 0.0001272, -0.0039465, 0.0001322, -0.0032472, 0.0033458
9: -0.0020114, 0.0026212, -0.0020172, 0.0027124, -0.0038796, 0.0037653

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A2_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026111, upper bound: 0.0025747
time: 1.78 seconds

## Relational analysis of NS_A1_A1_A2_A1_B1_B2_B2

### Relational analysis result of NS_A1_A1_A2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026111, upper bound: 0.0025747
time: 1.64 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.9885501, 0.9955158, 0.9887528, 0.9960285, -0.0061303, 0.0055170
1: -0.0041170, -0.0023813, -0.0040665, -0.0022535, -0.0015275, 0.0013747
2: 0.0025657, 0.0117637, 0.0018886, 0.0114961, -0.0072851, 0.0080950
3: -0.0066275, -0.0024409, -0.0065056, -0.0021327, -0.0036845, 0.0033159
4: 0.0010245, 0.0028047, 0.0008934, 0.0027529, -0.0014100, 0.0015668
5: 0.0021864, 0.0137551, 0.0013348, 0.0134185, -0.0091627, 0.0101814
6: -0.0019504, 0.0009859, -0.0018649, 0.0012020, -0.0025842, 0.0023256
7: -0.0081838, -0.0005868, -0.0079628, -0.0000276, -0.0066860, 0.0060170
8: -0.0038679, 0.0001272, -0.0037517, 0.0004213, -0.0035161, 0.0031643
9: -0.0020114, 0.0026212, -0.0023524, 0.0024864, -0.0036692, 0.0040771

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027009, upper bound: 0.0025747
time: 1.35 seconds

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B2

### Relational analysis result of NS_A1_A1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027009, upper bound: 0.0025747
time: 1.60 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.9885501, 0.9955158, 0.9880216, 0.9958356, -0.0060196, 0.0062478
1: -0.0041170, -0.0023813, -0.0042487, -0.0023016, -0.0014999, 0.0015568
2: 0.0025657, 0.0117637, 0.0021433, 0.0124617, -0.0082502, 0.0079488
3: -0.0066275, -0.0024409, -0.0069451, -0.0022487, -0.0036179, 0.0037551
4: 0.0010245, 0.0028047, 0.0009427, 0.0029398, -0.0015968, 0.0015385
5: 0.0021864, 0.0137551, 0.0016552, 0.0146330, -0.0103766, 0.0099974
6: -0.0019504, 0.0009859, -0.0021732, 0.0011207, -0.0025375, 0.0026337
7: -0.0081838, -0.0005868, -0.0087603, -0.0002379, -0.0065652, 0.0068142
8: -0.0038679, 0.0001272, -0.0041711, 0.0003107, -0.0034526, 0.0035835
9: -0.0020114, 0.0026212, -0.0022242, 0.0029727, -0.0041552, 0.0040034

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A2_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027009, upper bound: 0.0025747
time: 1.29 seconds

## Relational analysis of NS_A1_A1_A2_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027009, upper bound: 0.0025747
time: 1.76 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.9889771, 0.9957175, 0.9890759, 0.9959537, -0.0052928, 0.0050947
1: -0.0040106, -0.0023311, -0.0039859, -0.0022722, -0.0013188, 0.0012695
2: 0.0022994, 0.0111999, 0.0019873, 0.0110694, -0.0067275, 0.0069891
3: -0.0063708, -0.0023197, -0.0063114, -0.0021777, -0.0031812, 0.0030620
4: 0.0009729, 0.0026956, 0.0009125, 0.0026703, -0.0013021, 0.0013527
5: 0.0018514, 0.0130459, 0.0014590, 0.0128819, -0.0084614, 0.0087905
6: -0.0017704, 0.0010709, -0.0017287, 0.0011705, -0.0022311, 0.0021476
7: -0.0077181, -0.0003668, -0.0076103, -0.0001091, -0.0057726, 0.0055565
8: -0.0036230, 0.0002429, -0.0035664, 0.0003785, -0.0030358, 0.0029221
9: -0.0021455, 0.0023372, -0.0023027, 0.0022715, -0.0033883, 0.0035201

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A1_A1_B1_B1_B1

### Relational analysis result of NS_A1_A2_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026284, upper bound: 0.0025882
time: 1.27 seconds

## Relational analysis of NS_A1_A2_A1_A1_B1_B1_B2

### Relational analysis result of NS_A1_A2_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026284, upper bound: 0.0025882
time: 1.71 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.9889771, 0.9957175, 0.9884018, 0.9956866, -0.0051892, 0.0058272
1: -0.0040106, -0.0023311, -0.0041539, -0.0023387, -0.0012930, 0.0014520
2: 0.0022994, 0.0111999, 0.0023401, 0.0119597, -0.0076947, 0.0068523
3: -0.0063708, -0.0023197, -0.0067166, -0.0023382, -0.0031189, 0.0035023
4: 0.0009729, 0.0026956, 0.0009808, 0.0028427, -0.0014893, 0.0013263
5: 0.0018514, 0.0130459, 0.0019027, 0.0140016, -0.0096779, 0.0086184
6: -0.0017704, 0.0010709, -0.0020129, 0.0010579, -0.0021875, 0.0024564
7: -0.0077181, -0.0003668, -0.0083456, -0.0004005, -0.0056596, 0.0063553
8: -0.0036230, 0.0002429, -0.0039530, 0.0002252, -0.0029763, 0.0033422
9: -0.0021455, 0.0023372, -0.0021250, 0.0027199, -0.0038755, 0.0034512

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A1_A1_B1_B2_B1

### Relational analysis result of NS_A1_A2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026284, upper bound: 0.0025882
time: 1.67 seconds

## Relational analysis of NS_A1_A2_A1_A1_B1_B2_B2

### Relational analysis result of NS_A1_A2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026284, upper bound: 0.0025882
time: 1.80 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.9889771, 0.9957175, 0.9887375, 0.9961742, -0.0056546, 0.0055485
1: -0.0040106, -0.0023311, -0.0040702, -0.0022172, -0.0014090, 0.0013825
2: 0.0022994, 0.0111999, 0.0016962, 0.0115161, -0.0073268, 0.0074669
3: -0.0063708, -0.0023197, -0.0065148, -0.0020451, -0.0033986, 0.0033348
4: 0.0009729, 0.0026956, 0.0008562, 0.0027568, -0.0014181, 0.0014452
5: 0.0018514, 0.0130459, 0.0010928, 0.0134437, -0.0092151, 0.0093914
6: -0.0017704, 0.0010709, -0.0018713, 0.0012635, -0.0023836, 0.0023389
7: -0.0077181, -0.0003668, -0.0079793, 0.0001314, -0.0061672, 0.0060514
8: -0.0036230, 0.0002429, -0.0037604, 0.0005049, -0.0032433, 0.0031824
9: -0.0021455, 0.0023372, -0.0024493, 0.0024965, -0.0036901, 0.0037607

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_A2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027302, upper bound: 0.0025882
time: 1.61 seconds

## Relational analysis of NS_A1_A2_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_A2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027302, upper bound: 0.0025882
time: 1.75 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.9889771, 0.9957175, 0.9880080, 0.9959877, -0.0055482, 0.0063408
1: -0.0040106, -0.0023311, -0.0042520, -0.0022637, -0.0013825, 0.0015800
2: 0.0022994, 0.0111999, 0.0019425, 0.0124796, -0.0083730, 0.0073263
3: -0.0063708, -0.0023197, -0.0069533, -0.0021573, -0.0033346, 0.0038110
4: 0.0009729, 0.0026956, 0.0009039, 0.0029433, -0.0016206, 0.0014180
5: 0.0018514, 0.0130459, 0.0014027, 0.0146555, -0.0105310, 0.0092146
6: -0.0017704, 0.0010709, -0.0021789, 0.0011848, -0.0023388, 0.0026729
7: -0.0077181, -0.0003668, -0.0087751, -0.0000721, -0.0060511, 0.0069156
8: -0.0036230, 0.0002429, -0.0041789, 0.0003979, -0.0031822, 0.0036368
9: -0.0021455, 0.0023372, -0.0023253, 0.0029818, -0.0042171, 0.0036899

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027302, upper bound: 0.0025882
time: 1.82 seconds

## Relational analysis of NS_A1_A2_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_A2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027302, upper bound: 0.0025882
time: 1.24 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.9883012, 0.9954404, 0.9890923, 0.9957647, -0.0059672, 0.0050765
1: -0.0041790, -0.0024001, -0.0039819, -0.0023193, -0.0014869, 0.0012649
2: 0.0026652, 0.0120923, 0.0022370, 0.0110478, -0.0067035, 0.0078796
3: -0.0067770, -0.0024862, -0.0063016, -0.0022913, -0.0035864, 0.0030511
4: 0.0010437, 0.0028683, 0.0009609, 0.0026662, -0.0012975, 0.0015251
5: 0.0023116, 0.0141684, 0.0017730, 0.0128547, -0.0084313, 0.0099105
6: -0.0020553, 0.0009541, -0.0017218, 0.0010908, -0.0025154, 0.0021399
7: -0.0084552, -0.0006690, -0.0075925, -0.0003153, -0.0065081, 0.0055367
8: -0.0040107, 0.0000840, -0.0035570, 0.0002700, -0.0034225, 0.0029117
9: -0.0019613, 0.0027867, -0.0021770, 0.0022607, -0.0033762, 0.0039686

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A2_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A2_A1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026104, upper bound: 0.0026251
time: 1.62 seconds

## Relational analysis of NS_A1_A2_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_A2_A1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026104, upper bound: 0.0026251
time: 1.72 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9883012, 0.9954404, 0.9885438, 0.9956619, -0.0059783, 0.0057100
1: -0.0041790, -0.0024001, -0.0041185, -0.0023449, -0.0014896, 0.0014228
2: 0.0026652, 0.0120923, 0.0023726, 0.0117720, -0.0075400, 0.0078943
3: -0.0067770, -0.0024862, -0.0066312, -0.0023530, -0.0035932, 0.0034319
4: 0.0010437, 0.0028683, 0.0009871, 0.0028063, -0.0014593, 0.0015279
5: 0.0023116, 0.0141684, 0.0019436, 0.0137656, -0.0094833, 0.0099290
6: -0.0020553, 0.0009541, -0.0019530, 0.0010475, -0.0025201, 0.0024070
7: -0.0084552, -0.0006690, -0.0081907, -0.0004274, -0.0065202, 0.0062275
8: -0.0040107, 0.0000840, -0.0038715, 0.0002111, -0.0034289, 0.0032750
9: -0.0019613, 0.0027867, -0.0021086, 0.0026254, -0.0037975, 0.0039760

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A2_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_A2_A1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026104, upper bound: 0.0026251
time: 1.72 seconds

## Relational analysis of NS_A1_A2_A1_A2_B1_B2_B2

### Relational analysis result of NS_A1_A2_A1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026104, upper bound: 0.0026251
time: 1.44 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9883012, 0.9954404, 0.9887629, 0.9960266, -0.0063237, 0.0055711
1: -0.0041790, -0.0024001, -0.0040639, -0.0022540, -0.0015757, 0.0013882
2: 0.0026652, 0.0120923, 0.0018911, 0.0114827, -0.0073566, 0.0083504
3: -0.0067770, -0.0024862, -0.0064996, -0.0021339, -0.0038007, 0.0033484
4: 0.0010437, 0.0028683, 0.0008939, 0.0027503, -0.0014239, 0.0016162
5: 0.0023116, 0.0141684, 0.0013380, 0.0134017, -0.0092527, 0.0105026
6: -0.0020553, 0.0009541, -0.0018607, 0.0012012, -0.0026657, 0.0023484
7: -0.0084552, -0.0006690, -0.0079517, -0.0000296, -0.0068969, 0.0060761
8: -0.0040107, 0.0000840, -0.0037459, 0.0004203, -0.0036270, 0.0031954
9: -0.0019613, 0.0027867, -0.0023512, 0.0024797, -0.0037052, 0.0042057

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A2_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_A2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026879, upper bound: 0.0026251
time: 1.61 seconds

## Relational analysis of NS_A1_A2_A1_A2_B2_B1_B2

### Relational analysis result of NS_A1_A2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026879, upper bound: 0.0026251
time: 1.67 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9883012, 0.9954404, 0.9881945, 0.9959368, -0.0063415, 0.0061610
1: -0.0041790, -0.0024001, -0.0042056, -0.0022764, -0.0015801, 0.0015352
2: 0.0026652, 0.0120923, 0.0020098, 0.0122333, -0.0081355, 0.0083739
3: -0.0067770, -0.0024862, -0.0068412, -0.0021879, -0.0038114, 0.0037030
4: 0.0010437, 0.0028683, 0.0009169, 0.0028956, -0.0015746, 0.0016207
5: 0.0023116, 0.0141684, 0.0014872, 0.0143457, -0.0102324, 0.0105322
6: -0.0020553, 0.0009541, -0.0021003, 0.0011634, -0.0026732, 0.0025971
7: -0.0084552, -0.0006690, -0.0085716, -0.0001276, -0.0069163, 0.0067195
8: -0.0040107, 0.0000840, -0.0040719, 0.0003687, -0.0036372, 0.0035337
9: -0.0019613, 0.0027867, -0.0022914, 0.0028577, -0.0040975, 0.0042175

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_A2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_A2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026879, upper bound: 0.0026251
time: 1.80 seconds

## Relational analysis of NS_A1_A2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_A2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026879, upper bound: 0.0026251
time: 1.27 seconds

## BFS NS instance: NS_A1_A2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.9884502, 0.9956167, 0.9890872, 0.9958327, -0.0058489, 0.0050960
1: -0.0041419, -0.0023562, -0.0039831, -0.0023023, -0.0014574, 0.0012698
2: 0.0024324, 0.0118957, 0.0021472, 0.0110546, -0.0067292, 0.0077234
3: -0.0066875, -0.0023803, -0.0063047, -0.0022504, -0.0035153, 0.0030628
4: 0.0009987, 0.0028303, 0.0009435, 0.0026675, -0.0013024, 0.0014948
5: 0.0020188, 0.0139211, 0.0016601, 0.0128632, -0.0084635, 0.0097140
6: -0.0019925, 0.0010284, -0.0017240, 0.0011195, -0.0024655, 0.0021481
7: -0.0082928, -0.0004767, -0.0075981, -0.0002412, -0.0063790, 0.0055579
8: -0.0039252, 0.0001851, -0.0035599, 0.0003090, -0.0033547, 0.0029228
9: -0.0020785, 0.0026877, -0.0022222, 0.0022640, -0.0033892, 0.0038899

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A2_A1_B1_B1_B1

### Relational analysis result of NS_A1_A2_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026111, upper bound: 0.0026220
time: 1.29 seconds

## Relational analysis of NS_A1_A2_A2_A1_B1_B1_B2

### Relational analysis result of NS_A1_A2_A2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026111, upper bound: 0.0026220
time: 1.26 seconds

## BFS NS instance: NS_A1_A2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.9884502, 0.9956167, 0.9884123, 0.9955615, -0.0057447, 0.0059036
1: -0.0041419, -0.0023562, -0.0041513, -0.0023699, -0.0014314, 0.0014710
2: 0.0024324, 0.0118957, 0.0025053, 0.0119458, -0.0077957, 0.0075858
3: -0.0066875, -0.0023803, -0.0067103, -0.0024134, -0.0034527, 0.0035483
4: 0.0009987, 0.0028303, 0.0010128, 0.0028400, -0.0015088, 0.0014682
5: 0.0020188, 0.0139211, 0.0021104, 0.0139841, -0.0098049, 0.0095409
6: -0.0019925, 0.0010284, -0.0020085, 0.0010052, -0.0024216, 0.0024886
7: -0.0082928, -0.0004767, -0.0083342, -0.0005369, -0.0062654, 0.0064387
8: -0.0039252, 0.0001851, -0.0039470, 0.0001535, -0.0032949, 0.0033861
9: -0.0020785, 0.0026877, -0.0020418, 0.0027129, -0.0039263, 0.0038206

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A2_A1_B1_B2_B1

### Relational analysis result of NS_A1_A2_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026111, upper bound: 0.0026220
time: 1.68 seconds

## Relational analysis of NS_A1_A2_A2_A1_B1_B2_B2

### Relational analysis result of NS_A1_A2_A2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026111, upper bound: 0.0026220
time: 1.74 seconds

## BFS NS instance: NS_A1_A2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.9884502, 0.9956167, 0.9887518, 0.9960607, -0.0062155, 0.0055911
1: -0.0041419, -0.0023562, -0.0040667, -0.0022455, -0.0015487, 0.0013931
2: 0.0024324, 0.0118957, 0.0018461, 0.0114974, -0.0073829, 0.0082075
3: -0.0066875, -0.0023803, -0.0065062, -0.0021134, -0.0037357, 0.0033604
4: 0.0009987, 0.0028303, 0.0008852, 0.0027532, -0.0014290, 0.0015885
5: 0.0020188, 0.0139211, 0.0012814, 0.0134201, -0.0092858, 0.0103229
6: -0.0019925, 0.0010284, -0.0018653, 0.0012156, -0.0026201, 0.0023568
7: -0.0082928, -0.0004767, -0.0079638, 0.0000075, -0.0067789, 0.0060979
8: -0.0039252, 0.0001851, -0.0037522, 0.0004398, -0.0035649, 0.0032068
9: -0.0020785, 0.0026877, -0.0023738, 0.0024870, -0.0037184, 0.0041337

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A2_A1_B2_B1_B1

### Relational analysis result of NS_A1_A2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027009, upper bound: 0.0026220
time: 1.68 seconds

## Relational analysis of NS_A1_A2_A2_A1_B2_B1_B2

### Relational analysis result of NS_A1_A2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027009, upper bound: 0.0026220
time: 1.36 seconds

## BFS NS instance: NS_A1_A2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.9884502, 0.9956167, 0.9880205, 0.9958704, -0.0061115, 0.0063179
1: -0.0041419, -0.0023562, -0.0042489, -0.0022929, -0.0015228, 0.0015743
2: 0.0024324, 0.0118957, 0.0020974, 0.0124631, -0.0083427, 0.0080702
3: -0.0066875, -0.0023803, -0.0069458, -0.0022278, -0.0036732, 0.0037973
4: 0.0009987, 0.0028303, 0.0009338, 0.0029401, -0.0016147, 0.0015620
5: 0.0020188, 0.0139211, 0.0015974, 0.0146347, -0.0104930, 0.0101502
6: -0.0019925, 0.0010284, -0.0021736, 0.0011354, -0.0025762, 0.0026632
7: -0.0082928, -0.0004767, -0.0087614, -0.0002000, -0.0066655, 0.0068906
8: -0.0039252, 0.0001851, -0.0041717, 0.0003307, -0.0035053, 0.0036237
9: -0.0020785, 0.0026877, -0.0022473, 0.0029734, -0.0042018, 0.0040646

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A2_A1_B2_B2_B1

### Relational analysis result of NS_A1_A2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027009, upper bound: 0.0026220
time: 1.78 seconds

## Relational analysis of NS_A1_A2_A2_A1_B2_B2_B2

### Relational analysis result of NS_A1_A2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027009, upper bound: 0.0026220
time: 1.89 seconds

## BFS NS instance: NS_A1_A2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.9877682, 0.9953254, 0.9890872, 0.9958327, -0.0066412, 0.0049748
1: -0.0043118, -0.0024287, -0.0039831, -0.0023023, -0.0016548, 0.0012396
2: 0.0028169, 0.0127963, 0.0021472, 0.0110546, -0.0065692, 0.0087696
3: -0.0070974, -0.0025553, -0.0063047, -0.0022504, -0.0039915, 0.0029900
4: 0.0010731, 0.0030046, 0.0009435, 0.0026675, -0.0012715, 0.0016973
5: 0.0025024, 0.0150539, 0.0016601, 0.0128632, -0.0082623, 0.0110298
6: -0.0022800, 0.0009057, -0.0017240, 0.0011195, -0.0027995, 0.0020971
7: -0.0090367, -0.0007943, -0.0075981, -0.0002412, -0.0072431, 0.0054258
8: -0.0043164, 0.0000181, -0.0035599, 0.0003090, -0.0038091, 0.0028533
9: -0.0018849, 0.0031413, -0.0022222, 0.0022640, -0.0033086, 0.0044168

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A2_A2_B1_B1_B1

### Relational analysis result of NS_A1_A2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026448
time: 1.34 seconds

## Relational analysis of NS_A1_A2_A2_A2_B1_B1_B2

### Relational analysis result of NS_A1_A2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026448
time: 1.59 seconds

## BFS NS instance: NS_A1_A2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9877682, 0.9953254, 0.9884123, 0.9955615, -0.0059621, 0.0051974
1: -0.0043118, -0.0024287, -0.0041513, -0.0023699, -0.0014856, 0.0012951
2: 0.0028169, 0.0127963, 0.0025053, 0.0119458, -0.0068631, 0.0078728
3: -0.0070974, -0.0025553, -0.0067103, -0.0024134, -0.0035834, 0.0031238
4: 0.0010731, 0.0030046, 0.0010128, 0.0028400, -0.0013283, 0.0015238
5: 0.0025024, 0.0150539, 0.0021104, 0.0139841, -0.0086320, 0.0099020
6: -0.0022800, 0.0009057, -0.0020085, 0.0010052, -0.0025132, 0.0021909
7: -0.0090367, -0.0007943, -0.0083342, -0.0005369, -0.0065025, 0.0056685
8: -0.0043164, 0.0000181, -0.0039470, 0.0001535, -0.0034196, 0.0029810
9: -0.0018849, 0.0031413, -0.0020418, 0.0027129, -0.0034566, 0.0039652

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A2_A2_B1_B2_B1

### Relational analysis result of NS_A1_A2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026448
time: 1.68 seconds

## Relational analysis of NS_A1_A2_A2_A2_B1_B2_B2

### Relational analysis result of NS_A1_A2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026448
time: 1.80 seconds

## BFS NS instance: NS_A1_A2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9877682, 0.9953254, 0.9887518, 0.9960607, -0.0070078, 0.0054821
1: -0.0043118, -0.0024287, -0.0040667, -0.0022455, -0.0017462, 0.0013660
2: 0.0028169, 0.0127963, 0.0018461, 0.0114974, -0.0072391, 0.0092537
3: -0.0070974, -0.0025553, -0.0065062, -0.0021134, -0.0042119, 0.0032949
4: 0.0010731, 0.0030046, 0.0008852, 0.0027532, -0.0014011, 0.0017910
5: 0.0025024, 0.0150539, 0.0012814, 0.0134201, -0.0091049, 0.0116387
6: -0.0022800, 0.0009057, -0.0018653, 0.0012156, -0.0029540, 0.0023109
7: -0.0090367, -0.0007943, -0.0079638, 0.0000075, -0.0076430, 0.0059790
8: -0.0043164, 0.0000181, -0.0037522, 0.0004398, -0.0040194, 0.0031443
9: -0.0018849, 0.0031413, -0.0023738, 0.0024870, -0.0036460, 0.0046607

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A2_A2_B2_B1_B1

### Relational analysis result of NS_A1_A2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026448
time: 1.36 seconds

## Relational analysis of NS_A1_A2_A2_A2_B2_B1_B2

### Relational analysis result of NS_A1_A2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026448
time: 1.77 seconds

## BFS NS instance: NS_A1_A2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9877682, 0.9953254, 0.9880205, 0.9958704, -0.0063314, 0.0056294
1: -0.0043118, -0.0024287, -0.0042489, -0.0022929, -0.0015776, 0.0014027
2: 0.0028169, 0.0127963, 0.0020974, 0.0124631, -0.0074335, 0.0083605
3: -0.0070974, -0.0025553, -0.0069458, -0.0022278, -0.0038053, 0.0033834
4: 0.0010731, 0.0030046, 0.0009338, 0.0029401, -0.0014387, 0.0016182
5: 0.0025024, 0.0150539, 0.0015974, 0.0146347, -0.0093495, 0.0105153
6: -0.0022800, 0.0009057, -0.0021736, 0.0011354, -0.0026689, 0.0023730
7: -0.0090367, -0.0007943, -0.0087614, -0.0002000, -0.0069053, 0.0061396
8: -0.0043164, 0.0000181, -0.0041717, 0.0003307, -0.0036314, 0.0032288
9: -0.0018849, 0.0031413, -0.0022473, 0.0029734, -0.0037439, 0.0042108

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A2_A2_B2_B2_B1

### Relational analysis result of NS_A1_A2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026448
time: 1.31 seconds

## Relational analysis of NS_A1_A2_A2_A2_B2_B2_B2

### Relational analysis result of NS_A1_A2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026556, upper bound: 0.0026448
time: 1.91 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9887465, 0.9959960, 0.9890911, 0.9957576, -0.0055836, 0.0054754
1: -0.0040680, -0.0022616, -0.0039822, -0.0023210, -0.0013913, 0.0013643
2: 0.0019315, 0.0115044, 0.0022463, 0.0110494, -0.0072302, 0.0073731
3: -0.0065094, -0.0021523, -0.0063023, -0.0022956, -0.0033559, 0.0032909
4: 0.0009017, 0.0027545, 0.0009627, 0.0026665, -0.0013994, 0.0014270
5: 0.0013888, 0.0134290, 0.0017847, 0.0128567, -0.0090936, 0.0092734
6: -0.0018676, 0.0011883, -0.0017223, 0.0010878, -0.0023537, 0.0023081
7: -0.0079696, -0.0000630, -0.0075939, -0.0003230, -0.0060897, 0.0059717
8: -0.0037553, 0.0004027, -0.0035577, 0.0002660, -0.0032025, 0.0031404
9: -0.0023308, 0.0024906, -0.0021723, 0.0022615, -0.0036415, 0.0037135

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B1_B1_B1_A1_A1

### Relational analysis result of NS_A2_A1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026726
time: 1.60 seconds

## Relational analysis of NS_A2_A1_B1_B1_B1_A1_A2

### Relational analysis result of NS_A2_A1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027112
time: 1.65 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B1_A2

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

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B1_B1_B1_A2_A1

### Relational analysis result of NS_A2_A1_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027123
time: 1.33 seconds

## Relational analysis of NS_A2_A1_B1_B1_B1_A2_A2

### Relational analysis result of NS_A2_A1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027233
time: 1.34 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9887465, 0.9959960, 0.9887607, 0.9959766, -0.0053248, 0.0053679
1: -0.0040680, -0.0022616, -0.0040645, -0.0022665, -0.0013268, 0.0013375
2: 0.0019315, 0.0115044, 0.0019570, 0.0114857, -0.0070883, 0.0070313
3: -0.0065094, -0.0021523, -0.0065009, -0.0021639, -0.0032003, 0.0032263
4: 0.0009017, 0.0027545, 0.0009067, 0.0027509, -0.0013719, 0.0013609
5: 0.0013888, 0.0134290, 0.0014209, 0.0134054, -0.0089152, 0.0088435
6: -0.0018676, 0.0011883, -0.0018616, 0.0011802, -0.0022446, 0.0022628
7: -0.0079696, -0.0000630, -0.0079542, -0.0000841, -0.0058074, 0.0058545
8: -0.0037553, 0.0004027, -0.0037472, 0.0003916, -0.0030541, 0.0030788
9: -0.0023308, 0.0024906, -0.0023180, 0.0024812, -0.0035700, 0.0035413

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B1_B1_B2_A1_A1

### Relational analysis result of NS_A2_A1_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0026756
time: 1.23 seconds

## Relational analysis of NS_A2_A1_B1_B1_B2_A1_A2

### Relational analysis result of NS_A2_A1_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027157
time: 1.26 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B2_A2

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

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B1_B1_B2_A2_A1

### Relational analysis result of NS_A2_A1_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027181
time: 1.21 seconds

## Relational analysis of NS_A2_A1_B1_B1_B2_A2_A2

### Relational analysis result of NS_A2_A1_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027289
time: 1.20 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9887717, 0.9958357, 0.9884161, 0.9954856, -0.0055984, 0.0061307
1: -0.0040617, -0.0023016, -0.0041503, -0.0023888, -0.0013950, 0.0015276
2: 0.0021432, 0.0114710, 0.0026055, 0.0119407, -0.0080955, 0.0073926
3: -0.0064942, -0.0022486, -0.0067080, -0.0024590, -0.0033648, 0.0036847
4: 0.0009427, 0.0027481, 0.0010322, 0.0028390, -0.0015669, 0.0014308
5: 0.0016551, 0.0133870, 0.0022364, 0.0139777, -0.0101820, 0.0092980
6: -0.0018569, 0.0011208, -0.0020069, 0.0009732, -0.0023599, 0.0025843
7: -0.0079420, -0.0002379, -0.0083300, -0.0006196, -0.0061059, 0.0066864
8: -0.0037408, 0.0003108, -0.0039448, 0.0001100, -0.0032110, 0.0035163
9: -0.0022242, 0.0024738, -0.0019914, 0.0027103, -0.0040773, 0.0037233

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A1_B1_B2_B1_A1_A1

### Relational analysis result of NS_A2_A1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026391
time: 1.73 seconds

## Relational analysis of NS_A2_A1_B1_B2_B1_A1_A2

### Relational analysis result of NS_A2_A1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027123
time: 1.24 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9882032, 0.9957611, 0.9884161, 0.9954856, -0.0061942, 0.0061388
1: -0.0042034, -0.0023202, -0.0041503, -0.0023888, -0.0015434, 0.0015296
2: 0.0022418, 0.0122218, 0.0026055, 0.0119407, -0.0081062, 0.0081793
3: -0.0068360, -0.0022935, -0.0067080, -0.0024590, -0.0037229, 0.0036896
4: 0.0009618, 0.0028934, 0.0010322, 0.0028390, -0.0015689, 0.0015831
5: 0.0017790, 0.0143313, 0.0022364, 0.0139777, -0.0101955, 0.0102875
6: -0.0020966, 0.0010893, -0.0020069, 0.0009732, -0.0026111, 0.0025877
7: -0.0085622, -0.0003193, -0.0083300, -0.0006196, -0.0067556, 0.0066952
8: -0.0040669, 0.0002680, -0.0039448, 0.0001100, -0.0035527, 0.0035210
9: -0.0021746, 0.0028519, -0.0019914, 0.0027103, -0.0040827, 0.0041196

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A1_B1_B2_B1_A2_A1

### Relational analysis result of NS_A2_A1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026654
time: 1.23 seconds

## Relational analysis of NS_A2_A1_B1_B2_B1_A2_A2

### Relational analysis result of NS_A2_A1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027193
time: 1.55 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9887717, 0.9958357, 0.9880242, 0.9957840, -0.0053599, 0.0059788
1: -0.0040617, -0.0023016, -0.0042480, -0.0023145, -0.0013355, 0.0014898
2: 0.0021432, 0.0114710, 0.0022114, 0.0124582, -0.0078950, 0.0070776
3: -0.0064942, -0.0022486, -0.0069436, -0.0022797, -0.0032214, 0.0035935
4: 0.0009427, 0.0027481, 0.0009559, 0.0029391, -0.0015281, 0.0013699
5: 0.0016551, 0.0133870, 0.0017408, 0.0146286, -0.0099298, 0.0089018
6: -0.0018569, 0.0011208, -0.0021721, 0.0010990, -0.0022594, 0.0025203
7: -0.0079420, -0.0002379, -0.0087574, -0.0002942, -0.0058457, 0.0065208
8: -0.0037408, 0.0003108, -0.0041696, 0.0002811, -0.0030742, 0.0034292
9: -0.0022242, 0.0024738, -0.0021898, 0.0029710, -0.0039763, 0.0035647

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A1_B1_B2_B2_A1_A1

### Relational analysis result of NS_A2_A1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0026436
time: 1.67 seconds

## Relational analysis of NS_A2_A1_B1_B2_B2_A1_A2

### Relational analysis result of NS_A2_A1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027181
time: 1.60 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9882032, 0.9957611, 0.9880242, 0.9957840, -0.0059745, 0.0059706
1: -0.0042034, -0.0023202, -0.0042480, -0.0023145, -0.0014887, 0.0014877
2: 0.0022418, 0.0122218, 0.0022114, 0.0124582, -0.0078842, 0.0078892
3: -0.0068360, -0.0022935, -0.0069436, -0.0022797, -0.0035908, 0.0035885
4: 0.0009618, 0.0028934, 0.0009559, 0.0029391, -0.0015260, 0.0015269
5: 0.0017790, 0.0143313, 0.0017408, 0.0146286, -0.0099162, 0.0099226
6: -0.0020966, 0.0010893, -0.0021721, 0.0010990, -0.0025185, 0.0025168
7: -0.0085622, -0.0003193, -0.0087574, -0.0002942, -0.0065160, 0.0065118
8: -0.0040669, 0.0002680, -0.0041696, 0.0002811, -0.0034267, 0.0034245
9: -0.0021746, 0.0028519, -0.0021898, 0.0029710, -0.0039709, 0.0039734

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A1_B1_B2_B2_A2_A1

### Relational analysis result of NS_A2_A1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0026720
time: 1.96 seconds

## Relational analysis of NS_A2_A1_B1_B2_B2_A2_A2

### Relational analysis result of NS_A2_A1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027263
time: 1.57 seconds

## BFS NS instance: NS_A2_A1_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9887606, 0.9958824, 0.9885427, 0.9956633, -0.0056309, 0.0060156
1: -0.0040645, -0.0022900, -0.0041188, -0.0023445, -0.0014031, 0.0014989
2: 0.0020816, 0.0114857, 0.0023709, 0.0117737, -0.0079436, 0.0074355
3: -0.0065009, -0.0022206, -0.0066320, -0.0023522, -0.0033843, 0.0036156
4: 0.0009308, 0.0027509, 0.0009868, 0.0028067, -0.0015375, 0.0014391
5: 0.0015775, 0.0134055, 0.0019414, 0.0137676, -0.0099909, 0.0093519
6: -0.0018616, 0.0011404, -0.0019535, 0.0010481, -0.0023736, 0.0025358
7: -0.0079542, -0.0001870, -0.0081920, -0.0004259, -0.0061412, 0.0065609
8: -0.0037472, 0.0003375, -0.0038723, 0.0002119, -0.0032296, 0.0034503
9: -0.0022552, 0.0024812, -0.0021095, 0.0026262, -0.0040008, 0.0037449

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_A1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B2_B1_B1_A1_A1

### Relational analysis result of NS_A2_A1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026421
time: 1.28 seconds

## Relational analysis of NS_A2_A1_B2_B1_B1_A1_A2

### Relational analysis result of NS_A2_A1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027112
time: 1.68 seconds

## BFS NS instance: NS_A2_A1_B2_B1_B1_A2

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

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B2_B1_B1_A2_A1

### Relational analysis result of NS_A2_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026692
time: 1.79 seconds

## Relational analysis of NS_A2_A1_B2_B1_B1_A2_A2

### Relational analysis result of NS_A2_A1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027223
time: 1.28 seconds

## BFS NS instance: NS_A2_A1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9887606, 0.9958824, 0.9881929, 0.9958961, -0.0053834, 0.0059098
1: -0.0040645, -0.0022900, -0.0042060, -0.0022865, -0.0013414, 0.0014726
2: 0.0020816, 0.0114857, 0.0020634, 0.0122354, -0.0078038, 0.0071088
3: -0.0065009, -0.0022206, -0.0068422, -0.0022123, -0.0032356, 0.0035520
4: 0.0009308, 0.0027509, 0.0009273, 0.0028960, -0.0015104, 0.0013759
5: 0.0015775, 0.0134055, 0.0015546, 0.0143484, -0.0098151, 0.0089409
6: -0.0018616, 0.0011404, -0.0021009, 0.0011462, -0.0022693, 0.0024912
7: -0.0079542, -0.0001870, -0.0085734, -0.0001719, -0.0058714, 0.0064455
8: -0.0037472, 0.0003375, -0.0040728, 0.0003454, -0.0030877, 0.0033896
9: -0.0022552, 0.0024812, -0.0022644, 0.0028588, -0.0039304, 0.0035803

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B2_B1_B2_A1_A1

### Relational analysis result of NS_A2_A1_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0026460
time: 1.89 seconds

## Relational analysis of NS_A2_A1_B2_B1_B2_A1_A2

### Relational analysis result of NS_A2_A1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027157
time: 1.52 seconds

## BFS NS instance: NS_A2_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9880295, 0.9956904, 0.9881929, 0.9958961, -0.0061760, 0.0058062
1: -0.0042467, -0.0023378, -0.0042060, -0.0022865, -0.0015389, 0.0014468
2: 0.0023350, 0.0124512, 0.0020634, 0.0122354, -0.0076671, 0.0081553
3: -0.0069404, -0.0023359, -0.0068422, -0.0022123, -0.0037119, 0.0034897
4: 0.0009798, 0.0029378, 0.0009273, 0.0028960, -0.0014839, 0.0015784
5: 0.0018963, 0.0146197, 0.0015546, 0.0143484, -0.0096431, 0.0102572
6: -0.0021698, 0.0010595, -0.0021009, 0.0011462, -0.0026034, 0.0024475
7: -0.0087516, -0.0003963, -0.0085734, -0.0001719, -0.0067357, 0.0063325
8: -0.0041665, 0.0002274, -0.0040728, 0.0003454, -0.0035423, 0.0033302
9: -0.0021276, 0.0029674, -0.0022644, 0.0028588, -0.0038615, 0.0041074

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B2_B1_B2_A2_A1

### Relational analysis result of NS_A2_A1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0026763
time: 1.84 seconds

## Relational analysis of NS_A2_A1_B2_B1_B2_A2_A2

### Relational analysis result of NS_A2_A1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027281
time: 1.60 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9887606, 0.9958824, 0.9878703, 0.9953730, -0.0055223, 0.0068065
1: -0.0040645, -0.0022900, -0.0042863, -0.0024169, -0.0013760, 0.0016960
2: 0.0020816, 0.0114857, 0.0027543, 0.0126613, -0.0089878, 0.0072922
3: -0.0065009, -0.0022206, -0.0070360, -0.0025268, -0.0033191, 0.0040909
4: 0.0009308, 0.0027509, 0.0010610, 0.0029785, -0.0017396, 0.0014114
5: 0.0015775, 0.0134055, 0.0024236, 0.0148841, -0.0113043, 0.0091716
6: -0.0018616, 0.0011404, -0.0022369, 0.0009257, -0.0023279, 0.0028692
7: -0.0079542, -0.0001870, -0.0089252, -0.0007426, -0.0060229, 0.0074234
8: -0.0037472, 0.0003375, -0.0042578, 0.0000453, -0.0031674, 0.0039039
9: -0.0022552, 0.0024812, -0.0019164, 0.0030733, -0.0045268, 0.0036727

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B2_B2_B1_A1_A1

### Relational analysis result of NS_A2_A1_B2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026061
time: 1.52 seconds

## Relational analysis of NS_A2_A1_B2_B2_B1_A1_A2

### Relational analysis result of NS_A2_A1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026648
time: 1.74 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B1_A2

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

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B2_B2_B1_A2_A1

### Relational analysis result of NS_A2_A1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026678
time: 1.75 seconds

## Relational analysis of NS_A2_A1_B2_B2_B1_A2_A2

### Relational analysis result of NS_A2_A1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027090
time: 1.27 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9887606, 0.9958824, 0.9874700, 0.9956874, -0.0052597, 0.0066264
1: -0.0040645, -0.0022900, -0.0043861, -0.0023385, -0.0013106, 0.0016511
2: 0.0020816, 0.0114857, 0.0023390, 0.0131901, -0.0087501, 0.0069454
3: -0.0065009, -0.0022206, -0.0072767, -0.0023377, -0.0031613, 0.0039827
4: 0.0009308, 0.0027509, 0.0009806, 0.0030808, -0.0016936, 0.0013443
5: 0.0015775, 0.0134055, 0.0019013, 0.0155491, -0.0110054, 0.0087355
6: -0.0018616, 0.0011404, -0.0024057, 0.0010583, -0.0022172, 0.0027933
7: -0.0079542, -0.0001870, -0.0093619, -0.0003996, -0.0057365, 0.0072271
8: -0.0037472, 0.0003375, -0.0044875, 0.0002257, -0.0030168, 0.0038006
9: -0.0022552, 0.0024812, -0.0021256, 0.0033396, -0.0044070, 0.0034981

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_A1

### Relational analysis result of NS_A2_A1_B2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0026110
time: 1.71 seconds

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_A2

### Relational analysis result of NS_A2_A1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0026719
time: 1.71 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9880295, 0.9956904, 0.9874700, 0.9956874, -0.0054712, 0.0059769
1: -0.0042467, -0.0023378, -0.0043861, -0.0023385, -0.0013633, 0.0014893
2: 0.0023350, 0.0124512, 0.0023390, 0.0131901, -0.0078925, 0.0072247
3: -0.0069404, -0.0023359, -0.0072767, -0.0023377, -0.0032884, 0.0035923
4: 0.0009798, 0.0029378, 0.0009806, 0.0030808, -0.0015276, 0.0013983
5: 0.0018963, 0.0146197, 0.0019013, 0.0155491, -0.0099266, 0.0090868
6: -0.0021698, 0.0010595, -0.0024057, 0.0010583, -0.0023063, 0.0025195
7: -0.0087516, -0.0003963, -0.0093619, -0.0003996, -0.0059671, 0.0065187
8: -0.0041665, 0.0002274, -0.0044875, 0.0002257, -0.0031381, 0.0034281
9: -0.0021276, 0.0029674, -0.0021256, 0.0033396, -0.0039751, 0.0036387

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_A1

### Relational analysis result of NS_A2_A1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0026761
time: 1.55 seconds

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_A2

### Relational analysis result of NS_A2_A1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027196
time: 1.27 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9886309, 0.9960932, 0.9890904, 0.9957986, -0.0056597, 0.0055495
1: -0.0040968, -0.0022374, -0.0039823, -0.0023109, -0.0014102, 0.0013828
2: 0.0018032, 0.0116571, 0.0021923, 0.0110503, -0.0073280, 0.0074735
3: -0.0065789, -0.0020939, -0.0063027, -0.0022710, -0.0034016, 0.0033354
4: 0.0008769, 0.0027841, 0.0009522, 0.0026666, -0.0014183, 0.0014465
5: 0.0012274, 0.0136211, 0.0017168, 0.0128578, -0.0092167, 0.0093997
6: -0.0019163, 0.0012293, -0.0017226, 0.0011051, -0.0023857, 0.0023393
7: -0.0080958, 0.0000430, -0.0075945, -0.0002784, -0.0061727, 0.0060525
8: -0.0038216, 0.0004584, -0.0035580, 0.0002894, -0.0032461, 0.0031829
9: -0.0023954, 0.0025675, -0.0021995, 0.0022619, -0.0036908, 0.0037641

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A2_B1_B1_B1_A1_A1

### Relational analysis result of NS_A2_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027302
time: 1.30 seconds

## Relational analysis of NS_A2_A2_B1_B1_B1_A1_A2

### Relational analysis result of NS_A2_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027652
time: 1.43 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B1_A2

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

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A2_B1_B1_B1_A2_A1

### Relational analysis result of NS_A2_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027732
time: 1.31 seconds

## Relational analysis of NS_A2_A2_B1_B1_B1_A2_A2

### Relational analysis result of NS_A2_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027814
time: 1.30 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9886309, 0.9960932, 0.9887598, 0.9960143, -0.0053921, 0.0054402
1: -0.0040968, -0.0022374, -0.0040647, -0.0022571, -0.0013436, 0.0013555
2: 0.0018032, 0.0116571, 0.0019074, 0.0114869, -0.0071837, 0.0071203
3: -0.0065789, -0.0020939, -0.0065014, -0.0021413, -0.0032408, 0.0032697
4: 0.0008769, 0.0027841, 0.0008971, 0.0027511, -0.0013904, 0.0013781
5: 0.0012274, 0.0136211, 0.0013585, 0.0134069, -0.0090352, 0.0089554
6: -0.0019163, 0.0012293, -0.0018620, 0.0011960, -0.0022730, 0.0022932
7: -0.0080958, 0.0000430, -0.0079551, -0.0000431, -0.0058809, 0.0059333
8: -0.0038216, 0.0004584, -0.0037477, 0.0004132, -0.0030927, 0.0031203
9: -0.0023954, 0.0025675, -0.0023430, 0.0024818, -0.0036181, 0.0035861

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A2_B1_B1_B2_A1_A1

### Relational analysis result of NS_A2_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027340
time: 1.27 seconds

## Relational analysis of NS_A2_A2_B1_B1_B2_A1_A2

### Relational analysis result of NS_A2_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027692
time: 1.67 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B2_A2

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

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A2_B1_B1_B2_A2_A1

### Relational analysis result of NS_A2_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027793
time: 1.28 seconds

## Relational analysis of NS_A2_A2_B1_B1_B2_A2_A2

### Relational analysis result of NS_A2_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026002, upper bound: 0.0027880
time: 1.58 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9886565, 0.9959455, 0.9884154, 0.9955269, -0.0056756, 0.0062058
1: -0.0040905, -0.0022742, -0.0041506, -0.0023786, -0.0014142, 0.0015463
2: 0.0019982, 0.0116233, 0.0025511, 0.0119418, -0.0081947, 0.0074946
3: -0.0065635, -0.0021826, -0.0067085, -0.0024343, -0.0034112, 0.0037299
4: 0.0009146, 0.0027775, 0.0010216, 0.0028392, -0.0015861, 0.0014506
5: 0.0014726, 0.0135785, 0.0021680, 0.0139791, -0.0103068, 0.0094263
6: -0.0019055, 0.0011671, -0.0020072, 0.0009906, -0.0023925, 0.0026160
7: -0.0080678, -0.0001181, -0.0083309, -0.0005747, -0.0061901, 0.0067683
8: -0.0038069, 0.0003738, -0.0039453, 0.0001336, -0.0032553, 0.0035594
9: -0.0022972, 0.0025505, -0.0020188, 0.0027109, -0.0041273, 0.0037747

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 220

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0026879
time: 1.76 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025882, upper bound: 0.0027731
time: 1.76 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.89 + 598.39 = 602.28 seconds
