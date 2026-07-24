## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0012634299999999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0028880, 0.0028880)
1: (-0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0007196, 0.0007196)
2: (0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0038136, 0.0038136)
3: (-0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0017358, 0.0017358)
4: (0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0007381, 0.0007381)
5: (0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0047965, 0.0047965)
6: (-0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0012174, 0.0012174)
7: (-0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0031498, 0.0031498)
8: (-0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0016564, 0.0016564)
9: (0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0019207, 0.0019207)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.41 + 2.00 = 3.41 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0018048, upper bound: 0.0018049

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017293, upper bound: 0.0016839
time: 1.14 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017299, upper bound: 0.0017299
time: 1.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.42 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.42
Output dim: 0, lower bound: -0.0017293, upper bound: 0.0016839
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.42
Output dim: 0, lower bound: -0.0017299, upper bound: 0.0017299

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.9886956, 0.9919086, 0.9887278, 0.9921036, -0.0027227, 0.0025625
1: -0.0040807, -0.0032801, -0.0040727, -0.0032315, -0.0006784, 0.0006385
2: 0.0073289, 0.0115717, 0.0070715, 0.0115292, -0.0033837, 0.0035953
3: -0.0065401, -0.0046089, -0.0065207, -0.0044917, -0.0016364, 0.0015401
4: 0.0019464, 0.0027676, 0.0018966, 0.0027593, -0.0006549, 0.0006959
5: 0.0081772, 0.0135137, 0.0078535, 0.0134601, -0.0042558, 0.0045220
6: -0.0018891, -0.0005346, -0.0018755, -0.0004525, -0.0011477, 0.0010802
7: -0.0080252, -0.0045209, -0.0079901, -0.0043083, -0.0029695, 0.0027947
8: -0.0037845, -0.0019416, -0.0037661, -0.0018298, -0.0015616, 0.0014697
9: 0.0003876, 0.0025245, 0.0002579, 0.0025031, -0.0017042, 0.0018108

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015840, upper bound: 0.0015082
time: 1.11 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015818, upper bound: 0.0015214
time: 1.10 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.9887308, 0.9921090, 0.9887217, 0.9921960, -0.0028544, 0.0025981
1: -0.0040719, -0.0032302, -0.0040742, -0.0032085, -0.0007112, 0.0006474
2: 0.0070643, 0.0115251, 0.0069494, 0.0115372, -0.0034308, 0.0037692
3: -0.0065189, -0.0044885, -0.0065244, -0.0044362, -0.0017156, 0.0015616
4: 0.0018952, 0.0027585, 0.0018729, 0.0027609, -0.0006640, 0.0007295
5: 0.0078445, 0.0134550, 0.0077000, 0.0134702, -0.0043151, 0.0047406
6: -0.0018742, -0.0004502, -0.0018781, -0.0004135, -0.0012032, 0.0010952
7: -0.0079867, -0.0043024, -0.0079967, -0.0042075, -0.0031131, 0.0028336
8: -0.0037643, -0.0018267, -0.0037695, -0.0017768, -0.0016371, 0.0014902
9: 0.0002543, 0.0025010, 0.0001965, 0.0025071, -0.0017279, 0.0018984

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016839, upper bound: 0.0017293
time: 1.20 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016839, upper bound: 0.0017299
time: 1.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.76 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.76
Output dim: 0, lower bound: -0.0015840, upper bound: 0.0015082
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.76
Output dim: 0, lower bound: -0.0015818, upper bound: 0.0015214
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.76
Output dim: 0, lower bound: -0.0016839, upper bound: 0.0017293
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.76
Output dim: 0, lower bound: -0.0016839, upper bound: 0.0017299

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.9886956, 0.9919086, 0.9888329, 0.9921016, -0.0027180, 0.0024552
1: -0.0040807, -0.0032801, -0.0040465, -0.0032321, -0.0006773, 0.0006118
2: 0.0073289, 0.0115717, 0.0070742, 0.0113904, -0.0032421, 0.0035891
3: -0.0065401, -0.0046089, -0.0064575, -0.0044930, -0.0016336, 0.0014757
4: 0.0019464, 0.0027676, 0.0018971, 0.0027325, -0.0006275, 0.0006947
5: 0.0081772, 0.0135137, 0.0078570, 0.0132855, -0.0040777, 0.0045142
6: -0.0018891, -0.0005346, -0.0018312, -0.0004533, -0.0011457, 0.0010350
7: -0.0080252, -0.0045209, -0.0078754, -0.0043106, -0.0029644, 0.0026778
8: -0.0037845, -0.0019416, -0.0037058, -0.0018310, -0.0015589, 0.0014082
9: 0.0003876, 0.0025245, 0.0002593, 0.0024332, -0.0016329, 0.0018077

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015688, upper bound: 0.0015068
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015688, upper bound: 0.0015068
time: 1.43 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.9887743, 0.9919061, 0.9889213, 0.9924843, -0.0030732, 0.0025869
1: -0.0040611, -0.0032807, -0.0040245, -0.0031367, -0.0007658, 0.0006446
2: 0.0073322, 0.0114677, 0.0065688, 0.0112736, -0.0034159, 0.0040581
3: -0.0064927, -0.0046104, -0.0064044, -0.0042629, -0.0018471, 0.0015548
4: 0.0019470, 0.0027474, 0.0017993, 0.0027099, -0.0006611, 0.0007854
5: 0.0081814, 0.0133828, 0.0072212, 0.0131387, -0.0042964, 0.0051040
6: -0.0018559, -0.0005357, -0.0017939, -0.0002920, -0.0012955, 0.0010905
7: -0.0079393, -0.0045237, -0.0077790, -0.0038931, -0.0033517, 0.0028214
8: -0.0037394, -0.0019431, -0.0036551, -0.0016115, -0.0017626, 0.0014837
9: 0.0003893, 0.0024721, 0.0000047, 0.0023744, -0.0017204, 0.0020439

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012713, upper bound: 0.0009346
time: 0.94 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015102, upper bound: 0.0014520
time: 1.15 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.9887308, 0.9921090, 0.9886956, 0.9919086, -0.0025591, 0.0027637
1: -0.0040719, -0.0032302, -0.0040807, -0.0032801, -0.0006377, 0.0006887
2: 0.0070643, 0.0115251, 0.0073289, 0.0115717, -0.0036495, 0.0033793
3: -0.0065189, -0.0044885, -0.0065401, -0.0046089, -0.0015381, 0.0016611
4: 0.0018952, 0.0027585, 0.0019464, 0.0027676, -0.0007064, 0.0006541
5: 0.0078445, 0.0134550, 0.0081772, 0.0135137, -0.0045901, 0.0042503
6: -0.0018742, -0.0004502, -0.0018891, -0.0005346, -0.0010788, 0.0011650
7: -0.0079867, -0.0043024, -0.0080252, -0.0045209, -0.0027911, 0.0030143
8: -0.0037643, -0.0018267, -0.0037845, -0.0019416, -0.0014678, 0.0015852
9: 0.0002543, 0.0025010, 0.0003876, 0.0025245, -0.0018381, 0.0017020

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015082, upper bound: 0.0015840
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015214, upper bound: 0.0015818
time: 1.31 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.9887308, 0.9921090, 0.9887308, 0.9921090, -0.0025880, 0.0025880
1: -0.0040719, -0.0032302, -0.0040719, -0.0032302, -0.0006449, 0.0006449
2: 0.0070643, 0.0115251, 0.0070643, 0.0115251, -0.0034175, 0.0034175
3: -0.0065189, -0.0044885, -0.0065189, -0.0044885, -0.0015555, 0.0015555
4: 0.0018952, 0.0027585, 0.0018952, 0.0027585, -0.0006614, 0.0006614
5: 0.0078445, 0.0134550, 0.0078445, 0.0134550, -0.0042983, 0.0042983
6: -0.0018742, -0.0004502, -0.0018742, -0.0004502, -0.0010910, 0.0010910
7: -0.0079867, -0.0043024, -0.0079867, -0.0043024, -0.0028226, 0.0028226
8: -0.0037643, -0.0018267, -0.0037643, -0.0018267, -0.0014844, 0.0014844
9: 0.0002543, 0.0025010, 0.0002543, 0.0025010, -0.0017212, 0.0017212

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015082, upper bound: 0.0015988
time: 1.64 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015214, upper bound: 0.0015988
time: 1.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.26 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.26
Output dim: 0, lower bound: -0.0015688, upper bound: 0.0015068
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.26
Output dim: 0, lower bound: -0.0015688, upper bound: 0.0015068
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.26
Output dim: 0, lower bound: -0.0012713, upper bound: 0.0009346
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.26
Output dim: 0, lower bound: -0.0015102, upper bound: 0.0014520
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.26
Output dim: 0, lower bound: -0.0015082, upper bound: 0.0015840
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.26
Output dim: 0, lower bound: -0.0015214, upper bound: 0.0015818
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.26
Output dim: 0, lower bound: -0.0015082, upper bound: 0.0015988
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.26
Output dim: 0, lower bound: -0.0015214, upper bound: 0.0015988

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9888023, 0.9919065, 0.9888329, 0.9921016, -0.0025941, 0.0024504
1: -0.0040541, -0.0032806, -0.0040465, -0.0032321, -0.0006464, 0.0006106
2: 0.0073317, 0.0114307, 0.0070742, 0.0113904, -0.0032357, 0.0034254
3: -0.0064759, -0.0046102, -0.0064575, -0.0044930, -0.0015591, 0.0014728
4: 0.0019469, 0.0027403, 0.0018971, 0.0027325, -0.0006263, 0.0006630
5: 0.0081808, 0.0133363, 0.0078570, 0.0132855, -0.0040697, 0.0043083
6: -0.0018441, -0.0005355, -0.0018312, -0.0004533, -0.0010935, 0.0010329
7: -0.0079088, -0.0045232, -0.0078754, -0.0043106, -0.0028292, 0.0026725
8: -0.0037233, -0.0019429, -0.0037058, -0.0018310, -0.0014878, 0.0014054
9: 0.0003890, 0.0024535, 0.0002593, 0.0024332, -0.0016297, 0.0017252

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014001, upper bound: 0.0011018
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015051, upper bound: 0.0014321
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888738, 0.9923024, 0.9888329, 0.9921016, -0.0025845, 0.0028992
1: -0.0040363, -0.0031820, -0.0040465, -0.0032321, -0.0006440, 0.0007224
2: 0.0068090, 0.0113364, 0.0070742, 0.0113904, -0.0038284, 0.0034129
3: -0.0064329, -0.0043723, -0.0064575, -0.0044930, -0.0015534, 0.0017425
4: 0.0018457, 0.0027220, 0.0018971, 0.0027325, -0.0007410, 0.0006606
5: 0.0075233, 0.0132176, 0.0078570, 0.0132855, -0.0048151, 0.0042925
6: -0.0018139, -0.0003687, -0.0018312, -0.0004533, -0.0010895, 0.0012221
7: -0.0078308, -0.0040915, -0.0078754, -0.0043106, -0.0028188, 0.0031620
8: -0.0036823, -0.0017158, -0.0037058, -0.0018310, -0.0014824, 0.0016629
9: 0.0001257, 0.0024060, 0.0002593, 0.0024332, -0.0019282, 0.0017189

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014001, upper bound: 0.0011018
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015051, upper bound: 0.0014321
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.9889210, 0.9918966, 0.9894004, 0.9926941, -0.0028965, 0.0020758
1: -0.0040245, -0.0032831, -0.0039051, -0.0030844, -0.0007217, 0.0005172
2: 0.0073448, 0.0112740, 0.0062915, 0.0106410, -0.0027411, 0.0038249
3: -0.0064045, -0.0046162, -0.0061164, -0.0041368, -0.0017409, 0.0012476
4: 0.0019495, 0.0027099, 0.0017456, 0.0025874, -0.0005305, 0.0007403
5: 0.0081973, 0.0131391, 0.0068726, 0.0123430, -0.0034476, 0.0048107
6: -0.0017940, -0.0005397, -0.0015920, -0.0002035, -0.0012210, 0.0008750
7: -0.0077793, -0.0045340, -0.0072565, -0.0036641, -0.0031591, 0.0022640
8: -0.0036552, -0.0019486, -0.0033803, -0.0014911, -0.0016613, 0.0011906
9: 0.0003956, 0.0023745, -0.0001349, 0.0020557, -0.0013806, 0.0019264

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012713, upper bound: 0.0009346
time: 0.86 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012713, upper bound: 0.0009346
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.9887743, 0.9919061, 0.9889641, 0.9924816, -0.0030598, 0.0021905
1: -0.0040611, -0.0032807, -0.0040138, -0.0031373, -0.0007624, 0.0005458
2: 0.0073322, 0.0114677, 0.0065722, 0.0112170, -0.0028925, 0.0040405
3: -0.0064927, -0.0046104, -0.0063786, -0.0042645, -0.0018390, 0.0013165
4: 0.0019470, 0.0027474, 0.0017999, 0.0026989, -0.0005598, 0.0007820
5: 0.0081814, 0.0133828, 0.0072256, 0.0130675, -0.0036380, 0.0050819
6: -0.0018559, -0.0005357, -0.0017758, -0.0002931, -0.0012898, 0.0009234
7: -0.0079393, -0.0045237, -0.0077322, -0.0038960, -0.0033372, 0.0023890
8: -0.0037394, -0.0019431, -0.0036305, -0.0016130, -0.0017550, 0.0012564
9: 0.0003893, 0.0024721, 0.0000065, 0.0023458, -0.0014568, 0.0020350

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014520, upper bound: 0.0014520
time: 1.12 seconds

## Relational analysis of NS_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014520, upper bound: 0.0014520
time: 1.17 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9888360, 0.9921069, 0.9886956, 0.9919086, -0.0024519, 0.0027589
1: -0.0040457, -0.0032307, -0.0040807, -0.0032801, -0.0006109, 0.0006875
2: 0.0070670, 0.0113862, 0.0073289, 0.0115717, -0.0036431, 0.0032376
3: -0.0064556, -0.0044897, -0.0065401, -0.0046089, -0.0014736, 0.0016582
4: 0.0018957, 0.0027317, 0.0019464, 0.0027676, -0.0007051, 0.0006266
5: 0.0078479, 0.0132802, 0.0081772, 0.0135137, -0.0045821, 0.0040721
6: -0.0018298, -0.0004511, -0.0018891, -0.0005346, -0.0010335, 0.0011630
7: -0.0078720, -0.0043047, -0.0080252, -0.0045209, -0.0026741, 0.0030090
8: -0.0037039, -0.0018279, -0.0037845, -0.0019416, -0.0014063, 0.0015824
9: 0.0002557, 0.0024310, 0.0003876, 0.0025245, -0.0018349, 0.0016306

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015688
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015818
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9889241, 0.9924735, 0.9887743, 0.9919061, -0.0025835, 0.0030779
1: -0.0040238, -0.0031393, -0.0040611, -0.0032807, -0.0006437, 0.0007669
2: 0.0065828, 0.0112698, 0.0073322, 0.0114677, -0.0040643, 0.0034114
3: -0.0064027, -0.0042693, -0.0064927, -0.0046104, -0.0015527, 0.0018499
4: 0.0018020, 0.0027091, 0.0019470, 0.0027474, -0.0007866, 0.0006603
5: 0.0072389, 0.0131340, 0.0081814, 0.0133828, -0.0051118, 0.0042907
6: -0.0017927, -0.0002965, -0.0018559, -0.0005357, -0.0010890, 0.0012974
7: -0.0077759, -0.0039047, -0.0079393, -0.0045237, -0.0028176, 0.0033569
8: -0.0036534, -0.0016176, -0.0037394, -0.0019431, -0.0014818, 0.0017653
9: 0.0000118, 0.0023725, 0.0003893, 0.0024721, -0.0020470, 0.0017182

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009346, upper bound: 0.0012713
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014520, upper bound: 0.0015102
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9888360, 0.9921069, 0.9887308, 0.9921090, -0.0024657, 0.0025833
1: -0.0040457, -0.0032307, -0.0040719, -0.0032302, -0.0006144, 0.0006437
2: 0.0070670, 0.0113862, 0.0070643, 0.0115251, -0.0034112, 0.0032559
3: -0.0064556, -0.0044897, -0.0065189, -0.0044885, -0.0014819, 0.0015526
4: 0.0018957, 0.0027317, 0.0018952, 0.0027585, -0.0006602, 0.0006302
5: 0.0078479, 0.0132802, 0.0078445, 0.0134550, -0.0042904, 0.0040951
6: -0.0018298, -0.0004511, -0.0018742, -0.0004502, -0.0010394, 0.0010889
7: -0.0078720, -0.0043047, -0.0079867, -0.0043024, -0.0026892, 0.0028174
8: -0.0037039, -0.0018279, -0.0037643, -0.0018267, -0.0014142, 0.0014817
9: 0.0002557, 0.0024310, 0.0002543, 0.0025010, -0.0017180, 0.0016398

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015315, upper bound: 0.0015788
time: 1.21 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015315, upper bound: 0.0015988
time: 1.26 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9889241, 0.9924735, 0.9888129, 0.9921065, -0.0025758, 0.0029623
1: -0.0040238, -0.0031393, -0.0040515, -0.0032308, -0.0006418, 0.0007381
2: 0.0065828, 0.0112698, 0.0070676, 0.0114167, -0.0039117, 0.0034013
3: -0.0064027, -0.0042693, -0.0064695, -0.0044900, -0.0015481, 0.0017804
4: 0.0018020, 0.0027091, 0.0018958, 0.0027376, -0.0007571, 0.0006583
5: 0.0072389, 0.0131340, 0.0078486, 0.0133186, -0.0049198, 0.0042780
6: -0.0017927, -0.0002965, -0.0018396, -0.0004512, -0.0010858, 0.0012487
7: -0.0077759, -0.0039047, -0.0078972, -0.0043051, -0.0028093, 0.0032308
8: -0.0036534, -0.0016176, -0.0037172, -0.0018281, -0.0014774, 0.0016990
9: 0.0000118, 0.0023725, 0.0002560, 0.0024464, -0.0019701, 0.0017131

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013376, upper bound: 0.0011768
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014852, upper bound: 0.0015255
time: 1.14 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.56 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0014001, upper bound: 0.0011018
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0015051, upper bound: 0.0014321
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0014001, upper bound: 0.0011018
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0015051, upper bound: 0.0014321
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0012713, upper bound: 0.0009346
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0012713, upper bound: 0.0009346
NS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0014520, upper bound: 0.0014520
NS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0014520, upper bound: 0.0014520
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015688
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015818
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0009346, upper bound: 0.0012713
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0014520, upper bound: 0.0015102
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0015315, upper bound: 0.0015788
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0015315, upper bound: 0.0015988
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0013376, upper bound: 0.0011768
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -0.0014852, upper bound: 0.0015255

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9889489, 0.9918969, 0.9892772, 0.9923428, -0.0023795, 0.0019246
1: -0.0040176, -0.0032830, -0.0039358, -0.0031719, -0.0005929, 0.0004796
2: 0.0073442, 0.0112372, 0.0067555, 0.0108036, -0.0025414, 0.0031422
3: -0.0063878, -0.0046159, -0.0061905, -0.0043479, -0.0014302, 0.0011568
4: 0.0019494, 0.0027028, 0.0018354, 0.0026189, -0.0004919, 0.0006082
5: 0.0081966, 0.0130929, 0.0074561, 0.0125475, -0.0031965, 0.0039520
6: -0.0017823, -0.0005395, -0.0016439, -0.0003516, -0.0010031, 0.0008113
7: -0.0077489, -0.0045336, -0.0073908, -0.0040474, -0.0025952, 0.0020991
8: -0.0036392, -0.0019483, -0.0034509, -0.0016926, -0.0013648, 0.0011039
9: 0.0003953, 0.0023560, 0.0000988, 0.0021376, -0.0012800, 0.0015826

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014916, upper bound: 0.0014275
time: 1.38 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014916, upper bound: 0.0014275
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9888023, 0.9919065, 0.9888773, 0.9920990, -0.0025793, 0.0020247
1: -0.0040541, -0.0032806, -0.0040354, -0.0032327, -0.0006427, 0.0005045
2: 0.0073317, 0.0114307, 0.0070776, 0.0113316, -0.0026735, 0.0034059
3: -0.0064759, -0.0046102, -0.0064308, -0.0044945, -0.0015502, 0.0012169
4: 0.0019469, 0.0027403, 0.0018977, 0.0027211, -0.0005175, 0.0006592
5: 0.0081808, 0.0133363, 0.0078612, 0.0132117, -0.0033626, 0.0042837
6: -0.0018441, -0.0005355, -0.0018124, -0.0004544, -0.0010873, 0.0008535
7: -0.0079088, -0.0045232, -0.0078269, -0.0043133, -0.0028130, 0.0022082
8: -0.0037233, -0.0019429, -0.0036803, -0.0018325, -0.0014794, 0.0011613
9: 0.0003890, 0.0024535, 0.0002610, 0.0024036, -0.0013465, 0.0017154

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015003, upper bound: 0.0014996
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015003, upper bound: 0.0014996
time: 1.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9890224, 0.9922927, 0.9892772, 0.9923428, -0.0023809, 0.0023710
1: -0.0039993, -0.0031844, -0.0039358, -0.0031719, -0.0005932, 0.0005908
2: 0.0068217, 0.0111402, 0.0067555, 0.0108036, -0.0031308, 0.0031439
3: -0.0063436, -0.0043781, -0.0061905, -0.0043479, -0.0014310, 0.0014250
4: 0.0018482, 0.0026840, 0.0018354, 0.0026189, -0.0006060, 0.0006085
5: 0.0075394, 0.0129709, 0.0074561, 0.0125475, -0.0039378, 0.0039542
6: -0.0017513, -0.0003727, -0.0016439, -0.0003516, -0.0010036, 0.0009994
7: -0.0076688, -0.0041020, -0.0073908, -0.0040474, -0.0025967, 0.0025859
8: -0.0035971, -0.0017214, -0.0034509, -0.0016926, -0.0013656, 0.0013599
9: 0.0001321, 0.0023072, 0.0000988, 0.0021376, -0.0015768, 0.0015834

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013464, upper bound: 0.0011018
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013464, upper bound: 0.0011018
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888738, 0.9923024, 0.9888773, 0.9920990, -0.0025697, 0.0025213
1: -0.0040363, -0.0031820, -0.0040354, -0.0032327, -0.0006403, 0.0006283
2: 0.0068090, 0.0113364, 0.0070776, 0.0113316, -0.0033294, 0.0033933
3: -0.0064329, -0.0043723, -0.0064308, -0.0044945, -0.0015445, 0.0015154
4: 0.0018457, 0.0027220, 0.0018977, 0.0027211, -0.0006444, 0.0006568
5: 0.0075233, 0.0132176, 0.0078612, 0.0132117, -0.0041875, 0.0042679
6: -0.0018139, -0.0003687, -0.0018124, -0.0004544, -0.0010832, 0.0010628
7: -0.0078308, -0.0040915, -0.0078269, -0.0043133, -0.0028027, 0.0027499
8: -0.0036823, -0.0017158, -0.0036803, -0.0018325, -0.0014739, 0.0014461
9: 0.0001257, 0.0024060, 0.0002610, 0.0024036, -0.0016769, 0.0017090

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014565, upper bound: 0.0014321
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014565, upper bound: 0.0014321
time: 1.51 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9889489, 0.9918969, 0.9894004, 0.9926941, -0.0028392, 0.0018100
1: -0.0040176, -0.0032830, -0.0039051, -0.0030844, -0.0007074, 0.0004510
2: 0.0073442, 0.0112372, 0.0062915, 0.0106410, -0.0023901, 0.0037491
3: -0.0063878, -0.0046159, -0.0061164, -0.0041368, -0.0017064, 0.0010879
4: 0.0019494, 0.0027028, 0.0017456, 0.0025874, -0.0004626, 0.0007256
5: 0.0081966, 0.0130929, 0.0068726, 0.0123430, -0.0030061, 0.0047154
6: -0.0017823, -0.0005395, -0.0015920, -0.0002035, -0.0011968, 0.0007630
7: -0.0077489, -0.0045336, -0.0072565, -0.0036641, -0.0030965, 0.0019741
8: -0.0036392, -0.0019483, -0.0033803, -0.0014911, -0.0016284, 0.0010381
9: 0.0003953, 0.0023560, -0.0001349, 0.0020557, -0.0012038, 0.0018882

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 71

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012375, upper bound: 0.0009096
time: 0.89 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012386, upper bound: 0.0009096
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9890224, 0.9922927, 0.9894004, 0.9926941, -0.0025141, 0.0020725
1: -0.0039993, -0.0031844, -0.0039051, -0.0030844, -0.0006264, 0.0005164
2: 0.0068217, 0.0111402, 0.0062915, 0.0106410, -0.0027368, 0.0033198
3: -0.0063436, -0.0043781, -0.0061164, -0.0041368, -0.0015110, 0.0012457
4: 0.0018482, 0.0026840, 0.0017456, 0.0025874, -0.0005297, 0.0006425
5: 0.0075394, 0.0129709, 0.0068726, 0.0123430, -0.0034421, 0.0041754
6: -0.0017513, -0.0003727, -0.0015920, -0.0002035, -0.0010598, 0.0008736
7: -0.0076688, -0.0041020, -0.0072565, -0.0036641, -0.0027419, 0.0022604
8: -0.0035971, -0.0017214, -0.0033803, -0.0014911, -0.0014420, 0.0011887
9: 0.0001321, 0.0023072, -0.0001349, 0.0020557, -0.0013784, 0.0016720

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 71

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012375, upper bound: 0.0009096
time: 0.88 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012386, upper bound: 0.0009096
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9887743, 0.9919061, 0.9889178, 0.9922998, -0.0028964, 0.0021026
1: -0.0040611, -0.0032807, -0.0040254, -0.0031827, -0.0007217, 0.0005239
2: 0.0073322, 0.0114677, 0.0068124, 0.0112783, -0.0027764, 0.0038246
3: -0.0064927, -0.0046104, -0.0064065, -0.0043738, -0.0017408, 0.0012637
4: 0.0019470, 0.0027474, 0.0018464, 0.0027108, -0.0005374, 0.0007402
5: 0.0081814, 0.0133828, 0.0075277, 0.0131446, -0.0034920, 0.0048104
6: -0.0018559, -0.0005357, -0.0017954, -0.0003698, -0.0012209, 0.0008863
7: -0.0079393, -0.0045237, -0.0077829, -0.0040943, -0.0031589, 0.0022931
8: -0.0037394, -0.0019431, -0.0036571, -0.0017173, -0.0016612, 0.0012059
9: 0.0003893, 0.0024721, 0.0001275, 0.0023767, -0.0013983, 0.0019263

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of NS_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014317, upper bound: 0.0014520
time: 1.13 seconds

## Relational analysis of NS_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014317, upper bound: 0.0014317
time: 1.47 seconds

## BFS NS instance: NS_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9887743, 0.9919061, 0.9889671, 0.9924710, -0.0030647, 0.0021873
1: -0.0040611, -0.0032807, -0.0040131, -0.0031400, -0.0007636, 0.0005450
2: 0.0073322, 0.0114677, 0.0065863, 0.0112132, -0.0028883, 0.0040468
3: -0.0064927, -0.0046104, -0.0063769, -0.0042709, -0.0018419, 0.0013146
4: 0.0019470, 0.0027474, 0.0018026, 0.0026982, -0.0005590, 0.0007833
5: 0.0081814, 0.0133828, 0.0072432, 0.0130627, -0.0036327, 0.0050899
6: -0.0018559, -0.0005357, -0.0017746, -0.0002976, -0.0012919, 0.0009220
7: -0.0079393, -0.0045237, -0.0077291, -0.0039075, -0.0033424, 0.0023856
8: -0.0037394, -0.0019431, -0.0036288, -0.0016191, -0.0017578, 0.0012545
9: 0.0003893, 0.0024721, 0.0000136, 0.0023439, -0.0014547, 0.0020382

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of NS_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014317, upper bound: 0.0014520
time: 1.47 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014317, upper bound: 0.0014317
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9888360, 0.9921069, 0.9888023, 0.9919065, -0.0024470, 0.0026350
1: -0.0040457, -0.0032307, -0.0040541, -0.0032806, -0.0006097, 0.0006566
2: 0.0070670, 0.0113862, 0.0073317, 0.0114307, -0.0034795, 0.0032313
3: -0.0064556, -0.0044897, -0.0064759, -0.0046102, -0.0014707, 0.0015837
4: 0.0018957, 0.0027317, 0.0019469, 0.0027403, -0.0006734, 0.0006254
5: 0.0078479, 0.0132802, 0.0081808, 0.0133363, -0.0043762, 0.0040641
6: -0.0018298, -0.0004511, -0.0018441, -0.0005355, -0.0010315, 0.0011107
7: -0.0078720, -0.0043047, -0.0079088, -0.0045232, -0.0026688, 0.0028738
8: -0.0037039, -0.0018279, -0.0037233, -0.0019429, -0.0014035, 0.0015113
9: 0.0002557, 0.0024310, 0.0003890, 0.0024535, -0.0017524, 0.0016274

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011018, upper bound: 0.0014001
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014321, upper bound: 0.0015051
time: 1.12 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9888360, 0.9921069, 0.9888738, 0.9923024, -0.0028958, 0.0026254
1: -0.0040457, -0.0032307, -0.0040363, -0.0031820, -0.0007216, 0.0006542
2: 0.0070670, 0.0113862, 0.0068090, 0.0113364, -0.0034669, 0.0038239
3: -0.0064556, -0.0044897, -0.0064329, -0.0043723, -0.0017405, 0.0015780
4: 0.0018957, 0.0027317, 0.0018457, 0.0027220, -0.0006710, 0.0007401
5: 0.0078479, 0.0132802, 0.0075233, 0.0132176, -0.0043604, 0.0048095
6: -0.0018298, -0.0004511, -0.0018139, -0.0003687, -0.0012207, 0.0011067
7: -0.0078720, -0.0043047, -0.0078308, -0.0040915, -0.0031583, 0.0028634
8: -0.0037039, -0.0018279, -0.0036823, -0.0017158, -0.0016609, 0.0015058
9: 0.0002557, 0.0024310, 0.0001257, 0.0024060, -0.0017461, 0.0019259

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011018, upper bound: 0.0014001
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014321, upper bound: 0.0015111
time: 1.28 seconds

## BFS NS instance: NS_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.9894029, 0.9926836, 0.9889210, 0.9918966, -0.0020729, 0.0029348
1: -0.0039045, -0.0030870, -0.0040245, -0.0032831, -0.0005165, 0.0007313
2: 0.0063055, 0.0106377, 0.0073448, 0.0112740, -0.0038754, 0.0027372
3: -0.0061149, -0.0041431, -0.0064045, -0.0046162, -0.0012458, 0.0017639
4: 0.0017483, 0.0025868, 0.0019495, 0.0027099, -0.0007501, 0.0005298
5: 0.0068902, 0.0123389, 0.0081973, 0.0131391, -0.0048742, 0.0034427
6: -0.0015909, -0.0002080, -0.0017940, -0.0005397, -0.0008738, 0.0012371
7: -0.0072538, -0.0036757, -0.0077793, -0.0045340, -0.0022607, 0.0032008
8: -0.0033788, -0.0014972, -0.0036552, -0.0019486, -0.0011889, 0.0016833
9: -0.0001278, 0.0020541, 0.0003956, 0.0023745, -0.0019518, 0.0013786

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009346, upper bound: 0.0012713
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009346, upper bound: 0.0012713
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: 0.9889671, 0.9924710, 0.9887743, 0.9919061, -0.0021873, 0.0030647
1: -0.0040131, -0.0031400, -0.0040611, -0.0032807, -0.0005450, 0.0007636
2: 0.0065863, 0.0112132, 0.0073322, 0.0114677, -0.0040468, 0.0028883
3: -0.0063769, -0.0042709, -0.0064927, -0.0046104, -0.0013146, 0.0018419
4: 0.0018026, 0.0026982, 0.0019470, 0.0027474, -0.0007833, 0.0005590
5: 0.0072432, 0.0130627, 0.0081814, 0.0133828, -0.0050899, 0.0036327
6: -0.0017746, -0.0002976, -0.0018559, -0.0005357, -0.0009220, 0.0012919
7: -0.0077291, -0.0039075, -0.0079393, -0.0045237, -0.0023856, 0.0033424
8: -0.0036288, -0.0016191, -0.0037394, -0.0019431, -0.0012545, 0.0017578
9: 0.0000136, 0.0023439, 0.0003893, 0.0024721, -0.0020382, 0.0014547

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014520, upper bound: 0.0014924
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014520, upper bound: 0.0014924
time: 1.27 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9888360, 0.9921069, 0.9888360, 0.9921069, -0.0024609, 0.0024609
1: -0.0040457, -0.0032307, -0.0040457, -0.0032307, -0.0006132, 0.0006132
2: 0.0070670, 0.0113862, 0.0070670, 0.0113862, -0.0032496, 0.0032496
3: -0.0064556, -0.0044897, -0.0064556, -0.0044897, -0.0014791, 0.0014791
4: 0.0018957, 0.0027317, 0.0018957, 0.0027317, -0.0006290, 0.0006290
5: 0.0078479, 0.0132802, 0.0078479, 0.0132802, -0.0040871, 0.0040871
6: -0.0018298, -0.0004511, -0.0018298, -0.0004511, -0.0010374, 0.0010374
7: -0.0078720, -0.0043047, -0.0078720, -0.0043047, -0.0026840, 0.0026840
8: -0.0037039, -0.0018279, -0.0037039, -0.0018279, -0.0014115, 0.0014115
9: 0.0002557, 0.0024310, 0.0002557, 0.0024310, -0.0016367, 0.0016367

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012036, upper bound: 0.0014415
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014525, upper bound: 0.0015106
time: 1.38 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9888360, 0.9921069, 0.9889241, 0.9924735, -0.0029103, 0.0024474
1: -0.0040457, -0.0032307, -0.0040238, -0.0031393, -0.0007252, 0.0006098
2: 0.0070670, 0.0113862, 0.0065828, 0.0112698, -0.0032317, 0.0038430
3: -0.0064556, -0.0044897, -0.0064027, -0.0042693, -0.0017492, 0.0014709
4: 0.0018957, 0.0027317, 0.0018020, 0.0027091, -0.0006255, 0.0007438
5: 0.0078479, 0.0132802, 0.0072389, 0.0131340, -0.0040647, 0.0048335
6: -0.0018298, -0.0004511, -0.0017927, -0.0002965, -0.0012268, 0.0010317
7: -0.0078720, -0.0043047, -0.0077759, -0.0039047, -0.0031741, 0.0026692
8: -0.0037039, -0.0018279, -0.0036534, -0.0016176, -0.0016692, 0.0014037
9: 0.0002557, 0.0024310, 0.0000118, 0.0023725, -0.0016277, 0.0019356

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012036, upper bound: 0.0014415
time: 1.09 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014525, upper bound: 0.0015255
time: 1.26 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9890811, 0.9924639, 0.9892803, 0.9923530, -0.0023261, 0.0023537
1: -0.0039846, -0.0031417, -0.0039350, -0.0031694, -0.0005796, 0.0005865
2: 0.0065956, 0.0110625, 0.0067422, 0.0107996, -0.0031081, 0.0030716
3: -0.0063083, -0.0042751, -0.0061886, -0.0043419, -0.0013981, 0.0014147
4: 0.0018045, 0.0026690, 0.0018328, 0.0026181, -0.0006016, 0.0005945
5: 0.0072550, 0.0128732, 0.0074393, 0.0125425, -0.0039091, 0.0038633
6: -0.0017265, -0.0003006, -0.0016426, -0.0003473, -0.0009805, 0.0009922
7: -0.0076046, -0.0039153, -0.0073875, -0.0040363, -0.0025370, 0.0025671
8: -0.0035634, -0.0016231, -0.0034492, -0.0016868, -0.0013342, 0.0013500
9: 0.0000183, 0.0022680, 0.0000921, 0.0021356, -0.0015654, 0.0015470

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013376, upper bound: 0.0011768
time: 1.09 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013376, upper bound: 0.0011768
time: 1.07 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9889241, 0.9924735, 0.9888560, 0.9921041, -0.0025606, 0.0025015
1: -0.0040238, -0.0031393, -0.0040408, -0.0032314, -0.0006380, 0.0006233
2: 0.0065828, 0.0112698, 0.0070708, 0.0113599, -0.0033032, 0.0033812
3: -0.0064027, -0.0042693, -0.0064436, -0.0044915, -0.0015390, 0.0015035
4: 0.0018020, 0.0027091, 0.0018964, 0.0027266, -0.0006393, 0.0006544
5: 0.0072389, 0.0131340, 0.0078527, 0.0132472, -0.0041545, 0.0042527
6: -0.0017927, -0.0002965, -0.0018214, -0.0004523, -0.0010794, 0.0010545
7: -0.0077759, -0.0039047, -0.0078502, -0.0043078, -0.0027927, 0.0027282
8: -0.0036534, -0.0016176, -0.0036925, -0.0018296, -0.0014686, 0.0014347
9: 0.0000118, 0.0023725, 0.0002576, 0.0024178, -0.0016636, 0.0017030

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011406, upper bound: 0.0013728
time: 1.25 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011406, upper bound: 0.0015255
time: 1.44 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.43 seconds
NS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0014916, upper bound: 0.0014275
NS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0014916, upper bound: 0.0014275
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0015003, upper bound: 0.0014996
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0015003, upper bound: 0.0014996
NS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0013464, upper bound: 0.0011018
NS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0013464, upper bound: 0.0011018
NS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0014565, upper bound: 0.0014321
NS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0014565, upper bound: 0.0014321
NS_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0012375, upper bound: 0.0009096
NS_A1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0012386, upper bound: 0.0009096
NS_A1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0012375, upper bound: 0.0009096
NS_A1_B2_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0012386, upper bound: 0.0009096
NS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0014317, upper bound: 0.0014520
NS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0014317, upper bound: 0.0014317
NS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0014317, upper bound: 0.0014520
NS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0014317, upper bound: 0.0014317
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0011018, upper bound: 0.0014001
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0014321, upper bound: 0.0015051
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0011018, upper bound: 0.0014001
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0014321, upper bound: 0.0015111
NS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0009346, upper bound: 0.0012713
NS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0009346, upper bound: 0.0012713
NS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0014520, upper bound: 0.0014924
NS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0014520, upper bound: 0.0014924
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0012036, upper bound: 0.0014415
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0014525, upper bound: 0.0015106
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0012036, upper bound: 0.0014415
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0014525, upper bound: 0.0015255
NS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0013376, upper bound: 0.0011768
NS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0013376, upper bound: 0.0011768
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0011406, upper bound: 0.0013728
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0011406, upper bound: 0.0015255

## BFS NS instance: NS_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.9889489, 0.9918969, 0.9892468, 0.9921501, -0.0021499, 0.0018235
1: -0.0040176, -0.0032830, -0.0039434, -0.0032200, -0.0005357, 0.0004544
2: 0.0073442, 0.0112372, 0.0070101, 0.0108438, -0.0024080, 0.0028390
3: -0.0063878, -0.0046159, -0.0062088, -0.0044638, -0.0012922, 0.0010960
4: 0.0019494, 0.0027028, 0.0018847, 0.0026267, -0.0004661, 0.0005495
5: 0.0081966, 0.0130929, 0.0077763, 0.0125981, -0.0030286, 0.0035707
6: -0.0017823, -0.0005395, -0.0016567, -0.0004329, -0.0009063, 0.0007687
7: -0.0077489, -0.0045336, -0.0074240, -0.0042576, -0.0023448, 0.0019888
8: -0.0036392, -0.0019483, -0.0034684, -0.0018032, -0.0012331, 0.0010459
9: 0.0003953, 0.0023560, 0.0002270, 0.0021579, -0.0012128, 0.0014298

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014275
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014275
time: 1.45 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.9889489, 0.9918969, 0.9892800, 0.9923536, -0.0025085, 0.0019218
1: -0.0040176, -0.0032830, -0.0039351, -0.0031692, -0.0006251, 0.0004789
2: 0.0073442, 0.0112372, 0.0067414, 0.0107999, -0.0025377, 0.0033125
3: -0.0063878, -0.0046159, -0.0061888, -0.0043415, -0.0015077, 0.0011551
4: 0.0019494, 0.0027028, 0.0018327, 0.0026182, -0.0004912, 0.0006411
5: 0.0081966, 0.0130929, 0.0074383, 0.0125429, -0.0031918, 0.0041662
6: -0.0017823, -0.0005395, -0.0016427, -0.0003471, -0.0010574, 0.0008101
7: -0.0077489, -0.0045336, -0.0073878, -0.0040357, -0.0027359, 0.0020960
8: -0.0036392, -0.0019483, -0.0034493, -0.0016865, -0.0014388, 0.0011023
9: 0.0003953, 0.0023560, 0.0000917, 0.0021358, -0.0012781, 0.0016683

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014275
time: 1.93 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014275
time: 1.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.9888023, 0.9919065, 0.9888493, 0.9919039, -0.0023912, 0.0019362
1: -0.0040541, -0.0032806, -0.0040424, -0.0032813, -0.0005958, 0.0004825
2: 0.0073317, 0.0114307, 0.0073351, 0.0113688, -0.0025568, 0.0031576
3: -0.0064759, -0.0046102, -0.0064477, -0.0046117, -0.0014372, 0.0011637
4: 0.0019469, 0.0027403, 0.0019476, 0.0027283, -0.0004949, 0.0006111
5: 0.0081808, 0.0133363, 0.0081850, 0.0132584, -0.0032157, 0.0039715
6: -0.0018441, -0.0005355, -0.0018243, -0.0005366, -0.0010080, 0.0008162
7: -0.0079088, -0.0045232, -0.0078576, -0.0045260, -0.0026080, 0.0021117
8: -0.0037233, -0.0019429, -0.0036964, -0.0019443, -0.0013715, 0.0011105
9: 0.0003890, 0.0024535, 0.0003907, 0.0024223, -0.0012877, 0.0015903

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014912
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014555
time: 1.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.9888023, 0.9919065, 0.9888805, 0.9921045, -0.0026208, 0.0020217
1: -0.0040541, -0.0032806, -0.0040346, -0.0032313, -0.0006530, 0.0005038
2: 0.0073317, 0.0114307, 0.0070703, 0.0113274, -0.0026697, 0.0034607
3: -0.0064759, -0.0046102, -0.0064289, -0.0044912, -0.0015752, 0.0012151
4: 0.0019469, 0.0027403, 0.0018963, 0.0027203, -0.0005167, 0.0006698
5: 0.0081808, 0.0133363, 0.0078520, 0.0132064, -0.0033577, 0.0043526
6: -0.0018441, -0.0005355, -0.0018111, -0.0004521, -0.0011047, 0.0008522
7: -0.0079088, -0.0045232, -0.0078234, -0.0043073, -0.0028583, 0.0022050
8: -0.0037233, -0.0019429, -0.0036784, -0.0018293, -0.0015032, 0.0011596
9: 0.0003890, 0.0024535, 0.0002574, 0.0024015, -0.0013446, 0.0017430

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014912
time: 1.55 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014554
time: 1.62 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.9890224, 0.9922927, 0.9892468, 0.9921501, -0.0021512, 0.0022699
1: -0.0039993, -0.0031844, -0.0039434, -0.0032200, -0.0005360, 0.0005656
2: 0.0068217, 0.0111402, 0.0070101, 0.0108438, -0.0029974, 0.0028407
3: -0.0063436, -0.0043781, -0.0062088, -0.0044638, -0.0012930, 0.0013643
4: 0.0018482, 0.0026840, 0.0018847, 0.0026267, -0.0005801, 0.0005498
5: 0.0075394, 0.0129709, 0.0077763, 0.0125981, -0.0037699, 0.0035728
6: -0.0017513, -0.0003727, -0.0016567, -0.0004329, -0.0009068, 0.0009568
7: -0.0076688, -0.0041020, -0.0074240, -0.0042576, -0.0023462, 0.0024756
8: -0.0035971, -0.0017214, -0.0034684, -0.0018032, -0.0012339, 0.0013019
9: 0.0001321, 0.0023072, 0.0002270, 0.0021579, -0.0015096, 0.0014307

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A2_B1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013063, upper bound: 0.0010764
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013169, upper bound: 0.0010764
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9890224, 0.9922927, 0.9892800, 0.9923536, -0.0025098, 0.0023681
1: -0.0039993, -0.0031844, -0.0039351, -0.0031692, -0.0006254, 0.0005901
2: 0.0068217, 0.0111402, 0.0067414, 0.0107999, -0.0031271, 0.0033142
3: -0.0063436, -0.0043781, -0.0061888, -0.0043415, -0.0015085, 0.0014233
4: 0.0018482, 0.0026840, 0.0018327, 0.0026182, -0.0006052, 0.0006415
5: 0.0075394, 0.0129709, 0.0074383, 0.0125429, -0.0039331, 0.0041684
6: -0.0017513, -0.0003727, -0.0016427, -0.0003471, -0.0010580, 0.0009983
7: -0.0076688, -0.0041020, -0.0073878, -0.0040357, -0.0027373, 0.0025828
8: -0.0035971, -0.0017214, -0.0034493, -0.0016865, -0.0014395, 0.0013583
9: 0.0001321, 0.0023072, 0.0000917, 0.0021358, -0.0015750, 0.0016692

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A2_B1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013063, upper bound: 0.0010764
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013169, upper bound: 0.0010764
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9888738, 0.9923024, 0.9888493, 0.9919039, -0.0023817, 0.0024329
1: -0.0040363, -0.0031820, -0.0040424, -0.0032813, -0.0005935, 0.0006062
2: 0.0068090, 0.0113364, 0.0073351, 0.0113688, -0.0032127, 0.0031450
3: -0.0064329, -0.0043723, -0.0064477, -0.0046117, -0.0014315, 0.0014623
4: 0.0018457, 0.0027220, 0.0019476, 0.0027283, -0.0006218, 0.0006087
5: 0.0075233, 0.0132176, 0.0081850, 0.0132584, -0.0040407, 0.0039556
6: -0.0018139, -0.0003687, -0.0018243, -0.0005366, -0.0010040, 0.0010256
7: -0.0078308, -0.0040915, -0.0078576, -0.0045260, -0.0025976, 0.0026535
8: -0.0036823, -0.0017158, -0.0036964, -0.0019443, -0.0013661, 0.0013954
9: 0.0001257, 0.0024060, 0.0003907, 0.0024223, -0.0016181, 0.0015840

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014152, upper bound: 0.0014050
time: 1.40 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014300, upper bound: 0.0014055
time: 1.32 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9888738, 0.9923024, 0.9888805, 0.9921045, -0.0026112, 0.0025184
1: -0.0040363, -0.0031820, -0.0040346, -0.0032313, -0.0006506, 0.0006275
2: 0.0068090, 0.0113364, 0.0070703, 0.0113274, -0.0033255, 0.0034481
3: -0.0064329, -0.0043723, -0.0064289, -0.0044912, -0.0015694, 0.0015136
4: 0.0018457, 0.0027220, 0.0018963, 0.0027203, -0.0006436, 0.0006674
5: 0.0075233, 0.0132176, 0.0078520, 0.0132064, -0.0041826, 0.0043368
6: -0.0018139, -0.0003687, -0.0018111, -0.0004521, -0.0011007, 0.0010616
7: -0.0078308, -0.0040915, -0.0078234, -0.0043073, -0.0028479, 0.0027467
8: -0.0036823, -0.0017158, -0.0036784, -0.0018293, -0.0014977, 0.0014445
9: 0.0001257, 0.0024060, 0.0002574, 0.0024015, -0.0016749, 0.0017366

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014152, upper bound: 0.0014050
time: 1.41 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014300, upper bound: 0.0014055
time: 1.60 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9888023, 0.9919065, 0.9889178, 0.9922998, -0.0028416, 0.0018901
1: -0.0040541, -0.0032806, -0.0040254, -0.0031827, -0.0007081, 0.0004710
2: 0.0073317, 0.0114307, 0.0068124, 0.0112783, -0.0024958, 0.0037523
3: -0.0064759, -0.0046102, -0.0064065, -0.0043738, -0.0017079, 0.0011360
4: 0.0019469, 0.0027403, 0.0018464, 0.0027108, -0.0004831, 0.0007263
5: 0.0081808, 0.0133363, 0.0075277, 0.0131446, -0.0031391, 0.0047195
6: -0.0018441, -0.0005355, -0.0017954, -0.0003698, -0.0011978, 0.0007967
7: -0.0079088, -0.0045232, -0.0077829, -0.0040943, -0.0030992, 0.0020614
8: -0.0037233, -0.0019429, -0.0036571, -0.0017173, -0.0016298, 0.0010841
9: 0.0003890, 0.0024535, 0.0001275, 0.0023767, -0.0012570, 0.0018899

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B2_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014045, upper bound: 0.0014101
time: 1.36 seconds

## Relational analysis of NS_A1_B2_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014050, upper bound: 0.0014253
time: 1.27 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888738, 0.9923024, 0.9889178, 0.9922998, -0.0025215, 0.0020940
1: -0.0040363, -0.0031820, -0.0040254, -0.0031827, -0.0006283, 0.0005218
2: 0.0068090, 0.0113364, 0.0068124, 0.0112783, -0.0027651, 0.0033296
3: -0.0064329, -0.0043723, -0.0064065, -0.0043738, -0.0015155, 0.0012585
4: 0.0018457, 0.0027220, 0.0018464, 0.0027108, -0.0005352, 0.0006444
5: 0.0075233, 0.0132176, 0.0075277, 0.0131446, -0.0034777, 0.0041878
6: -0.0018139, -0.0003687, -0.0017954, -0.0003698, -0.0010629, 0.0008827
7: -0.0078308, -0.0040915, -0.0077829, -0.0040943, -0.0027501, 0.0022838
8: -0.0036823, -0.0017158, -0.0036571, -0.0017173, -0.0014462, 0.0012010
9: 0.0001257, 0.0024060, 0.0001275, 0.0023767, -0.0013926, 0.0016770

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B2_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013903, upper bound: 0.0014045
time: 1.54 seconds

## Relational analysis of NS_A1_B2_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014050, upper bound: 0.0014050
time: 1.30 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9888023, 0.9919065, 0.9889671, 0.9924710, -0.0030099, 0.0018968
1: -0.0040541, -0.0032806, -0.0040131, -0.0031400, -0.0007500, 0.0004726
2: 0.0073317, 0.0114307, 0.0065863, 0.0112132, -0.0025047, 0.0039746
3: -0.0064759, -0.0046102, -0.0063769, -0.0042709, -0.0018091, 0.0011400
4: 0.0019469, 0.0027403, 0.0018026, 0.0026982, -0.0004848, 0.0007693
5: 0.0081808, 0.0133363, 0.0072432, 0.0130627, -0.0031502, 0.0049990
6: -0.0018441, -0.0005355, -0.0017746, -0.0002976, -0.0012688, 0.0007996
7: -0.0079088, -0.0045232, -0.0077291, -0.0039075, -0.0032827, 0.0020687
8: -0.0037233, -0.0019429, -0.0036288, -0.0016191, -0.0017264, 0.0010879
9: 0.0003890, 0.0024535, 0.0000136, 0.0023439, -0.0012615, 0.0020018

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B2_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014513, upper bound: 0.0014250
time: 1.18 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014641, upper bound: 0.0014253
time: 1.32 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9888738, 0.9923024, 0.9889671, 0.9924710, -0.0027523, 0.0021787
1: -0.0040363, -0.0031820, -0.0040131, -0.0031400, -0.0006858, 0.0005429
2: 0.0068090, 0.0113364, 0.0065863, 0.0112132, -0.0028769, 0.0036343
3: -0.0064329, -0.0043723, -0.0063769, -0.0042709, -0.0016542, 0.0013095
4: 0.0018457, 0.0027220, 0.0018026, 0.0026982, -0.0005568, 0.0007034
5: 0.0075233, 0.0132176, 0.0072432, 0.0130627, -0.0036184, 0.0045710
6: -0.0018139, -0.0003687, -0.0017746, -0.0002976, -0.0011602, 0.0009184
7: -0.0078308, -0.0040915, -0.0077291, -0.0039075, -0.0030017, 0.0023762
8: -0.0036823, -0.0017158, -0.0036288, -0.0016191, -0.0015786, 0.0012496
9: 0.0001257, 0.0024060, 0.0000136, 0.0023439, -0.0014490, 0.0018304

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B2_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014513, upper bound: 0.0014045
time: 1.70 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014641, upper bound: 0.0014050
time: 1.68 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9892800, 0.9923536, 0.9889489, 0.9918969, -0.0019218, 0.0025085
1: -0.0039351, -0.0031692, -0.0040176, -0.0032830, -0.0004789, 0.0006251
2: 0.0067414, 0.0107999, 0.0073442, 0.0112372, -0.0033125, 0.0025377
3: -0.0061888, -0.0043415, -0.0063878, -0.0046159, -0.0011551, 0.0015077
4: 0.0018327, 0.0026182, 0.0019494, 0.0027028, -0.0006411, 0.0004912
5: 0.0074383, 0.0125429, 0.0081966, 0.0130929, -0.0041662, 0.0031918
6: -0.0016427, -0.0003471, -0.0017823, -0.0005395, -0.0008101, 0.0010574
7: -0.0073878, -0.0040357, -0.0077489, -0.0045336, -0.0020960, 0.0027359
8: -0.0034493, -0.0016865, -0.0036392, -0.0019483, -0.0011023, 0.0014388
9: 0.0000917, 0.0021358, 0.0003953, 0.0023560, -0.0016683, 0.0012781

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014738
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0015379
time: 1.26 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888805, 0.9921045, 0.9888023, 0.9919065, -0.0020217, 0.0026208
1: -0.0040346, -0.0032313, -0.0040541, -0.0032806, -0.0005038, 0.0006530
2: 0.0070703, 0.0113274, 0.0073317, 0.0114307, -0.0034607, 0.0026697
3: -0.0064289, -0.0044912, -0.0064759, -0.0046102, -0.0012151, 0.0015752
4: 0.0018963, 0.0027203, 0.0019469, 0.0027403, -0.0006698, 0.0005167
5: 0.0078520, 0.0132064, 0.0081808, 0.0133363, -0.0043526, 0.0033577
6: -0.0018111, -0.0004521, -0.0018441, -0.0005355, -0.0008522, 0.0011047
7: -0.0078234, -0.0043073, -0.0079088, -0.0045232, -0.0022050, 0.0028583
8: -0.0036784, -0.0018293, -0.0037233, -0.0019429, -0.0011596, 0.0015032
9: 0.0002574, 0.0024015, 0.0003890, 0.0024535, -0.0017430, 0.0013446

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014912, upper bound: 0.0014738
time: 1.42 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014912, upper bound: 0.0015464
time: 1.48 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9892800, 0.9923536, 0.9890224, 0.9922927, -0.0023681, 0.0025098
1: -0.0039351, -0.0031692, -0.0039993, -0.0031844, -0.0005901, 0.0006254
2: 0.0067414, 0.0107999, 0.0068217, 0.0111402, -0.0033142, 0.0031271
3: -0.0061888, -0.0043415, -0.0063436, -0.0043781, -0.0014233, 0.0015085
4: 0.0018327, 0.0026182, 0.0018482, 0.0026840, -0.0006415, 0.0006052
5: 0.0074383, 0.0125429, 0.0075394, 0.0129709, -0.0041684, 0.0039331
6: -0.0016427, -0.0003471, -0.0017513, -0.0003727, -0.0009983, 0.0010580
7: -0.0073878, -0.0040357, -0.0076688, -0.0041020, -0.0025828, 0.0027373
8: -0.0034493, -0.0016865, -0.0035971, -0.0017214, -0.0013583, 0.0014395
9: 0.0000917, 0.0021358, 0.0001321, 0.0023072, -0.0016692, 0.0015750

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010764, upper bound: 0.0013595
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010764, upper bound: 0.0013694
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9888805, 0.9921045, 0.9888738, 0.9923024, -0.0025184, 0.0026112
1: -0.0040346, -0.0032313, -0.0040363, -0.0031820, -0.0006275, 0.0006506
2: 0.0070703, 0.0113274, 0.0068090, 0.0113364, -0.0034481, 0.0033255
3: -0.0064289, -0.0044912, -0.0064329, -0.0043723, -0.0015136, 0.0015694
4: 0.0018963, 0.0027203, 0.0018457, 0.0027220, -0.0006674, 0.0006436
5: 0.0078520, 0.0132064, 0.0075233, 0.0132176, -0.0043368, 0.0041826
6: -0.0018111, -0.0004521, -0.0018139, -0.0003687, -0.0010616, 0.0011007
7: -0.0078234, -0.0043073, -0.0078308, -0.0040915, -0.0027467, 0.0028479
8: -0.0036784, -0.0018293, -0.0036823, -0.0017158, -0.0014445, 0.0014977
9: 0.0002574, 0.0024015, 0.0001257, 0.0024060, -0.0017366, 0.0016749

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B2_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014050, upper bound: 0.0014662
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014055, upper bound: 0.0014827
time: 1.29 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9894029, 0.9926836, 0.9889489, 0.9918969, -0.0018077, 0.0028774
1: -0.0039045, -0.0030870, -0.0040176, -0.0032830, -0.0004504, 0.0007170
2: 0.0063055, 0.0106377, 0.0073442, 0.0112372, -0.0037996, 0.0023870
3: -0.0061149, -0.0041431, -0.0063878, -0.0046159, -0.0010865, 0.0017294
4: 0.0017483, 0.0025868, 0.0019494, 0.0027028, -0.0007354, 0.0004620
5: 0.0068902, 0.0123389, 0.0081966, 0.0130929, -0.0047789, 0.0030022
6: -0.0015909, -0.0002080, -0.0017823, -0.0005395, -0.0007620, 0.0012129
7: -0.0072538, -0.0036757, -0.0077489, -0.0045336, -0.0019715, 0.0031382
8: -0.0033788, -0.0014972, -0.0036392, -0.0019483, -0.0010368, 0.0016504
9: -0.0001278, 0.0020541, 0.0003953, 0.0023560, -0.0019137, 0.0012022

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009096, upper bound: 0.0012375
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009096, upper bound: 0.0012386
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9894029, 0.9926836, 0.9890224, 0.9922927, -0.0020696, 0.0026441
1: -0.0039045, -0.0030870, -0.0039993, -0.0031844, -0.0005157, 0.0006588
2: 0.0063055, 0.0106377, 0.0068217, 0.0111402, -0.0034915, 0.0027328
3: -0.0061149, -0.0041431, -0.0063436, -0.0043781, -0.0012439, 0.0015892
4: 0.0017483, 0.0025868, 0.0018482, 0.0026840, -0.0006758, 0.0005289
5: 0.0068902, 0.0123389, 0.0075394, 0.0129709, -0.0043913, 0.0034372
6: -0.0015909, -0.0002080, -0.0017513, -0.0003727, -0.0008724, 0.0011146
7: -0.0072538, -0.0036757, -0.0076688, -0.0041020, -0.0022571, 0.0028837
8: -0.0033788, -0.0014972, -0.0035971, -0.0017214, -0.0011870, 0.0015165
9: -0.0001278, 0.0020541, 0.0001321, 0.0023072, -0.0017585, 0.0013764

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009096, upper bound: 0.0012375
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009096, upper bound: 0.0012386
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9889671, 0.9924710, 0.9888023, 0.9919065, -0.0018968, 0.0030099
1: -0.0040131, -0.0031400, -0.0040541, -0.0032806, -0.0004726, 0.0007500
2: 0.0065863, 0.0112132, 0.0073317, 0.0114307, -0.0039746, 0.0025047
3: -0.0063769, -0.0042709, -0.0064759, -0.0046102, -0.0011400, 0.0018091
4: 0.0018026, 0.0026982, 0.0019469, 0.0027403, -0.0007693, 0.0004848
5: 0.0072432, 0.0130627, 0.0081808, 0.0133363, -0.0049990, 0.0031502
6: -0.0017746, -0.0002976, -0.0018441, -0.0005355, -0.0007996, 0.0012688
7: -0.0077291, -0.0039075, -0.0079088, -0.0045232, -0.0020687, 0.0032827
8: -0.0036288, -0.0016191, -0.0037233, -0.0019429, -0.0010879, 0.0017264
9: 0.0000136, 0.0023439, 0.0003890, 0.0024535, -0.0020018, 0.0012615

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014249, upper bound: 0.0014513
time: 1.16 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014252, upper bound: 0.0014641
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9889671, 0.9924710, 0.9888738, 0.9923024, -0.0021787, 0.0027523
1: -0.0040131, -0.0031400, -0.0040363, -0.0031820, -0.0005429, 0.0006858
2: 0.0065863, 0.0112132, 0.0068090, 0.0113364, -0.0036343, 0.0028769
3: -0.0063769, -0.0042709, -0.0064329, -0.0043723, -0.0013095, 0.0016542
4: 0.0018026, 0.0026982, 0.0018457, 0.0027220, -0.0007034, 0.0005568
5: 0.0072432, 0.0130627, 0.0075233, 0.0132176, -0.0045710, 0.0036184
6: -0.0017746, -0.0002976, -0.0018139, -0.0003687, -0.0009184, 0.0011602
7: -0.0077291, -0.0039075, -0.0078308, -0.0040915, -0.0023762, 0.0030017
8: -0.0036288, -0.0016191, -0.0036823, -0.0017158, -0.0012496, 0.0015786
9: 0.0000136, 0.0023439, 0.0001257, 0.0024060, -0.0018304, 0.0014490

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014249, upper bound: 0.0014513
time: 1.16 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014252, upper bound: 0.0014641
time: 1.27 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9892800, 0.9923536, 0.9889867, 0.9920977, -0.0018718, 0.0022052
1: -0.0039351, -0.0031692, -0.0040082, -0.0032330, -0.0004664, 0.0005495
2: 0.0067414, 0.0107999, 0.0070791, 0.0111873, -0.0029119, 0.0024718
3: -0.0061888, -0.0043415, -0.0063651, -0.0044952, -0.0011250, 0.0013254
4: 0.0018327, 0.0026182, 0.0018980, 0.0026932, -0.0005636, 0.0004784
5: 0.0074383, 0.0125429, 0.0078632, 0.0130301, -0.0036624, 0.0031088
6: -0.0016427, -0.0003471, -0.0017663, -0.0004549, -0.0007891, 0.0009296
7: -0.0073878, -0.0040357, -0.0077077, -0.0043146, -0.0020415, 0.0024051
8: -0.0034493, -0.0016865, -0.0036175, -0.0018332, -0.0010736, 0.0012648
9: 0.0000917, 0.0021358, 0.0002618, 0.0023309, -0.0014666, 0.0012449

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0014738
time: 1.22 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0015380
time: 1.21 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888805, 0.9921045, 0.9888360, 0.9921069, -0.0019853, 0.0024457
1: -0.0040346, -0.0032313, -0.0040457, -0.0032307, -0.0004947, 0.0006094
2: 0.0070703, 0.0113274, 0.0070670, 0.0113862, -0.0032295, 0.0026215
3: -0.0064289, -0.0044912, -0.0064556, -0.0044897, -0.0011932, 0.0014699
4: 0.0018963, 0.0027203, 0.0018957, 0.0027317, -0.0006251, 0.0005074
5: 0.0078520, 0.0132064, 0.0078479, 0.0132802, -0.0040619, 0.0032972
6: -0.0018111, -0.0004521, -0.0018298, -0.0004511, -0.0008369, 0.0010309
7: -0.0078234, -0.0043073, -0.0078720, -0.0043047, -0.0021652, 0.0026674
8: -0.0036784, -0.0018293, -0.0037039, -0.0018279, -0.0011387, 0.0014027
9: 0.0002574, 0.0024015, 0.0002557, 0.0024310, -0.0016265, 0.0013203

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014928, upper bound: 0.0014738
time: 1.26 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014928, upper bound: 0.0015464
time: 1.19 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9892800, 0.9923536, 0.9890811, 0.9924639, -0.0023188, 0.0022020
1: -0.0039351, -0.0031692, -0.0039846, -0.0031417, -0.0005778, 0.0005487
2: 0.0067414, 0.0107999, 0.0065956, 0.0110625, -0.0029077, 0.0030620
3: -0.0061888, -0.0043415, -0.0063083, -0.0042751, -0.0013937, 0.0013235
4: 0.0018327, 0.0026182, 0.0018045, 0.0026690, -0.0005628, 0.0005926
5: 0.0074383, 0.0125429, 0.0072550, 0.0128732, -0.0036572, 0.0038512
6: -0.0016427, -0.0003471, -0.0017265, -0.0003006, -0.0009775, 0.0009282
7: -0.0073878, -0.0040357, -0.0076046, -0.0039153, -0.0025290, 0.0024016
8: -0.0034493, -0.0016865, -0.0035634, -0.0016231, -0.0013300, 0.0012630
9: 0.0000917, 0.0021358, 0.0000183, 0.0022680, -0.0014645, 0.0015422

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011831, upper bound: 0.0012334
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011831, upper bound: 0.0014415
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9888805, 0.9921045, 0.9889241, 0.9924735, -0.0024825, 0.0024321
1: -0.0040346, -0.0032313, -0.0040238, -0.0031393, -0.0006186, 0.0006060
2: 0.0070703, 0.0113274, 0.0065828, 0.0112698, -0.0032116, 0.0032781
3: -0.0064289, -0.0044912, -0.0064027, -0.0042693, -0.0014920, 0.0014618
4: 0.0018963, 0.0027203, 0.0018020, 0.0027091, -0.0006216, 0.0006345
5: 0.0078520, 0.0132064, 0.0072389, 0.0131340, -0.0040394, 0.0041229
6: -0.0018111, -0.0004521, -0.0017927, -0.0002965, -0.0010464, 0.0010252
7: -0.0078234, -0.0043073, -0.0077759, -0.0039047, -0.0027075, 0.0026526
8: -0.0036784, -0.0018293, -0.0036534, -0.0016176, -0.0014238, 0.0013950
9: 0.0002574, 0.0024015, 0.0000118, 0.0023725, -0.0016175, 0.0016510

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0012350
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0012350
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.9890811, 0.9924639, 0.9892800, 0.9923536, -0.0022020, 0.0023188
1: -0.0039846, -0.0031417, -0.0039351, -0.0031692, -0.0005487, 0.0005778
2: 0.0065956, 0.0110625, 0.0067414, 0.0107999, -0.0030620, 0.0029077
3: -0.0063083, -0.0042751, -0.0061888, -0.0043415, -0.0013235, 0.0013937
4: 0.0018045, 0.0026690, 0.0018327, 0.0026182, -0.0005926, 0.0005628
5: 0.0072550, 0.0128732, 0.0074383, 0.0125429, -0.0038512, 0.0036572
6: -0.0017265, -0.0003006, -0.0016427, -0.0003471, -0.0009282, 0.0009775
7: -0.0076046, -0.0039153, -0.0073878, -0.0040357, -0.0024016, 0.0025290
8: -0.0035634, -0.0016231, -0.0034493, -0.0016865, -0.0012630, 0.0013300
9: 0.0000183, 0.0022680, 0.0000917, 0.0021358, -0.0015422, 0.0014645

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0011494
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0011768
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9890811, 0.9924639, 0.9894029, 0.9926836, -0.0023194, 0.0020057
1: -0.0039846, -0.0031417, -0.0039045, -0.0030870, -0.0005779, 0.0004998
2: 0.0065956, 0.0110625, 0.0063055, 0.0106377, -0.0026485, 0.0030627
3: -0.0063083, -0.0042751, -0.0061149, -0.0041431, -0.0013940, 0.0012055
4: 0.0018045, 0.0026690, 0.0017483, 0.0025868, -0.0005126, 0.0005928
5: 0.0072550, 0.0128732, 0.0068902, 0.0123389, -0.0033311, 0.0038521
6: -0.0017265, -0.0003006, -0.0015909, -0.0002080, -0.0009777, 0.0008455
7: -0.0076046, -0.0039153, -0.0072538, -0.0036757, -0.0025296, 0.0021875
8: -0.0035634, -0.0016231, -0.0033788, -0.0014972, -0.0013303, 0.0011504
9: 0.0000183, 0.0022680, -0.0001278, 0.0020541, -0.0013339, 0.0015425

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0011494
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0011768
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9894029, 0.9926836, 0.9888560, 0.9921041, -0.0020039, 0.0029995
1: -0.0039045, -0.0030870, -0.0040408, -0.0032314, -0.0004993, 0.0007474
2: 0.0063055, 0.0106377, 0.0070708, 0.0113599, -0.0039608, 0.0026461
3: -0.0061149, -0.0041431, -0.0064436, -0.0044915, -0.0012044, 0.0018028
4: 0.0017483, 0.0025868, 0.0018964, 0.0027266, -0.0007666, 0.0005122
5: 0.0068902, 0.0123389, 0.0078527, 0.0132472, -0.0049816, 0.0033281
6: -0.0015909, -0.0002080, -0.0018214, -0.0004523, -0.0008447, 0.0012644
7: -0.0072538, -0.0036757, -0.0078502, -0.0043078, -0.0021855, 0.0032713
8: -0.0033788, -0.0014972, -0.0036925, -0.0018296, -0.0011494, 0.0017204
9: -0.0001278, 0.0020541, 0.0002576, 0.0024178, -0.0019949, 0.0013327

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0013728
time: 1.27 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0011494
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9889671, 0.9924710, 0.9888560, 0.9921041, -0.0021256, 0.0024923
1: -0.0040131, -0.0031400, -0.0040408, -0.0032314, -0.0005296, 0.0006210
2: 0.0065863, 0.0112132, 0.0070708, 0.0113599, -0.0032910, 0.0028068
3: -0.0063769, -0.0042709, -0.0064436, -0.0044915, -0.0012776, 0.0014979
4: 0.0018026, 0.0026982, 0.0018964, 0.0027266, -0.0006370, 0.0005433
5: 0.0072432, 0.0130627, 0.0078527, 0.0132472, -0.0041393, 0.0035303
6: -0.0017746, -0.0002976, -0.0018214, -0.0004523, -0.0008960, 0.0010506
7: -0.0077291, -0.0039075, -0.0078502, -0.0043078, -0.0023183, 0.0027182
8: -0.0036288, -0.0016191, -0.0036925, -0.0018296, -0.0012192, 0.0014295
9: 0.0000136, 0.0023439, 0.0002576, 0.0024178, -0.0016575, 0.0014137

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0014523
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0014523
time: 0.94 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.58 seconds
NS_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014275
NS_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014275
NS_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014275
NS_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014275
NS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014912
NS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014555
NS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014912
NS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014554
NS_A1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0013063, upper bound: 0.0010764
NS_A1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0013169, upper bound: 0.0010764
NS_A1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0013063, upper bound: 0.0010764
NS_A1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0013169, upper bound: 0.0010764
NS_A1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014152, upper bound: 0.0014050
NS_A1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014300, upper bound: 0.0014055
NS_A1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014152, upper bound: 0.0014050
NS_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014300, upper bound: 0.0014055
NS_A1_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014045, upper bound: 0.0014101
NS_A1_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014050, upper bound: 0.0014253
NS_A1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0013903, upper bound: 0.0014045
NS_A1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014050, upper bound: 0.0014050
NS_A1_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014513, upper bound: 0.0014250
NS_A1_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014641, upper bound: 0.0014253
NS_A1_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014513, upper bound: 0.0014045
NS_A1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014641, upper bound: 0.0014050
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014738
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0015379
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014912, upper bound: 0.0014738
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014912, upper bound: 0.0015464
NS_A2_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0010764, upper bound: 0.0013595
NS_A2_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0010764, upper bound: 0.0013694
NS_A2_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014050, upper bound: 0.0014662
NS_A2_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014055, upper bound: 0.0014827
NS_A2_B1_A2_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0009096, upper bound: 0.0012375
NS_A2_B1_A2_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0009096, upper bound: 0.0012386
NS_A2_B1_A2_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0009096, upper bound: 0.0012375
NS_A2_B1_A2_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0009096, upper bound: 0.0012386
NS_A2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014249, upper bound: 0.0014513
NS_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014252, upper bound: 0.0014641
NS_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014249, upper bound: 0.0014513
NS_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014252, upper bound: 0.0014641
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0014738
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0015380
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014928, upper bound: 0.0014738
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0014928, upper bound: 0.0015464
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0011831, upper bound: 0.0012334
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0011831, upper bound: 0.0014415
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0012350
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0012350
NS_A2_B2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0011494
NS_A2_B2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0011768
NS_A2_B2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0011494
NS_A2_B2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0011768
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0013728
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0011494
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0014523
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0014523

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9892468, 0.9921501, 0.9892468, 0.9921501, -0.0018250, 0.0018250
1: -0.0039434, -0.0032200, -0.0039434, -0.0032200, -0.0004547, 0.0004547
2: 0.0070101, 0.0108438, 0.0070101, 0.0108438, -0.0024098, 0.0024098
3: -0.0062088, -0.0044638, -0.0062088, -0.0044638, -0.0010968, 0.0010968
4: 0.0018847, 0.0026267, 0.0018847, 0.0026267, -0.0004664, 0.0004664
5: 0.0077763, 0.0125981, 0.0077763, 0.0125981, -0.0030309, 0.0030309
6: -0.0016567, -0.0004329, -0.0016567, -0.0004329, -0.0007693, 0.0007693
7: -0.0074240, -0.0042576, -0.0074240, -0.0042576, -0.0019904, 0.0019904
8: -0.0034684, -0.0018032, -0.0034684, -0.0018032, -0.0010467, 0.0010467
9: 0.0002270, 0.0021579, 0.0002270, 0.0021579, -0.0012137, 0.0012137

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0013840
time: 1.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014031
time: 1.48 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888493, 0.9919039, 0.9892468, 0.9921501, -0.0023762, 0.0018180
1: -0.0040424, -0.0032813, -0.0039434, -0.0032200, -0.0005921, 0.0004530
2: 0.0073351, 0.0113688, 0.0070101, 0.0108438, -0.0024006, 0.0031377
3: -0.0064477, -0.0046117, -0.0062088, -0.0044638, -0.0014282, 0.0010927
4: 0.0019476, 0.0027283, 0.0018847, 0.0026267, -0.0004646, 0.0006073
5: 0.0081850, 0.0132584, 0.0077763, 0.0125981, -0.0030193, 0.0039465
6: -0.0018243, -0.0005366, -0.0016567, -0.0004329, -0.0010017, 0.0007663
7: -0.0078576, -0.0045260, -0.0074240, -0.0042576, -0.0025916, 0.0019828
8: -0.0036964, -0.0019443, -0.0034684, -0.0018032, -0.0013629, 0.0010427
9: 0.0003907, 0.0024223, 0.0002270, 0.0021579, -0.0012091, 0.0015803

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013840, upper bound: 0.0014031
time: 1.50 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014031
time: 1.65 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9892468, 0.9921501, 0.9892800, 0.9923536, -0.0021835, 0.0019232
1: -0.0039434, -0.0032200, -0.0039351, -0.0031692, -0.0005441, 0.0004792
2: 0.0070101, 0.0108438, 0.0067414, 0.0107999, -0.0025396, 0.0028833
3: -0.0062088, -0.0044638, -0.0061888, -0.0043415, -0.0013124, 0.0011559
4: 0.0018847, 0.0026267, 0.0018327, 0.0026182, -0.0004915, 0.0005581
5: 0.0077763, 0.0125981, 0.0074383, 0.0125429, -0.0031941, 0.0036265
6: -0.0016567, -0.0004329, -0.0016427, -0.0003471, -0.0009204, 0.0008107
7: -0.0074240, -0.0042576, -0.0073878, -0.0040357, -0.0023815, 0.0020975
8: -0.0034684, -0.0018032, -0.0034493, -0.0016865, -0.0012524, 0.0011031
9: 0.0002270, 0.0021579, 0.0000917, 0.0021358, -0.0012791, 0.0014522

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014351, upper bound: 0.0014031
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014505, upper bound: 0.0014031
time: 1.34 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9888493, 0.9919039, 0.9892800, 0.9923536, -0.0027348, 0.0019162
1: -0.0040424, -0.0032813, -0.0039351, -0.0031692, -0.0006814, 0.0004775
2: 0.0073351, 0.0113688, 0.0067414, 0.0107999, -0.0025304, 0.0036113
3: -0.0064477, -0.0046117, -0.0061888, -0.0043415, -0.0016437, 0.0011517
4: 0.0019476, 0.0027283, 0.0018327, 0.0026182, -0.0004897, 0.0006990
5: 0.0081850, 0.0132584, 0.0074383, 0.0125429, -0.0031825, 0.0045420
6: -0.0018243, -0.0005366, -0.0016427, -0.0003471, -0.0011528, 0.0008078
7: -0.0078576, -0.0045260, -0.0073878, -0.0040357, -0.0029827, 0.0020899
8: -0.0036964, -0.0019443, -0.0034493, -0.0016865, -0.0015686, 0.0010991
9: 0.0003907, 0.0024223, 0.0000917, 0.0021358, -0.0012744, 0.0018188

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014351, upper bound: 0.0014031
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014505, upper bound: 0.0014031
time: 1.28 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9892468, 0.9921501, 0.9888493, 0.9919039, -0.0018180, 0.0023762
1: -0.0039434, -0.0032200, -0.0040424, -0.0032813, -0.0004530, 0.0005921
2: 0.0070101, 0.0108438, 0.0073351, 0.0113688, -0.0031377, 0.0024006
3: -0.0062088, -0.0044638, -0.0064477, -0.0046117, -0.0010927, 0.0014282
4: 0.0018847, 0.0026267, 0.0019476, 0.0027283, -0.0006073, 0.0004646
5: 0.0077763, 0.0125981, 0.0081850, 0.0132584, -0.0039465, 0.0030193
6: -0.0016567, -0.0004329, -0.0018243, -0.0005366, -0.0007663, 0.0010017
7: -0.0074240, -0.0042576, -0.0078576, -0.0045260, -0.0019828, 0.0025916
8: -0.0034684, -0.0018032, -0.0036964, -0.0019443, -0.0010427, 0.0013629
9: 0.0002270, 0.0021579, 0.0003907, 0.0024223, -0.0015803, 0.0012091

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014510
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014668
time: 1.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888493, 0.9919039, 0.9888493, 0.9919039, -0.0019294, 0.0019294
1: -0.0040424, -0.0032813, -0.0040424, -0.0032813, -0.0004808, 0.0004808
2: 0.0073351, 0.0113688, 0.0073351, 0.0113688, -0.0025478, 0.0025478
3: -0.0064477, -0.0046117, -0.0064477, -0.0046117, -0.0011597, 0.0011597
4: 0.0019476, 0.0027283, 0.0019476, 0.0027283, -0.0004931, 0.0004931
5: 0.0081850, 0.0132584, 0.0081850, 0.0132584, -0.0032045, 0.0032045
6: -0.0018243, -0.0005366, -0.0018243, -0.0005366, -0.0008133, 0.0008133
7: -0.0078576, -0.0045260, -0.0078576, -0.0045260, -0.0021043, 0.0021043
8: -0.0036964, -0.0019443, -0.0036964, -0.0019443, -0.0011066, 0.0011066
9: 0.0003907, 0.0024223, 0.0003907, 0.0024223, -0.0012832, 0.0012832

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014100
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014311
time: 1.29 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9892468, 0.9921501, 0.9888805, 0.9921045, -0.0020475, 0.0024046
1: -0.0039434, -0.0032200, -0.0040346, -0.0032313, -0.0005102, 0.0005992
2: 0.0070101, 0.0108438, 0.0070703, 0.0113274, -0.0031753, 0.0027037
3: -0.0062088, -0.0044638, -0.0064289, -0.0044912, -0.0012306, 0.0014453
4: 0.0018847, 0.0026267, 0.0018963, 0.0027203, -0.0006146, 0.0005233
5: 0.0077763, 0.0125981, 0.0078520, 0.0132064, -0.0039937, 0.0034005
6: -0.0016567, -0.0004329, -0.0018111, -0.0004521, -0.0008631, 0.0010136
7: -0.0074240, -0.0042576, -0.0078234, -0.0043073, -0.0022331, 0.0026226
8: -0.0034684, -0.0018032, -0.0036784, -0.0018293, -0.0011743, 0.0013792
9: 0.0002270, 0.0021579, 0.0002574, 0.0024015, -0.0015992, 0.0013617

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014502, upper bound: 0.0014503
time: 1.23 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014505, upper bound: 0.0014664
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9888493, 0.9919039, 0.9888805, 0.9921045, -0.0022717, 0.0020149
1: -0.0040424, -0.0032813, -0.0040346, -0.0032313, -0.0005661, 0.0005021
2: 0.0073351, 0.0113688, 0.0070703, 0.0113274, -0.0026607, 0.0029998
3: -0.0064477, -0.0046117, -0.0064289, -0.0044912, -0.0013654, 0.0012110
4: 0.0019476, 0.0027283, 0.0018963, 0.0027203, -0.0005150, 0.0005806
5: 0.0081850, 0.0132584, 0.0078520, 0.0132064, -0.0033464, 0.0037729
6: -0.0018243, -0.0005366, -0.0018111, -0.0004521, -0.0009576, 0.0008494
7: -0.0078576, -0.0045260, -0.0078234, -0.0043073, -0.0024776, 0.0021976
8: -0.0036964, -0.0019443, -0.0036784, -0.0018293, -0.0013030, 0.0011557
9: 0.0003907, 0.0024223, 0.0002574, 0.0024015, -0.0013401, 0.0015108

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014351, upper bound: 0.0014299
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014505, upper bound: 0.0014311
time: 1.27 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: 0.9890239, 0.9922627, 0.9892615, 0.9920586, -0.0020421, 0.0021961
1: -0.0039989, -0.0031919, -0.0039397, -0.0032427, -0.0005088, 0.0005472
2: 0.0068614, 0.0111382, 0.0071307, 0.0108243, -0.0029000, 0.0026966
3: -0.0063427, -0.0043961, -0.0061999, -0.0045187, -0.0012274, 0.0013199
4: 0.0018559, 0.0026837, 0.0019080, 0.0026229, -0.0005613, 0.0005219
5: 0.0075893, 0.0129684, 0.0079280, 0.0125736, -0.0036474, 0.0033916
6: -0.0017507, -0.0003854, -0.0016505, -0.0004714, -0.0008608, 0.0009257
7: -0.0076672, -0.0041348, -0.0074079, -0.0043572, -0.0022272, 0.0023952
8: -0.0035962, -0.0017386, -0.0034599, -0.0018556, -0.0011713, 0.0012596
9: 0.0001521, 0.0023062, 0.0002878, 0.0021481, -0.0014606, 0.0013582

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A2_B1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013063, upper bound: 0.0010629
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013063, upper bound: 0.0010764
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: 0.9890233, 0.9922774, 0.9892507, 0.9921016, -0.0020558, 0.0022185
1: -0.0039991, -0.0031882, -0.0039424, -0.0032320, -0.0005123, 0.0005528
2: 0.0068420, 0.0111389, 0.0070741, 0.0108385, -0.0029296, 0.0027147
3: -0.0063431, -0.0043873, -0.0062063, -0.0044929, -0.0012356, 0.0013334
4: 0.0018521, 0.0026838, 0.0018971, 0.0026257, -0.0005670, 0.0005254
5: 0.0075649, 0.0129693, 0.0078568, 0.0125914, -0.0036846, 0.0034144
6: -0.0017509, -0.0003792, -0.0016550, -0.0004533, -0.0008666, 0.0009352
7: -0.0076678, -0.0041188, -0.0074196, -0.0043105, -0.0022422, 0.0024196
8: -0.0035966, -0.0017302, -0.0034661, -0.0018310, -0.0011791, 0.0012725
9: 0.0001424, 0.0023065, 0.0002593, 0.0021552, -0.0014755, 0.0013673

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013159, upper bound: 0.0010629
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013159, upper bound: 0.0010764
time: 1.46 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: 0.9890239, 0.9922627, 0.9892911, 0.9922644, -0.0024160, 0.0023126
1: -0.0039989, -0.0031919, -0.0039323, -0.0031914, -0.0006020, 0.0005762
2: 0.0068614, 0.0111382, 0.0068589, 0.0107853, -0.0030537, 0.0031903
3: -0.0063427, -0.0043961, -0.0061821, -0.0043950, -0.0014521, 0.0013899
4: 0.0018559, 0.0026837, 0.0018554, 0.0026154, -0.0005910, 0.0006175
5: 0.0075893, 0.0129684, 0.0075861, 0.0125245, -0.0038408, 0.0040126
6: -0.0017507, -0.0003854, -0.0016380, -0.0003846, -0.0010184, 0.0009748
7: -0.0076672, -0.0041348, -0.0073757, -0.0041327, -0.0026350, 0.0025222
8: -0.0035962, -0.0017386, -0.0034430, -0.0017375, -0.0013857, 0.0013264
9: 0.0001521, 0.0023062, 0.0001509, 0.0021284, -0.0015380, 0.0016068

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A2_B1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013595, upper bound: 0.0010629
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013595, upper bound: 0.0010764
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: 0.9890233, 0.9922774, 0.9892840, 0.9923024, -0.0024323, 0.0023218
1: -0.0039991, -0.0031882, -0.0039341, -0.0031820, -0.0006061, 0.0005785
2: 0.0068420, 0.0111389, 0.0068089, 0.0107946, -0.0030659, 0.0032119
3: -0.0063431, -0.0043873, -0.0061864, -0.0043722, -0.0014619, 0.0013955
4: 0.0018521, 0.0026838, 0.0018457, 0.0026172, -0.0005934, 0.0006217
5: 0.0075649, 0.0129693, 0.0075232, 0.0125362, -0.0038561, 0.0040397
6: -0.0017509, -0.0003792, -0.0016410, -0.0003686, -0.0010253, 0.0009787
7: -0.0076678, -0.0041188, -0.0073834, -0.0040914, -0.0026528, 0.0025322
8: -0.0035966, -0.0017302, -0.0034470, -0.0017158, -0.0013951, 0.0013317
9: 0.0001424, 0.0023065, 0.0001257, 0.0021331, -0.0015441, 0.0016177

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A2_B1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013692, upper bound: 0.0010629
time: 1.24 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013692, upper bound: 0.0010764
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: 0.9888752, 0.9922723, 0.9888603, 0.9918175, -0.0022852, 0.0023532
1: -0.0040359, -0.0031895, -0.0040397, -0.0033028, -0.0005694, 0.0005864
2: 0.0068487, 0.0113344, 0.0074493, 0.0113541, -0.0031074, 0.0030176
3: -0.0064320, -0.0043903, -0.0064410, -0.0046637, -0.0013735, 0.0014144
4: 0.0018534, 0.0027216, 0.0019697, 0.0027255, -0.0006014, 0.0005840
5: 0.0075733, 0.0132151, 0.0083287, 0.0132399, -0.0039083, 0.0037953
6: -0.0018133, -0.0003813, -0.0018196, -0.0005731, -0.0009633, 0.0009920
7: -0.0078292, -0.0041243, -0.0078455, -0.0046204, -0.0024923, 0.0025665
8: -0.0036814, -0.0017331, -0.0036900, -0.0019940, -0.0013107, 0.0013497
9: 0.0001457, 0.0024050, 0.0004482, 0.0024149, -0.0015651, 0.0015198

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A2_B2_B1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009394, upper bound: 0.0009965
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008584, upper bound: 0.0008620
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9888747, 0.9922870, 0.9888533, 0.9918526, -0.0023078, 0.0023858
1: -0.0040361, -0.0031858, -0.0040414, -0.0032941, -0.0005750, 0.0005945
2: 0.0068292, 0.0113351, 0.0074029, 0.0113634, -0.0031504, 0.0030474
3: -0.0064324, -0.0043815, -0.0064452, -0.0046426, -0.0013870, 0.0014339
4: 0.0018497, 0.0027218, 0.0019607, 0.0027272, -0.0006098, 0.0005898
5: 0.0075488, 0.0132161, 0.0082704, 0.0132516, -0.0039624, 0.0038328
6: -0.0018135, -0.0003751, -0.0018226, -0.0005583, -0.0009728, 0.0010057
7: -0.0078298, -0.0041082, -0.0078531, -0.0045821, -0.0025170, 0.0026020
8: -0.0036818, -0.0017246, -0.0036940, -0.0019738, -0.0013236, 0.0013684
9: 0.0001359, 0.0024053, 0.0004249, 0.0024196, -0.0015867, 0.0015348

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009970, upper bound: 0.0010066
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009256, upper bound: 0.0008846
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9888752, 0.9922723, 0.9888886, 0.9920204, -0.0025182, 0.0024558
1: -0.0040359, -0.0031895, -0.0040326, -0.0032523, -0.0006275, 0.0006119
2: 0.0068487, 0.0113344, 0.0071812, 0.0113169, -0.0032429, 0.0033253
3: -0.0064320, -0.0043903, -0.0064241, -0.0045417, -0.0015135, 0.0014760
4: 0.0018534, 0.0027216, 0.0019178, 0.0027182, -0.0006277, 0.0006436
5: 0.0075733, 0.0132151, 0.0079916, 0.0131931, -0.0040787, 0.0041824
6: -0.0018133, -0.0003813, -0.0018077, -0.0004875, -0.0010615, 0.0010352
7: -0.0078292, -0.0041243, -0.0078147, -0.0043990, -0.0027465, 0.0026784
8: -0.0036814, -0.0017331, -0.0036738, -0.0018775, -0.0014444, 0.0014086
9: 0.0001457, 0.0024050, 0.0003132, 0.0023962, -0.0016333, 0.0016748

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A2_B2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010031, upper bound: 0.0009965
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009246, upper bound: 0.0008650
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9888747, 0.9922870, 0.9888846, 0.9920495, -0.0025419, 0.0024760
1: -0.0040361, -0.0031858, -0.0040336, -0.0032450, -0.0006334, 0.0006170
2: 0.0068292, 0.0113351, 0.0071429, 0.0113221, -0.0032696, 0.0033565
3: -0.0064324, -0.0043815, -0.0064264, -0.0045243, -0.0015277, 0.0014882
4: 0.0018497, 0.0027218, 0.0019104, 0.0027193, -0.0006328, 0.0006496
5: 0.0075488, 0.0132161, 0.0079434, 0.0131997, -0.0041123, 0.0042216
6: -0.0018135, -0.0003751, -0.0018094, -0.0004753, -0.0010715, 0.0010437
7: -0.0078298, -0.0041082, -0.0078190, -0.0043673, -0.0027723, 0.0027005
8: -0.0036818, -0.0017246, -0.0036761, -0.0018609, -0.0014579, 0.0014201
9: 0.0001359, 0.0024053, 0.0002939, 0.0023988, -0.0016467, 0.0016905

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A2_B2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010583, upper bound: 0.0010066
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009879, upper bound: 0.0008854
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.9888144, 0.9918199, 0.9889191, 0.9922696, -0.0027853, 0.0017848
1: -0.0040511, -0.0033022, -0.0040250, -0.0031901, -0.0006940, 0.0004447
2: 0.0074460, 0.0114149, 0.0068521, 0.0112764, -0.0023568, 0.0036779
3: -0.0064687, -0.0046622, -0.0064056, -0.0043919, -0.0016740, 0.0010727
4: 0.0019690, 0.0027372, 0.0018541, 0.0027104, -0.0004562, 0.0007118
5: 0.0083245, 0.0133164, 0.0075776, 0.0131422, -0.0029643, 0.0046258
6: -0.0018390, -0.0005720, -0.0017948, -0.0003824, -0.0011741, 0.0007524
7: -0.0078957, -0.0046176, -0.0077813, -0.0041271, -0.0030377, 0.0019466
8: -0.0037164, -0.0019925, -0.0036563, -0.0017345, -0.0015975, 0.0010237
9: 0.0004466, 0.0024455, 0.0001474, 0.0023758, -0.0011870, 0.0018524

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_B2_B1_A1_A1_A1

### Relational analysis result of NS_A1_B2_B2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009965, upper bound: 0.0009394
time: 0.94 seconds

## Relational analysis of NS_A1_B2_B2_B1_A1_A1_A2

### Relational analysis result of NS_A1_B2_B2_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008620, upper bound: 0.0008584
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.9888064, 0.9918550, 0.9889186, 0.9922845, -0.0027946, 0.0017783
1: -0.0040531, -0.0032935, -0.0040251, -0.0031865, -0.0006964, 0.0004431
2: 0.0073997, 0.0114253, 0.0068327, 0.0112771, -0.0023483, 0.0036903
3: -0.0064734, -0.0046411, -0.0064060, -0.0043831, -0.0016797, 0.0010688
4: 0.0019601, 0.0027392, 0.0018503, 0.0027106, -0.0004545, 0.0007142
5: 0.0082663, 0.0133295, 0.0075531, 0.0131431, -0.0029535, 0.0046414
6: -0.0018423, -0.0005572, -0.0017950, -0.0003762, -0.0011780, 0.0007496
7: -0.0079043, -0.0045794, -0.0077819, -0.0041111, -0.0030480, 0.0019395
8: -0.0037209, -0.0019724, -0.0036566, -0.0017261, -0.0016029, 0.0010200
9: 0.0004232, 0.0024508, 0.0001377, 0.0023761, -0.0011827, 0.0018586

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_B2_B1_A1_A2_A1

### Relational analysis result of NS_A1_B2_B2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010066, upper bound: 0.0009970
time: 0.90 seconds

## Relational analysis of NS_A1_B2_B2_B1_A1_A2_A2

### Relational analysis result of NS_A1_B2_B2_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008846, upper bound: 0.0009256
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9888752, 0.9922723, 0.9889259, 0.9922126, -0.0024252, 0.0020142
1: -0.0040359, -0.0031895, -0.0040233, -0.0032044, -0.0006043, 0.0005019
2: 0.0068487, 0.0113344, 0.0069275, 0.0112675, -0.0026598, 0.0032024
3: -0.0064320, -0.0043903, -0.0064016, -0.0044262, -0.0014576, 0.0012106
4: 0.0018534, 0.0027216, 0.0018687, 0.0027087, -0.0005148, 0.0006198
5: 0.0075733, 0.0132151, 0.0076724, 0.0131311, -0.0033453, 0.0040278
6: -0.0018133, -0.0003813, -0.0017920, -0.0004065, -0.0010223, 0.0008491
7: -0.0078292, -0.0041243, -0.0077740, -0.0041894, -0.0026450, 0.0021968
8: -0.0036814, -0.0017331, -0.0036524, -0.0017673, -0.0013910, 0.0011553
9: 0.0001457, 0.0024050, 0.0001854, 0.0023713, -0.0013396, 0.0016129

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A1_B2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 201

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014101, upper bound: 0.0013903
time: 1.30 seconds

## Relational analysis of NS_A1_B2_B2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014101, upper bound: 0.0014045
time: 1.26 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888747, 0.9922870, 0.9889209, 0.9922482, -0.0024443, 0.0020516
1: -0.0040361, -0.0031858, -0.0040246, -0.0031955, -0.0006090, 0.0005112
2: 0.0068292, 0.0113351, 0.0068806, 0.0112742, -0.0027091, 0.0032276
3: -0.0064324, -0.0043815, -0.0064046, -0.0044049, -0.0014691, 0.0012331
4: 0.0018497, 0.0027218, 0.0018596, 0.0027100, -0.0005243, 0.0006247
5: 0.0075488, 0.0132161, 0.0076134, 0.0131394, -0.0034073, 0.0040595
6: -0.0018135, -0.0003751, -0.0017941, -0.0003915, -0.0010303, 0.0008648
7: -0.0078298, -0.0041082, -0.0077794, -0.0041506, -0.0026658, 0.0022375
8: -0.0036818, -0.0017246, -0.0036553, -0.0017469, -0.0014019, 0.0011767
9: 0.0001359, 0.0024053, 0.0001618, 0.0023746, -0.0013644, 0.0016256

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009587, upper bound: 0.0009028
time: 0.87 seconds

## Relational analysis of NS_A1_B2_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007211, upper bound: 0.0007211
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9888041, 0.9918770, 0.9889757, 0.9923847, -0.0029148, 0.0018301
1: -0.0040537, -0.0032880, -0.0040109, -0.0031615, -0.0007263, 0.0004560
2: 0.0073706, 0.0114284, 0.0067002, 0.0112018, -0.0024166, 0.0038489
3: -0.0064748, -0.0046279, -0.0063717, -0.0043228, -0.0017519, 0.0011000
4: 0.0019545, 0.0027398, 0.0018247, 0.0026960, -0.0004677, 0.0007449
5: 0.0082298, 0.0133334, 0.0073866, 0.0130483, -0.0030395, 0.0048409
6: -0.0018433, -0.0005480, -0.0017710, -0.0003340, -0.0012287, 0.0007715
7: -0.0079069, -0.0045554, -0.0077197, -0.0040017, -0.0031790, 0.0019960
8: -0.0037223, -0.0019598, -0.0036238, -0.0016686, -0.0016718, 0.0010497
9: 0.0004086, 0.0024523, 0.0000710, 0.0023382, -0.0012172, 0.0019385

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011588, upper bound: 0.0010580
time: 1.00 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010591, upper bound: 0.0010039
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9888034, 0.9918913, 0.9889703, 0.9924167, -0.0029367, 0.0018584
1: -0.0040538, -0.0032844, -0.0040123, -0.0031535, -0.0007317, 0.0004631
2: 0.0073517, 0.0114292, 0.0066580, 0.0112090, -0.0024540, 0.0038779
3: -0.0064752, -0.0046193, -0.0063750, -0.0043035, -0.0017650, 0.0011169
4: 0.0019508, 0.0027400, 0.0018165, 0.0026974, -0.0004750, 0.0007506
5: 0.0082059, 0.0133343, 0.0073334, 0.0130574, -0.0030865, 0.0048773
6: -0.0018436, -0.0005419, -0.0017733, -0.0003205, -0.0012379, 0.0007834
7: -0.0079075, -0.0045397, -0.0077256, -0.0039668, -0.0032029, 0.0020268
8: -0.0037226, -0.0019516, -0.0036270, -0.0016502, -0.0016844, 0.0010659
9: 0.0003991, 0.0024527, 0.0000497, 0.0023418, -0.0012360, 0.0019531

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011738, upper bound: 0.0010670
time: 1.27 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010807, upper bound: 0.0010133
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9888752, 0.9922723, 0.9889757, 0.9923847, -0.0026603, 0.0021134
1: -0.0040359, -0.0031895, -0.0040109, -0.0031615, -0.0006629, 0.0005266
2: 0.0068487, 0.0113344, 0.0067002, 0.0112018, -0.0027907, 0.0035129
3: -0.0064320, -0.0043903, -0.0063717, -0.0043228, -0.0015989, 0.0012702
4: 0.0018534, 0.0027216, 0.0018247, 0.0026960, -0.0005401, 0.0006799
5: 0.0075733, 0.0132151, 0.0073866, 0.0130483, -0.0035099, 0.0044183
6: -0.0018133, -0.0003813, -0.0017710, -0.0003340, -0.0011214, 0.0008909
7: -0.0078292, -0.0041243, -0.0077197, -0.0040017, -0.0029014, 0.0023049
8: -0.0036814, -0.0017331, -0.0036238, -0.0016686, -0.0015258, 0.0012121
9: 0.0001457, 0.0024050, 0.0000710, 0.0023382, -0.0014055, 0.0017693

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009515, upper bound: 0.0009449
time: 0.90 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007969, upper bound: 0.0007559
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888747, 0.9922870, 0.9889703, 0.9924167, -0.0026804, 0.0021397
1: -0.0040361, -0.0031858, -0.0040123, -0.0031535, -0.0006679, 0.0005332
2: 0.0068292, 0.0113351, 0.0066580, 0.0112090, -0.0028255, 0.0035394
3: -0.0064324, -0.0043815, -0.0063750, -0.0043035, -0.0016110, 0.0012860
4: 0.0018497, 0.0027218, 0.0018165, 0.0026974, -0.0005469, 0.0006850
5: 0.0075488, 0.0132161, 0.0073334, 0.0130574, -0.0035537, 0.0044517
6: -0.0018135, -0.0003751, -0.0017733, -0.0003205, -0.0011299, 0.0009020
7: -0.0078298, -0.0041082, -0.0077256, -0.0039668, -0.0029233, 0.0023337
8: -0.0036818, -0.0017246, -0.0036270, -0.0016502, -0.0015374, 0.0012273
9: 0.0001359, 0.0024053, 0.0000497, 0.0023418, -0.0014231, 0.0017826

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009911, upper bound: 0.0009592
time: 1.05 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008607, upper bound: 0.0007854
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9892800, 0.9923536, 0.9892468, 0.9921501, -0.0019232, 0.0021835
1: -0.0039351, -0.0031692, -0.0039434, -0.0032200, -0.0004792, 0.0005441
2: 0.0067414, 0.0107999, 0.0070101, 0.0108438, -0.0028833, 0.0025396
3: -0.0061888, -0.0043415, -0.0062088, -0.0044638, -0.0011559, 0.0013124
4: 0.0018327, 0.0026182, 0.0018847, 0.0026267, -0.0005581, 0.0004915
5: 0.0074383, 0.0125429, 0.0077763, 0.0125981, -0.0036265, 0.0031941
6: -0.0016427, -0.0003471, -0.0016567, -0.0004329, -0.0008107, 0.0009204
7: -0.0073878, -0.0040357, -0.0074240, -0.0042576, -0.0020975, 0.0023815
8: -0.0034493, -0.0016865, -0.0034684, -0.0018032, -0.0011031, 0.0012524
9: 0.0000917, 0.0021358, 0.0002270, 0.0021579, -0.0014522, 0.0012791

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014351
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014505
time: 1.43 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9892800, 0.9923536, 0.9888493, 0.9919039, -0.0019162, 0.0027348
1: -0.0039351, -0.0031692, -0.0040424, -0.0032813, -0.0004775, 0.0006814
2: 0.0067414, 0.0107999, 0.0073351, 0.0113688, -0.0036113, 0.0025304
3: -0.0061888, -0.0043415, -0.0064477, -0.0046117, -0.0011517, 0.0016437
4: 0.0018327, 0.0026182, 0.0019476, 0.0027283, -0.0006990, 0.0004897
5: 0.0074383, 0.0125429, 0.0081850, 0.0132584, -0.0045420, 0.0031825
6: -0.0016427, -0.0003471, -0.0018243, -0.0005366, -0.0008078, 0.0011528
7: -0.0073878, -0.0040357, -0.0078576, -0.0045260, -0.0020899, 0.0029827
8: -0.0034493, -0.0016865, -0.0036964, -0.0019443, -0.0010991, 0.0015686
9: 0.0000917, 0.0021358, 0.0003907, 0.0024223, -0.0018188, 0.0012744

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014978
time: 1.36 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0015129
time: 1.36 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9888805, 0.9921045, 0.9892468, 0.9921501, -0.0024046, 0.0020475
1: -0.0040346, -0.0032313, -0.0039434, -0.0032200, -0.0005992, 0.0005102
2: 0.0070703, 0.0113274, 0.0070101, 0.0108438, -0.0027037, 0.0031753
3: -0.0064289, -0.0044912, -0.0062088, -0.0044638, -0.0014453, 0.0012306
4: 0.0018963, 0.0027203, 0.0018847, 0.0026267, -0.0005233, 0.0006146
5: 0.0078520, 0.0132064, 0.0077763, 0.0125981, -0.0034005, 0.0039937
6: -0.0018111, -0.0004521, -0.0016567, -0.0004329, -0.0010136, 0.0008631
7: -0.0078234, -0.0043073, -0.0074240, -0.0042576, -0.0026226, 0.0022331
8: -0.0036784, -0.0018293, -0.0034684, -0.0018032, -0.0013792, 0.0011743
9: 0.0002574, 0.0024015, 0.0002270, 0.0021579, -0.0013617, 0.0015992

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014503, upper bound: 0.0014502
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014664, upper bound: 0.0014505
time: 1.32 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888805, 0.9921045, 0.9888493, 0.9919039, -0.0020149, 0.0022717
1: -0.0040346, -0.0032313, -0.0040424, -0.0032813, -0.0005021, 0.0005661
2: 0.0070703, 0.0113274, 0.0073351, 0.0113688, -0.0029998, 0.0026607
3: -0.0064289, -0.0044912, -0.0064477, -0.0046117, -0.0012110, 0.0013654
4: 0.0018963, 0.0027203, 0.0019476, 0.0027283, -0.0005806, 0.0005150
5: 0.0078520, 0.0132064, 0.0081850, 0.0132584, -0.0037729, 0.0033464
6: -0.0018111, -0.0004521, -0.0018243, -0.0005366, -0.0008494, 0.0009576
7: -0.0078234, -0.0043073, -0.0078576, -0.0045260, -0.0021976, 0.0024776
8: -0.0036784, -0.0018293, -0.0036964, -0.0019443, -0.0011557, 0.0013030
9: 0.0002574, 0.0024015, 0.0003907, 0.0024223, -0.0015108, 0.0013401

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014645, upper bound: 0.0014594
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014664, upper bound: 0.0014759
time: 1.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: 0.9892911, 0.9922644, 0.9890239, 0.9922627, -0.0023126, 0.0024160
1: -0.0039323, -0.0031914, -0.0039989, -0.0031919, -0.0005762, 0.0006020
2: 0.0068589, 0.0107853, 0.0068614, 0.0111382, -0.0031903, 0.0030537
3: -0.0061821, -0.0043950, -0.0063427, -0.0043961, -0.0013899, 0.0014521
4: 0.0018554, 0.0026154, 0.0018559, 0.0026837, -0.0006175, 0.0005910
5: 0.0075861, 0.0125245, 0.0075893, 0.0129684, -0.0040126, 0.0038408
6: -0.0016380, -0.0003846, -0.0017507, -0.0003854, -0.0009748, 0.0010184
7: -0.0073757, -0.0041327, -0.0076672, -0.0041348, -0.0025222, 0.0026350
8: -0.0034430, -0.0017375, -0.0035962, -0.0017386, -0.0013264, 0.0013857
9: 0.0001509, 0.0021284, 0.0001521, 0.0023062, -0.0016068, 0.0015380

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A2_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B2_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010629, upper bound: 0.0013595
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010629, upper bound: 0.0013595
time: 1.12 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: 0.9892840, 0.9923024, 0.9890233, 0.9922774, -0.0023218, 0.0024323
1: -0.0039341, -0.0031820, -0.0039991, -0.0031882, -0.0005785, 0.0006061
2: 0.0068089, 0.0107946, 0.0068420, 0.0111389, -0.0032119, 0.0030659
3: -0.0061864, -0.0043722, -0.0063431, -0.0043873, -0.0013955, 0.0014619
4: 0.0018457, 0.0026172, 0.0018521, 0.0026838, -0.0006217, 0.0005934
5: 0.0075232, 0.0125362, 0.0075649, 0.0129693, -0.0040397, 0.0038561
6: -0.0016410, -0.0003686, -0.0017509, -0.0003792, -0.0009787, 0.0010253
7: -0.0073834, -0.0040914, -0.0076678, -0.0041188, -0.0025322, 0.0026528
8: -0.0034470, -0.0017158, -0.0035966, -0.0017302, -0.0013317, 0.0013951
9: 0.0001257, 0.0021331, 0.0001424, 0.0023065, -0.0016177, 0.0015441

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A2_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010629, upper bound: 0.0013692
time: 1.08 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010629, upper bound: 0.0013694
time: 1.37 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: 0.9888886, 0.9920204, 0.9888752, 0.9922723, -0.0024558, 0.0025182
1: -0.0040326, -0.0032523, -0.0040359, -0.0031895, -0.0006119, 0.0006275
2: 0.0071812, 0.0113169, 0.0068487, 0.0113344, -0.0033253, 0.0032429
3: -0.0064241, -0.0045417, -0.0064320, -0.0043903, -0.0014760, 0.0015135
4: 0.0019178, 0.0027182, 0.0018534, 0.0027216, -0.0006436, 0.0006277
5: 0.0079916, 0.0131931, 0.0075733, 0.0132151, -0.0041824, 0.0040787
6: -0.0018077, -0.0004875, -0.0018133, -0.0003813, -0.0010352, 0.0010615
7: -0.0078147, -0.0043990, -0.0078292, -0.0041243, -0.0026784, 0.0027465
8: -0.0036738, -0.0018775, -0.0036814, -0.0017331, -0.0014086, 0.0014444
9: 0.0003132, 0.0023962, 0.0001457, 0.0024050, -0.0016748, 0.0016333

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A1_B2_A2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009965, upper bound: 0.0010031
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_A1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008650, upper bound: 0.0009246
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: 0.9888846, 0.9920495, 0.9888747, 0.9922870, -0.0024760, 0.0025419
1: -0.0040336, -0.0032450, -0.0040361, -0.0031858, -0.0006170, 0.0006334
2: 0.0071429, 0.0113221, 0.0068292, 0.0113351, -0.0033565, 0.0032696
3: -0.0064264, -0.0045243, -0.0064324, -0.0043815, -0.0014882, 0.0015277
4: 0.0019104, 0.0027193, 0.0018497, 0.0027218, -0.0006496, 0.0006328
5: 0.0079434, 0.0131997, 0.0075488, 0.0132161, -0.0042216, 0.0041123
6: -0.0018094, -0.0004753, -0.0018135, -0.0003751, -0.0010437, 0.0010715
7: -0.0078190, -0.0043673, -0.0078298, -0.0041082, -0.0027005, 0.0027723
8: -0.0036761, -0.0018609, -0.0036818, -0.0017246, -0.0014201, 0.0014579
9: 0.0002939, 0.0023988, 0.0001359, 0.0024053, -0.0016905, 0.0016467

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A1_B2_A2_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010066, upper bound: 0.0010583
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008854, upper bound: 0.0009879
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9889757, 0.9923847, 0.9888041, 0.9918770, -0.0018301, 0.0029148
1: -0.0040109, -0.0031615, -0.0040537, -0.0032880, -0.0004560, 0.0007263
2: 0.0067002, 0.0112018, 0.0073706, 0.0114284, -0.0038489, 0.0024166
3: -0.0063717, -0.0043228, -0.0064748, -0.0046279, -0.0011000, 0.0017519
4: 0.0018247, 0.0026960, 0.0019545, 0.0027398, -0.0007449, 0.0004677
5: 0.0073866, 0.0130483, 0.0082298, 0.0133334, -0.0048409, 0.0030395
6: -0.0017710, -0.0003340, -0.0018433, -0.0005480, -0.0007715, 0.0012287
7: -0.0077197, -0.0040017, -0.0079069, -0.0045554, -0.0019960, 0.0031790
8: -0.0036238, -0.0016686, -0.0037223, -0.0019598, -0.0010497, 0.0016718
9: 0.0000710, 0.0023382, 0.0004086, 0.0024523, -0.0019385, 0.0012172

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010580, upper bound: 0.0011588
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010039, upper bound: 0.0010591
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9889703, 0.9924167, 0.9888034, 0.9918913, -0.0018584, 0.0029367
1: -0.0040123, -0.0031535, -0.0040538, -0.0032844, -0.0004631, 0.0007317
2: 0.0066580, 0.0112090, 0.0073517, 0.0114292, -0.0038779, 0.0024540
3: -0.0063750, -0.0043035, -0.0064752, -0.0046193, -0.0011169, 0.0017650
4: 0.0018165, 0.0026974, 0.0019508, 0.0027400, -0.0007506, 0.0004750
5: 0.0073334, 0.0130574, 0.0082059, 0.0133343, -0.0048773, 0.0030865
6: -0.0017733, -0.0003205, -0.0018436, -0.0005419, -0.0007834, 0.0012379
7: -0.0077256, -0.0039668, -0.0079075, -0.0045397, -0.0020268, 0.0032029
8: -0.0036270, -0.0016502, -0.0037226, -0.0019516, -0.0010659, 0.0016844
9: 0.0000497, 0.0023418, 0.0003991, 0.0024527, -0.0019531, 0.0012360

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010670, upper bound: 0.0011738
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010133, upper bound: 0.0010807
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9889757, 0.9923847, 0.9888752, 0.9922723, -0.0021134, 0.0026603
1: -0.0040109, -0.0031615, -0.0040359, -0.0031895, -0.0005266, 0.0006629
2: 0.0067002, 0.0112018, 0.0068487, 0.0113344, -0.0035129, 0.0027907
3: -0.0063717, -0.0043228, -0.0064320, -0.0043903, -0.0012702, 0.0015989
4: 0.0018247, 0.0026960, 0.0018534, 0.0027216, -0.0006799, 0.0005401
5: 0.0073866, 0.0130483, 0.0075733, 0.0132151, -0.0044183, 0.0035099
6: -0.0017710, -0.0003340, -0.0018133, -0.0003813, -0.0008909, 0.0011214
7: -0.0077197, -0.0040017, -0.0078292, -0.0041243, -0.0023049, 0.0029014
8: -0.0036238, -0.0016686, -0.0036814, -0.0017331, -0.0012121, 0.0015258
9: 0.0000710, 0.0023382, 0.0001457, 0.0024050, -0.0017693, 0.0014055

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009449, upper bound: 0.0009515
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007559, upper bound: 0.0007969
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9889703, 0.9924167, 0.9888747, 0.9922870, -0.0021397, 0.0026804
1: -0.0040123, -0.0031535, -0.0040361, -0.0031858, -0.0005332, 0.0006679
2: 0.0066580, 0.0112090, 0.0068292, 0.0113351, -0.0035394, 0.0028255
3: -0.0063750, -0.0043035, -0.0064324, -0.0043815, -0.0012860, 0.0016110
4: 0.0018165, 0.0026974, 0.0018497, 0.0027218, -0.0006850, 0.0005469
5: 0.0073334, 0.0130574, 0.0075488, 0.0132161, -0.0044517, 0.0035537
6: -0.0017733, -0.0003205, -0.0018135, -0.0003751, -0.0009020, 0.0011299
7: -0.0077256, -0.0039668, -0.0078298, -0.0041082, -0.0023337, 0.0029233
8: -0.0036270, -0.0016502, -0.0036818, -0.0017246, -0.0012273, 0.0015374
9: 0.0000497, 0.0023418, 0.0001359, 0.0024053, -0.0017826, 0.0014231

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009592, upper bound: 0.0009911
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007854, upper bound: 0.0008607
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9892800, 0.9923536, 0.9892800, 0.9923536, -0.0018743, 0.0018743
1: -0.0039351, -0.0031692, -0.0039351, -0.0031692, -0.0004670, 0.0004670
2: 0.0067414, 0.0107999, 0.0067414, 0.0107999, -0.0024750, 0.0024750
3: -0.0061888, -0.0043415, -0.0061888, -0.0043415, -0.0011265, 0.0011265
4: 0.0018327, 0.0026182, 0.0018327, 0.0026182, -0.0004790, 0.0004790
5: 0.0074383, 0.0125429, 0.0074383, 0.0125429, -0.0031129, 0.0031129
6: -0.0016427, -0.0003471, -0.0016427, -0.0003471, -0.0007901, 0.0007901
7: -0.0073878, -0.0040357, -0.0073878, -0.0040357, -0.0020442, 0.0020442
8: -0.0034493, -0.0016865, -0.0034493, -0.0016865, -0.0010750, 0.0010750
9: 0.0000917, 0.0021358, 0.0000917, 0.0021358, -0.0012465, 0.0012465

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014084, upper bound: 0.0014353
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014085, upper bound: 0.0014508
time: 1.36 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9892800, 0.9923536, 0.9888805, 0.9921045, -0.0018663, 0.0024318
1: -0.0039351, -0.0031692, -0.0040346, -0.0032313, -0.0004650, 0.0006060
2: 0.0067414, 0.0107999, 0.0070703, 0.0113274, -0.0032112, 0.0024644
3: -0.0061888, -0.0043415, -0.0064289, -0.0044912, -0.0011217, 0.0014616
4: 0.0018327, 0.0026182, 0.0018963, 0.0027203, -0.0006215, 0.0004770
5: 0.0074383, 0.0125429, 0.0078520, 0.0132064, -0.0040389, 0.0030996
6: -0.0016427, -0.0003471, -0.0018111, -0.0004521, -0.0007867, 0.0010251
7: -0.0073878, -0.0040357, -0.0078234, -0.0043073, -0.0020355, 0.0026523
8: -0.0034493, -0.0016865, -0.0036784, -0.0018293, -0.0010704, 0.0013948
9: 0.0000917, 0.0021358, 0.0002574, 0.0024015, -0.0016173, 0.0012412

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014084, upper bound: 0.0014981
time: 1.32 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014085, upper bound: 0.0015131
time: 1.31 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9888805, 0.9921045, 0.9892800, 0.9923536, -0.0024318, 0.0018663
1: -0.0040346, -0.0032313, -0.0039351, -0.0031692, -0.0006060, 0.0004650
2: 0.0070703, 0.0113274, 0.0067414, 0.0107999, -0.0024644, 0.0032112
3: -0.0064289, -0.0044912, -0.0061888, -0.0043415, -0.0014616, 0.0011217
4: 0.0018963, 0.0027203, 0.0018327, 0.0026182, -0.0004770, 0.0006215
5: 0.0078520, 0.0132064, 0.0074383, 0.0125429, -0.0030996, 0.0040389
6: -0.0018111, -0.0004521, -0.0016427, -0.0003471, -0.0010251, 0.0007867
7: -0.0078234, -0.0043073, -0.0073878, -0.0040357, -0.0026523, 0.0020355
8: -0.0036784, -0.0018293, -0.0034493, -0.0016865, -0.0013948, 0.0010704
9: 0.0002574, 0.0024015, 0.0000917, 0.0021358, -0.0012412, 0.0016173

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014528, upper bound: 0.0014506
time: 1.35 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014682, upper bound: 0.0014507
time: 1.35 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888805, 0.9921045, 0.9888805, 0.9921045, -0.0019785, 0.0019785
1: -0.0040346, -0.0032313, -0.0040346, -0.0032313, -0.0004930, 0.0004930
2: 0.0070703, 0.0113274, 0.0070703, 0.0113274, -0.0026126, 0.0026126
3: -0.0064289, -0.0044912, -0.0064289, -0.0044912, -0.0011891, 0.0011891
4: 0.0018963, 0.0027203, 0.0018963, 0.0027203, -0.0005057, 0.0005057
5: 0.0078520, 0.0132064, 0.0078520, 0.0132064, -0.0032860, 0.0032860
6: -0.0018111, -0.0004521, -0.0018111, -0.0004521, -0.0008340, 0.0008340
7: -0.0078234, -0.0043073, -0.0078234, -0.0043073, -0.0021578, 0.0021578
8: -0.0036784, -0.0018293, -0.0036784, -0.0018293, -0.0011348, 0.0011348
9: 0.0002574, 0.0024015, 0.0002574, 0.0024015, -0.0013158, 0.0013158

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014666, upper bound: 0.0014594
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014682, upper bound: 0.0014759
time: 1.38 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9892800, 0.9923536, 0.9889671, 0.9924710, -0.0023173, 0.0024082
1: -0.0039351, -0.0031692, -0.0040131, -0.0031400, -0.0005774, 0.0006001
2: 0.0067414, 0.0107999, 0.0065863, 0.0112132, -0.0031800, 0.0030600
3: -0.0061888, -0.0043415, -0.0063769, -0.0042709, -0.0013928, 0.0014474
4: 0.0018327, 0.0026182, 0.0018026, 0.0026982, -0.0006155, 0.0005922
5: 0.0074383, 0.0125429, 0.0072432, 0.0130627, -0.0039996, 0.0038486
6: -0.0016427, -0.0003471, -0.0017746, -0.0002976, -0.0009768, 0.0010151
7: -0.0073878, -0.0040357, -0.0077291, -0.0039075, -0.0025273, 0.0026265
8: -0.0034493, -0.0016865, -0.0036288, -0.0016191, -0.0013291, 0.0013812
9: 0.0000917, 0.0021358, 0.0000136, 0.0023439, -0.0016016, 0.0015412

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011541, upper bound: 0.0011935
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011539, upper bound: 0.0014139
time: 1.31 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9888805, 0.9921045, 0.9894029, 0.9926836, -0.0029508, 0.0018424
1: -0.0040346, -0.0032313, -0.0039045, -0.0030870, -0.0007353, 0.0004591
2: 0.0070703, 0.0113274, 0.0063055, 0.0106377, -0.0024329, 0.0038966
3: -0.0064289, -0.0044912, -0.0061149, -0.0041431, -0.0017735, 0.0011074
4: 0.0018963, 0.0027203, 0.0017483, 0.0025868, -0.0004709, 0.0007542
5: 0.0078520, 0.0132064, 0.0068902, 0.0123389, -0.0030600, 0.0049009
6: -0.0018111, -0.0004521, -0.0015909, -0.0002080, -0.0012439, 0.0007766
7: -0.0078234, -0.0043073, -0.0072538, -0.0036757, -0.0032183, 0.0020094
8: -0.0036784, -0.0018293, -0.0033788, -0.0014972, -0.0016925, 0.0010567
9: 0.0002574, 0.0024015, -0.0001278, 0.0020541, -0.0012253, 0.0019625

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013135, upper bound: 0.0012086
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0012086
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888805, 0.9921045, 0.9889671, 0.9924710, -0.0024733, 0.0019295
1: -0.0040346, -0.0032313, -0.0040131, -0.0031400, -0.0006163, 0.0004808
2: 0.0070703, 0.0113274, 0.0065863, 0.0112132, -0.0025478, 0.0032659
3: -0.0064289, -0.0044912, -0.0063769, -0.0042709, -0.0014865, 0.0011597
4: 0.0018963, 0.0027203, 0.0018026, 0.0026982, -0.0004931, 0.0006321
5: 0.0078520, 0.0132064, 0.0072432, 0.0130627, -0.0032045, 0.0041077
6: -0.0018111, -0.0004521, -0.0017746, -0.0002976, -0.0010426, 0.0008133
7: -0.0078234, -0.0043073, -0.0077291, -0.0039075, -0.0026975, 0.0021043
8: -0.0036784, -0.0018293, -0.0036288, -0.0016191, -0.0014186, 0.0011067
9: 0.0002574, 0.0024015, 0.0000136, 0.0023439, -0.0012832, 0.0016449

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0014181
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0014365
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9894029, 0.9926836, 0.9888805, 0.9921045, -0.0018424, 0.0029508
1: -0.0039045, -0.0030870, -0.0040346, -0.0032313, -0.0004591, 0.0007353
2: 0.0063055, 0.0106377, 0.0070703, 0.0113274, -0.0038966, 0.0024329
3: -0.0061149, -0.0041431, -0.0064289, -0.0044912, -0.0011074, 0.0017735
4: 0.0017483, 0.0025868, 0.0018963, 0.0027203, -0.0007542, 0.0004709
5: 0.0068902, 0.0123389, 0.0078520, 0.0132064, -0.0049009, 0.0030600
6: -0.0015909, -0.0002080, -0.0018111, -0.0004521, -0.0007766, 0.0012439
7: -0.0072538, -0.0036757, -0.0078234, -0.0043073, -0.0020094, 0.0032183
8: -0.0033788, -0.0014972, -0.0036784, -0.0018293, -0.0010567, 0.0016925
9: -0.0001278, 0.0020541, 0.0002574, 0.0024015, -0.0019625, 0.0012253

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011137, upper bound: 0.0013402
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011137, upper bound: 0.0013440
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9889671, 0.9924710, 0.9888805, 0.9921045, -0.0019295, 0.0024733
1: -0.0040131, -0.0031400, -0.0040346, -0.0032313, -0.0004808, 0.0006163
2: 0.0065863, 0.0112132, 0.0070703, 0.0113274, -0.0032659, 0.0025478
3: -0.0063769, -0.0042709, -0.0064289, -0.0044912, -0.0011597, 0.0014865
4: 0.0018026, 0.0026982, 0.0018963, 0.0027203, -0.0006321, 0.0004931
5: 0.0072432, 0.0130627, 0.0078520, 0.0132064, -0.0041077, 0.0032045
6: -0.0017746, -0.0002976, -0.0018111, -0.0004521, -0.0008133, 0.0010426
7: -0.0077291, -0.0039075, -0.0078234, -0.0043073, -0.0021043, 0.0026975
8: -0.0036288, -0.0016191, -0.0036784, -0.0018293, -0.0011067, 0.0014186
9: 0.0000136, 0.0023439, 0.0002574, 0.0024015, -0.0016449, 0.0012832

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014431, upper bound: 0.0014248
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014568, upper bound: 0.0014249
time: 1.27 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9889671, 0.9924710, 0.9889671, 0.9924710, -0.0021167, 0.0021167
1: -0.0040131, -0.0031400, -0.0040131, -0.0031400, -0.0005274, 0.0005274
2: 0.0065863, 0.0112132, 0.0065863, 0.0112132, -0.0027951, 0.0027951
3: -0.0063769, -0.0042709, -0.0063769, -0.0042709, -0.0012722, 0.0012722
4: 0.0018026, 0.0026982, 0.0018026, 0.0026982, -0.0005410, 0.0005410
5: 0.0072432, 0.0130627, 0.0072432, 0.0130627, -0.0035155, 0.0035155
6: -0.0017746, -0.0002976, -0.0017746, -0.0002976, -0.0008923, 0.0008923
7: -0.0077291, -0.0039075, -0.0077291, -0.0039075, -0.0023086, 0.0023086
8: -0.0036288, -0.0016191, -0.0036288, -0.0016191, -0.0012141, 0.0012141
9: 0.0000136, 0.0023439, 0.0000136, 0.0023439, -0.0014078, 0.0014078

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014568, upper bound: 0.0014085
time: 1.22 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014568, upper bound: 0.0014249
time: 1.28 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.00 seconds
NS_A1_B1_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0013840
NS_A1_B1_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014031
NS_A1_B1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0013840, upper bound: 0.0014031
NS_A1_B1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014031
NS_A1_B1_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014351, upper bound: 0.0014031
NS_A1_B1_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014505, upper bound: 0.0014031
NS_A1_B1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014351, upper bound: 0.0014031
NS_A1_B1_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014505, upper bound: 0.0014031
NS_A1_B1_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014510
NS_A1_B1_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014668
NS_A1_B1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014100
NS_A1_B1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014311
NS_A1_B1_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014502, upper bound: 0.0014503
NS_A1_B1_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014505, upper bound: 0.0014664
NS_A1_B1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014351, upper bound: 0.0014299
NS_A1_B1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014505, upper bound: 0.0014311
NS_A1_B1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0013063, upper bound: 0.0010629
NS_A1_B1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0013063, upper bound: 0.0010764
NS_A1_B1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0013159, upper bound: 0.0010629
NS_A1_B1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0013159, upper bound: 0.0010764
NS_A1_B1_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0013595, upper bound: 0.0010629
NS_A1_B1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0013595, upper bound: 0.0010764
NS_A1_B1_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0013692, upper bound: 0.0010629
NS_A1_B1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0013692, upper bound: 0.0010764
NS_A1_B1_A2_B2_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0009394, upper bound: 0.0009965
NS_A1_B1_A2_B2_B1_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0008584, upper bound: 0.0008620
NS_A1_B1_A2_B2_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0009970, upper bound: 0.0010066
NS_A1_B1_A2_B2_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0009256, upper bound: 0.0008846
NS_A1_B1_A2_B2_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0010031, upper bound: 0.0009965
NS_A1_B1_A2_B2_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0009246, upper bound: 0.0008650
NS_A1_B1_A2_B2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0010583, upper bound: 0.0010066
NS_A1_B1_A2_B2_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0009879, upper bound: 0.0008854
NS_A1_B2_B2_B1_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0009965, upper bound: 0.0009394
NS_A1_B2_B2_B1_A1_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0008620, upper bound: 0.0008584
NS_A1_B2_B2_B1_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0010066, upper bound: 0.0009970
NS_A1_B2_B2_B1_A1_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0008846, upper bound: 0.0009256
NS_A1_B2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014101, upper bound: 0.0013903
NS_A1_B2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014101, upper bound: 0.0014045
NS_A1_B2_B2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0009587, upper bound: 0.0009028
NS_A1_B2_B2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0007211, upper bound: 0.0007211
NS_A1_B2_B2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0011588, upper bound: 0.0010580
NS_A1_B2_B2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0010591, upper bound: 0.0010039
NS_A1_B2_B2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0011738, upper bound: 0.0010670
NS_A1_B2_B2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0010807, upper bound: 0.0010133
NS_A1_B2_B2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0009515, upper bound: 0.0009449
NS_A1_B2_B2_B2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0007969, upper bound: 0.0007559
NS_A1_B2_B2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0009911, upper bound: 0.0009592
NS_A1_B2_B2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0008607, upper bound: 0.0007854
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014351
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014505
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014978
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0015129
NS_A2_B1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014503, upper bound: 0.0014502
NS_A2_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014664, upper bound: 0.0014505
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014645, upper bound: 0.0014594
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014664, upper bound: 0.0014759
NS_A2_B1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0010629, upper bound: 0.0013595
NS_A2_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0010629, upper bound: 0.0013595
NS_A2_B1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0010629, upper bound: 0.0013692
NS_A2_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0010629, upper bound: 0.0013694
NS_A2_B1_A1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0009965, upper bound: 0.0010031
NS_A2_B1_A1_B2_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0008650, upper bound: 0.0009246
NS_A2_B1_A1_B2_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0010066, upper bound: 0.0010583
NS_A2_B1_A1_B2_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0008854, upper bound: 0.0009879
NS_A2_B1_A2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0010580, upper bound: 0.0011588
NS_A2_B1_A2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0010039, upper bound: 0.0010591
NS_A2_B1_A2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0010670, upper bound: 0.0011738
NS_A2_B1_A2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0010133, upper bound: 0.0010807
NS_A2_B1_A2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0009449, upper bound: 0.0009515
NS_A2_B1_A2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0007559, upper bound: 0.0007969
NS_A2_B1_A2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0009592, upper bound: 0.0009911
NS_A2_B1_A2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0007854, upper bound: 0.0008607
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014084, upper bound: 0.0014353
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014085, upper bound: 0.0014508
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014084, upper bound: 0.0014981
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014085, upper bound: 0.0015131
NS_A2_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014528, upper bound: 0.0014506
NS_A2_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014682, upper bound: 0.0014507
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014666, upper bound: 0.0014594
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014682, upper bound: 0.0014759
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0011541, upper bound: 0.0011935
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0011539, upper bound: 0.0014139
NS_A2_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0013135, upper bound: 0.0012086
NS_A2_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0012086
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0014181
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0014365
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0011137, upper bound: 0.0013402
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0011137, upper bound: 0.0013440
NS_A2_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014431, upper bound: 0.0014248
NS_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014568, upper bound: 0.0014249
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014568, upper bound: 0.0014085
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 0, lower bound: -0.0014568, upper bound: 0.0014249

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.9892615, 0.9920586, 0.9892485, 0.9921194, -0.0017420, 0.0017166
1: -0.0039397, -0.0032427, -0.0039429, -0.0032276, -0.0004341, 0.0004277
2: 0.0071307, 0.0108243, 0.0070506, 0.0108415, -0.0022668, 0.0023002
3: -0.0061999, -0.0045187, -0.0062077, -0.0044822, -0.0010470, 0.0010318
4: 0.0019080, 0.0026229, 0.0018925, 0.0026262, -0.0004387, 0.0004452
5: 0.0079280, 0.0125736, 0.0078273, 0.0125952, -0.0028510, 0.0028931
6: -0.0016505, -0.0004714, -0.0016560, -0.0004458, -0.0007343, 0.0007236
7: -0.0074079, -0.0043572, -0.0074221, -0.0042911, -0.0018999, 0.0018722
8: -0.0034599, -0.0018556, -0.0034674, -0.0018208, -0.0009991, 0.0009846
9: 0.0002878, 0.0021481, 0.0002474, 0.0021567, -0.0011417, 0.0011585

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011422, upper bound: 0.0012111
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013833, upper bound: 0.0013638
time: 1.22 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.9892507, 0.9921016, 0.9892479, 0.9921356, -0.0017823, 0.0017166
1: -0.0039424, -0.0032320, -0.0039431, -0.0032236, -0.0004441, 0.0004277
2: 0.0070741, 0.0108385, 0.0070292, 0.0108423, -0.0022667, 0.0023534
3: -0.0062063, -0.0044929, -0.0062081, -0.0044725, -0.0010712, 0.0010317
4: 0.0018971, 0.0026257, 0.0018884, 0.0026264, -0.0004387, 0.0004555
5: 0.0078568, 0.0125914, 0.0078004, 0.0125962, -0.0028509, 0.0029600
6: -0.0016550, -0.0004533, -0.0016562, -0.0004390, -0.0007513, 0.0007236
7: -0.0074196, -0.0043105, -0.0074228, -0.0042734, -0.0019438, 0.0018721
8: -0.0034661, -0.0018310, -0.0034677, -0.0018115, -0.0010222, 0.0009845
9: 0.0002593, 0.0021552, 0.0002367, 0.0021571, -0.0011416, 0.0011853

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011394, upper bound: 0.0012286
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013834, upper bound: 0.0013834
time: 1.36 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9888510, 0.9918745, 0.9892615, 0.9920586, -0.0022668, 0.0017423
1: -0.0040420, -0.0032886, -0.0039397, -0.0032427, -0.0005648, 0.0004341
2: 0.0073740, 0.0113665, 0.0071307, 0.0108243, -0.0023007, 0.0029933
3: -0.0064466, -0.0046294, -0.0061999, -0.0045187, -0.0013624, 0.0010472
4: 0.0019551, 0.0027278, 0.0019080, 0.0026229, -0.0004453, 0.0005793
5: 0.0082340, 0.0132555, 0.0079280, 0.0125736, -0.0028937, 0.0037648
6: -0.0018235, -0.0005490, -0.0016505, -0.0004714, -0.0009555, 0.0007344
7: -0.0078557, -0.0045582, -0.0074079, -0.0043572, -0.0024723, 0.0019002
8: -0.0036954, -0.0019612, -0.0034599, -0.0018556, -0.0013002, 0.0009993
9: 0.0004103, 0.0024211, 0.0002878, 0.0021481, -0.0011587, 0.0015076

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012882, upper bound: 0.0012292
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014297, upper bound: 0.0013833
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888504, 0.9918888, 0.9892507, 0.9921016, -0.0022887, 0.0017677
1: -0.0040421, -0.0032850, -0.0039424, -0.0032320, -0.0005703, 0.0004405
2: 0.0073550, 0.0113672, 0.0070741, 0.0108385, -0.0023342, 0.0030222
3: -0.0064470, -0.0046208, -0.0062063, -0.0044929, -0.0013756, 0.0010624
4: 0.0019514, 0.0027280, 0.0018971, 0.0026257, -0.0004518, 0.0005849
5: 0.0082101, 0.0132564, 0.0078568, 0.0125914, -0.0029358, 0.0038011
6: -0.0018238, -0.0005430, -0.0016550, -0.0004533, -0.0009648, 0.0007451
7: -0.0078563, -0.0045425, -0.0074196, -0.0043105, -0.0024961, 0.0019279
8: -0.0036957, -0.0019530, -0.0034661, -0.0018310, -0.0013127, 0.0010139
9: 0.0004008, 0.0024215, 0.0002593, 0.0021552, -0.0011756, 0.0015221

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012901, upper bound: 0.0012286
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014457, upper bound: 0.0013834
time: 1.27 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9892485, 0.9921194, 0.9892911, 0.9922644, -0.0020905, 0.0018584
1: -0.0039429, -0.0032276, -0.0039323, -0.0031914, -0.0005209, 0.0004631
2: 0.0070506, 0.0108415, 0.0068589, 0.0107853, -0.0024540, 0.0027605
3: -0.0062077, -0.0044822, -0.0061821, -0.0043950, -0.0012565, 0.0011170
4: 0.0018925, 0.0026262, 0.0018554, 0.0026154, -0.0004750, 0.0005343
5: 0.0078273, 0.0125952, 0.0075861, 0.0125245, -0.0030865, 0.0034720
6: -0.0016560, -0.0004458, -0.0016380, -0.0003846, -0.0008812, 0.0007834
7: -0.0074221, -0.0042911, -0.0073757, -0.0041327, -0.0022800, 0.0020269
8: -0.0034674, -0.0018208, -0.0034430, -0.0017375, -0.0011990, 0.0010659
9: 0.0002474, 0.0021567, 0.0001509, 0.0021284, -0.0012360, 0.0013903

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012731, upper bound: 0.0011422
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014150, upper bound: 0.0013833
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9892479, 0.9921356, 0.9892840, 0.9923024, -0.0020931, 0.0018855
1: -0.0039431, -0.0032236, -0.0039341, -0.0031820, -0.0005215, 0.0004698
2: 0.0070292, 0.0108423, 0.0068089, 0.0107946, -0.0024898, 0.0027639
3: -0.0062081, -0.0044725, -0.0061864, -0.0043722, -0.0012580, 0.0011332
4: 0.0018884, 0.0026264, 0.0018457, 0.0026172, -0.0004819, 0.0005349
5: 0.0078004, 0.0125962, 0.0075232, 0.0125362, -0.0031315, 0.0034762
6: -0.0016562, -0.0004390, -0.0016410, -0.0003686, -0.0008823, 0.0007948
7: -0.0074228, -0.0042734, -0.0073834, -0.0040914, -0.0022828, 0.0020564
8: -0.0034677, -0.0018115, -0.0034470, -0.0017158, -0.0012005, 0.0010814
9: 0.0002367, 0.0021571, 0.0001257, 0.0021331, -0.0012540, 0.0013920

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012871, upper bound: 0.0011394
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014304, upper bound: 0.0013834
time: 1.30 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9888510, 0.9918745, 0.9892911, 0.9922644, -0.0026407, 0.0018588
1: -0.0040420, -0.0032886, -0.0039323, -0.0031914, -0.0006580, 0.0004632
2: 0.0073740, 0.0113665, 0.0068589, 0.0107853, -0.0024545, 0.0034870
3: -0.0064466, -0.0046294, -0.0061821, -0.0043950, -0.0015871, 0.0011172
4: 0.0019551, 0.0027278, 0.0018554, 0.0026154, -0.0004751, 0.0006749
5: 0.0082340, 0.0132555, 0.0075861, 0.0125245, -0.0030871, 0.0043858
6: -0.0018235, -0.0005490, -0.0016380, -0.0003846, -0.0011132, 0.0007835
7: -0.0078557, -0.0045582, -0.0073757, -0.0041327, -0.0028801, 0.0020272
8: -0.0036954, -0.0019612, -0.0034430, -0.0017375, -0.0015146, 0.0010661
9: 0.0004103, 0.0024211, 0.0001509, 0.0021284, -0.0012362, 0.0017563

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012174, upper bound: 0.0009102
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010741, upper bound: 0.0008091
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888504, 0.9918888, 0.9892840, 0.9923024, -0.0026652, 0.0018709
1: -0.0040421, -0.0032850, -0.0039341, -0.0031820, -0.0006641, 0.0004662
2: 0.0073550, 0.0113672, 0.0068089, 0.0107946, -0.0024705, 0.0035194
3: -0.0064470, -0.0046208, -0.0061864, -0.0043722, -0.0016019, 0.0011245
4: 0.0019514, 0.0027280, 0.0018457, 0.0026172, -0.0004782, 0.0006812
5: 0.0082101, 0.0132564, 0.0075232, 0.0125362, -0.0031072, 0.0044264
6: -0.0018238, -0.0005430, -0.0016410, -0.0003686, -0.0011235, 0.0007886
7: -0.0078563, -0.0045425, -0.0073834, -0.0040914, -0.0029068, 0.0020405
8: -0.0036957, -0.0019530, -0.0034470, -0.0017158, -0.0015287, 0.0010731
9: 0.0004008, 0.0024215, 0.0001257, 0.0021331, -0.0012443, 0.0017725

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012491, upper bound: 0.0009372
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011187, upper bound: 0.0008443
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.9892615, 0.9920586, 0.9888510, 0.9918745, -0.0017423, 0.0022668
1: -0.0039397, -0.0032427, -0.0040420, -0.0032886, -0.0004341, 0.0005648
2: 0.0071307, 0.0108243, 0.0073740, 0.0113665, -0.0029933, 0.0023007
3: -0.0061999, -0.0045187, -0.0064466, -0.0046294, -0.0010472, 0.0013624
4: 0.0019080, 0.0026229, 0.0019551, 0.0027278, -0.0005793, 0.0004453
5: 0.0079280, 0.0125736, 0.0082340, 0.0132555, -0.0037648, 0.0028937
6: -0.0016505, -0.0004714, -0.0018235, -0.0005490, -0.0007344, 0.0009555
7: -0.0074079, -0.0043572, -0.0078557, -0.0045582, -0.0019002, 0.0024723
8: -0.0034599, -0.0018556, -0.0036954, -0.0019612, -0.0009993, 0.0013002
9: 0.0002878, 0.0021481, 0.0004103, 0.0024211, -0.0015076, 0.0011587

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012292, upper bound: 0.0012882
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013833, upper bound: 0.0014298
time: 1.27 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.9892507, 0.9921016, 0.9888504, 0.9918888, -0.0017677, 0.0022887
1: -0.0039424, -0.0032320, -0.0040421, -0.0032850, -0.0004405, 0.0005703
2: 0.0070741, 0.0108385, 0.0073550, 0.0113672, -0.0030222, 0.0023342
3: -0.0062063, -0.0044929, -0.0064470, -0.0046208, -0.0010624, 0.0013756
4: 0.0018971, 0.0026257, 0.0019514, 0.0027280, -0.0005849, 0.0004518
5: 0.0078568, 0.0125914, 0.0082101, 0.0132564, -0.0038011, 0.0029358
6: -0.0016550, -0.0004533, -0.0018238, -0.0005430, -0.0007451, 0.0009648
7: -0.0074196, -0.0043105, -0.0078563, -0.0045425, -0.0019279, 0.0024961
8: -0.0034661, -0.0018310, -0.0036957, -0.0019530, -0.0010139, 0.0013127
9: 0.0002593, 0.0021552, 0.0004008, 0.0024215, -0.0015221, 0.0011756

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012286, upper bound: 0.0012901
time: 1.28 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013834, upper bound: 0.0014457
time: 1.28 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.9888603, 0.9918175, 0.9888510, 0.9918745, -0.0018467, 0.0018250
1: -0.0040397, -0.0033028, -0.0040420, -0.0032886, -0.0004601, 0.0004547
2: 0.0074493, 0.0113541, 0.0073740, 0.0113665, -0.0024099, 0.0024385
3: -0.0064410, -0.0046637, -0.0064466, -0.0046294, -0.0011099, 0.0010969
4: 0.0019697, 0.0027255, 0.0019551, 0.0027278, -0.0004664, 0.0004720
5: 0.0083287, 0.0132399, 0.0082340, 0.0132555, -0.0030310, 0.0030670
6: -0.0018196, -0.0005731, -0.0018235, -0.0005490, -0.0007784, 0.0007693
7: -0.0078455, -0.0046204, -0.0078557, -0.0045582, -0.0020141, 0.0019904
8: -0.0036900, -0.0019940, -0.0036954, -0.0019612, -0.0010592, 0.0010467
9: 0.0004482, 0.0024149, 0.0004103, 0.0024211, -0.0012138, 0.0012282

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013563, upper bound: 0.0012640
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013261, upper bound: 0.0012523
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: 0.9888533, 0.9918526, 0.9888504, 0.9918888, -0.0018862, 0.0018168
1: -0.0040414, -0.0032941, -0.0040421, -0.0032850, -0.0004700, 0.0004527
2: 0.0074029, 0.0113634, 0.0073550, 0.0113672, -0.0023990, 0.0024907
3: -0.0064452, -0.0046426, -0.0064470, -0.0046208, -0.0011337, 0.0010919
4: 0.0019607, 0.0027272, 0.0019514, 0.0027280, -0.0004643, 0.0004821
5: 0.0082704, 0.0132516, 0.0082101, 0.0132564, -0.0030173, 0.0031326
6: -0.0018226, -0.0005583, -0.0018238, -0.0005430, -0.0007951, 0.0007658
7: -0.0078531, -0.0045821, -0.0078563, -0.0045425, -0.0020572, 0.0019814
8: -0.0036940, -0.0019738, -0.0036957, -0.0019530, -0.0010818, 0.0010420
9: 0.0004249, 0.0024196, 0.0004008, 0.0024215, -0.0012083, 0.0012544

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013662, upper bound: 0.0012984
time: 1.40 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013397, upper bound: 0.0012898
time: 1.35 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: 0.9892615, 0.9920586, 0.9888822, 0.9920760, -0.0019736, 0.0022942
1: -0.0039397, -0.0032427, -0.0040342, -0.0032384, -0.0004918, 0.0005717
2: 0.0071307, 0.0108243, 0.0071079, 0.0113251, -0.0030295, 0.0026061
3: -0.0061999, -0.0045187, -0.0064278, -0.0045083, -0.0011862, 0.0013789
4: 0.0019080, 0.0026229, 0.0019036, 0.0027198, -0.0005864, 0.0005044
5: 0.0079280, 0.0125736, 0.0078993, 0.0132035, -0.0038103, 0.0032778
6: -0.0016505, -0.0004714, -0.0018103, -0.0004641, -0.0008319, 0.0009671
7: -0.0074079, -0.0043572, -0.0078215, -0.0043384, -0.0021525, 0.0025022
8: -0.0034599, -0.0018556, -0.0036774, -0.0018456, -0.0011320, 0.0013159
9: 0.0002878, 0.0021481, 0.0002763, 0.0024003, -0.0015258, 0.0013126

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012889, upper bound: 0.0012882
time: 1.35 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014301, upper bound: 0.0014290
time: 1.48 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: 0.9892507, 0.9921016, 0.9888818, 0.9920881, -0.0019960, 0.0023283
1: -0.0039424, -0.0032320, -0.0040343, -0.0032354, -0.0004974, 0.0005801
2: 0.0070741, 0.0108385, 0.0070919, 0.0113259, -0.0030745, 0.0026357
3: -0.0062063, -0.0044929, -0.0064282, -0.0045010, -0.0011997, 0.0013994
4: 0.0018971, 0.0026257, 0.0019005, 0.0027200, -0.0005951, 0.0005101
5: 0.0078568, 0.0125914, 0.0078791, 0.0132044, -0.0038669, 0.0033150
6: -0.0016550, -0.0004533, -0.0018106, -0.0004590, -0.0008414, 0.0009815
7: -0.0074196, -0.0043105, -0.0078222, -0.0043251, -0.0021769, 0.0025393
8: -0.0034661, -0.0018310, -0.0036777, -0.0018387, -0.0011448, 0.0013354
9: 0.0002593, 0.0021552, 0.0002682, 0.0024007, -0.0015485, 0.0013275

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012872, upper bound: 0.0012901
time: 1.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014304, upper bound: 0.0014452
time: 1.48 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9888510, 0.9918745, 0.9888886, 0.9920204, -0.0021794, 0.0019493
1: -0.0040420, -0.0032886, -0.0040326, -0.0032523, -0.0005431, 0.0004857
2: 0.0073740, 0.0113665, 0.0071812, 0.0113169, -0.0025740, 0.0028779
3: -0.0064466, -0.0046294, -0.0064241, -0.0045417, -0.0013099, 0.0011716
4: 0.0019551, 0.0027278, 0.0019178, 0.0027182, -0.0004982, 0.0005570
5: 0.0082340, 0.0132555, 0.0079916, 0.0131931, -0.0032374, 0.0036196
6: -0.0018235, -0.0005490, -0.0018077, -0.0004875, -0.0009187, 0.0008217
7: -0.0078557, -0.0045582, -0.0078147, -0.0043990, -0.0023770, 0.0021260
8: -0.0036954, -0.0019612, -0.0036738, -0.0018775, -0.0012500, 0.0011180
9: 0.0004103, 0.0024211, 0.0003132, 0.0023962, -0.0012964, 0.0014495

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013671, upper bound: 0.0013107
time: 1.37 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013603, upper bound: 0.0012775
time: 1.37 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888504, 0.9918888, 0.9888846, 0.9920495, -0.0021757, 0.0019764
1: -0.0040421, -0.0032850, -0.0040336, -0.0032450, -0.0005421, 0.0004925
2: 0.0073550, 0.0113672, 0.0071429, 0.0113221, -0.0026099, 0.0028730
3: -0.0064470, -0.0046208, -0.0064264, -0.0045243, -0.0013077, 0.0011879
4: 0.0019514, 0.0027280, 0.0019104, 0.0027193, -0.0005051, 0.0005561
5: 0.0082101, 0.0132564, 0.0079434, 0.0131997, -0.0032825, 0.0036134
6: -0.0018238, -0.0005430, -0.0018094, -0.0004753, -0.0009171, 0.0008331
7: -0.0078563, -0.0045425, -0.0078190, -0.0043673, -0.0023729, 0.0021556
8: -0.0036957, -0.0019530, -0.0036761, -0.0018609, -0.0012479, 0.0011336
9: 0.0004008, 0.0024215, 0.0002939, 0.0023988, -0.0013145, 0.0014470

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013983, upper bound: 0.0013194
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013936, upper bound: 0.0012898
time: 1.37 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9890327, 0.9922054, 0.9892615, 0.9920586, -0.0020190, 0.0021398
1: -0.0039967, -0.0032062, -0.0039397, -0.0032427, -0.0005031, 0.0005332
2: 0.0069369, 0.0111264, 0.0071307, 0.0108243, -0.0028256, 0.0026661
3: -0.0063374, -0.0044305, -0.0061999, -0.0045187, -0.0012135, 0.0012861
4: 0.0018705, 0.0026814, 0.0019080, 0.0026229, -0.0005469, 0.0005160
5: 0.0076843, 0.0129536, 0.0079280, 0.0125736, -0.0035538, 0.0033532
6: -0.0017469, -0.0004095, -0.0016505, -0.0004714, -0.0008511, 0.0009020
7: -0.0076575, -0.0041972, -0.0074079, -0.0043572, -0.0022020, 0.0023337
8: -0.0035911, -0.0017714, -0.0034599, -0.0018556, -0.0011580, 0.0012273
9: 0.0001902, 0.0023002, 0.0002878, 0.0021481, -0.0014231, 0.0013428

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 56

### Candidate
type: A, layer: 1, pos: 56

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A2_B1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012936, upper bound: 0.0010548
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012934, upper bound: 0.0010600
time: 1.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9890256, 0.9922410, 0.9892615, 0.9920586, -0.0020231, 0.0021572
1: -0.0039985, -0.0031973, -0.0039397, -0.0032427, -0.0005041, 0.0005375
2: 0.0068899, 0.0111359, 0.0071307, 0.0108243, -0.0028486, 0.0026714
3: -0.0063417, -0.0044091, -0.0061999, -0.0045187, -0.0012159, 0.0012966
4: 0.0018614, 0.0026832, 0.0019080, 0.0026229, -0.0005513, 0.0005171
5: 0.0076251, 0.0129655, 0.0079280, 0.0125736, -0.0035828, 0.0033600
6: -0.0017499, -0.0003945, -0.0016505, -0.0004714, -0.0008528, 0.0009093
7: -0.0076653, -0.0041583, -0.0074079, -0.0043572, -0.0022064, 0.0023528
8: -0.0035952, -0.0017510, -0.0034599, -0.0018556, -0.0011603, 0.0012373
9: 0.0001665, 0.0023050, 0.0002878, 0.0021481, -0.0014347, 0.0013455

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 56

### Candidate
type: A, layer: 1, pos: 56

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A2_B1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012936, upper bound: 0.0010626
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012934, upper bound: 0.0010692
time: 1.30 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9890327, 0.9922054, 0.9892507, 0.9921016, -0.0020856, 0.0021532
1: -0.0039967, -0.0032062, -0.0039424, -0.0032320, -0.0005197, 0.0005365
2: 0.0069369, 0.0111264, 0.0070741, 0.0108385, -0.0028433, 0.0027540
3: -0.0063374, -0.0044305, -0.0062063, -0.0044929, -0.0012535, 0.0012942
4: 0.0018705, 0.0026814, 0.0018971, 0.0026257, -0.0005503, 0.0005330
5: 0.0076843, 0.0129536, 0.0078568, 0.0125914, -0.0035762, 0.0034639
6: -0.0017469, -0.0004095, -0.0016550, -0.0004533, -0.0008792, 0.0009077
7: -0.0076575, -0.0041972, -0.0074196, -0.0043105, -0.0022747, 0.0023484
8: -0.0035911, -0.0017714, -0.0034661, -0.0018310, -0.0011962, 0.0012350
9: 0.0001902, 0.0023002, 0.0002593, 0.0021552, -0.0014321, 0.0013871

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 56

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012936, upper bound: 0.0010492
time: 1.30 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012934, upper bound: 0.0010552
time: 1.27 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9890256, 0.9922410, 0.9892507, 0.9921016, -0.0020404, 0.0021654
1: -0.0039985, -0.0031973, -0.0039424, -0.0032320, -0.0005084, 0.0005396
2: 0.0068899, 0.0111359, 0.0070741, 0.0108385, -0.0028593, 0.0026943
3: -0.0063417, -0.0044091, -0.0062063, -0.0044929, -0.0012263, 0.0013015
4: 0.0018614, 0.0026832, 0.0018971, 0.0026257, -0.0005534, 0.0005215
5: 0.0076251, 0.0129655, 0.0078568, 0.0125914, -0.0035963, 0.0033887
6: -0.0017499, -0.0003945, -0.0016550, -0.0004533, -0.0008601, 0.0009128
7: -0.0076653, -0.0041583, -0.0074196, -0.0043105, -0.0022253, 0.0023616
8: -0.0035952, -0.0017510, -0.0034661, -0.0018310, -0.0011703, 0.0012420
9: 0.0001665, 0.0023050, 0.0002593, 0.0021552, -0.0014401, 0.0013570

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 56

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012936, upper bound: 0.0010626
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012934, upper bound: 0.0010692
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9890327, 0.9922054, 0.9892911, 0.9922644, -0.0023929, 0.0022563
1: -0.0039967, -0.0032062, -0.0039323, -0.0031914, -0.0005962, 0.0005622
2: 0.0069369, 0.0111264, 0.0068589, 0.0107853, -0.0029794, 0.0031598
3: -0.0063374, -0.0044305, -0.0061821, -0.0043950, -0.0014382, 0.0013561
4: 0.0018705, 0.0026814, 0.0018554, 0.0026154, -0.0005766, 0.0006116
5: 0.0076843, 0.0129536, 0.0075861, 0.0125245, -0.0037473, 0.0039742
6: -0.0017469, -0.0004095, -0.0016380, -0.0003846, -0.0010087, 0.0009511
7: -0.0076575, -0.0041972, -0.0073757, -0.0041327, -0.0026098, 0.0024608
8: -0.0035911, -0.0017714, -0.0034430, -0.0017375, -0.0013725, 0.0012941
9: 0.0001902, 0.0023002, 0.0001509, 0.0021284, -0.0015006, 0.0015914

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 56

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013457, upper bound: 0.0010548
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013464, upper bound: 0.0010600
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9890256, 0.9922410, 0.9892911, 0.9922644, -0.0023970, 0.0022737
1: -0.0039985, -0.0031973, -0.0039323, -0.0031914, -0.0005973, 0.0005665
2: 0.0068899, 0.0111359, 0.0068589, 0.0107853, -0.0030024, 0.0031652
3: -0.0063417, -0.0044091, -0.0061821, -0.0043950, -0.0014406, 0.0013666
4: 0.0018614, 0.0026832, 0.0018554, 0.0026154, -0.0005811, 0.0006126
5: 0.0076251, 0.0129655, 0.0075861, 0.0125245, -0.0037762, 0.0039809
6: -0.0017499, -0.0003945, -0.0016380, -0.0003846, -0.0010104, 0.0009584
7: -0.0076653, -0.0041583, -0.0073757, -0.0041327, -0.0026142, 0.0024798
8: -0.0035952, -0.0017510, -0.0034430, -0.0017375, -0.0013748, 0.0013041
9: 0.0001665, 0.0023050, 0.0001509, 0.0021284, -0.0015122, 0.0015941

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 56

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A2_B1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013457, upper bound: 0.0010626
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013464, upper bound: 0.0010692
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9890327, 0.9922054, 0.9892840, 0.9923024, -0.0024210, 0.0022565
1: -0.0039967, -0.0032062, -0.0039341, -0.0031820, -0.0006032, 0.0005623
2: 0.0069369, 0.0111264, 0.0068089, 0.0107946, -0.0029797, 0.0031969
3: -0.0063374, -0.0044305, -0.0061864, -0.0043722, -0.0014551, 0.0013562
4: 0.0018705, 0.0026814, 0.0018457, 0.0026172, -0.0005767, 0.0006188
5: 0.0076843, 0.0129536, 0.0075232, 0.0125362, -0.0037476, 0.0040209
6: -0.0017469, -0.0004095, -0.0016410, -0.0003686, -0.0010205, 0.0009512
7: -0.0076575, -0.0041972, -0.0073834, -0.0040914, -0.0026404, 0.0024610
8: -0.0035911, -0.0017714, -0.0034470, -0.0017158, -0.0013886, 0.0012942
9: 0.0001902, 0.0023002, 0.0001257, 0.0021331, -0.0015007, 0.0016101

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 56

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A2_B1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013457, upper bound: 0.0010492
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013464, upper bound: 0.0010552
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9890256, 0.9922410, 0.9892840, 0.9923024, -0.0024169, 0.0022835
1: -0.0039985, -0.0031973, -0.0039341, -0.0031820, -0.0006022, 0.0005690
2: 0.0068899, 0.0111359, 0.0068089, 0.0107946, -0.0030154, 0.0031915
3: -0.0063417, -0.0044091, -0.0061864, -0.0043722, -0.0014526, 0.0013725
4: 0.0018614, 0.0026832, 0.0018457, 0.0026172, -0.0005836, 0.0006177
5: 0.0076251, 0.0129655, 0.0075232, 0.0125362, -0.0037926, 0.0040141
6: -0.0017499, -0.0003945, -0.0016410, -0.0003686, -0.0010188, 0.0009626
7: -0.0076653, -0.0041583, -0.0073834, -0.0040914, -0.0026360, 0.0024905
8: -0.0035952, -0.0017510, -0.0034470, -0.0017158, -0.0013862, 0.0013097
9: 0.0001665, 0.0023050, 0.0001257, 0.0021331, -0.0015187, 0.0016074

Time for backsubstitution: 1.45 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.41 + 596.83 = 600.24 seconds
