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
execution time: IAR + RelationalAnalysis = 1.47 + 2.03 = 3.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0018048, upper bound: 0.0018049

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017293, upper bound: 0.0016839
time: 1.19 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017299, upper bound: 0.0017299
time: 1.20 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.52 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.52
Output dim: 0, lower bound: -0.0017293, upper bound: 0.0016839
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.52
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

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015840, upper bound: 0.0015082
time: 1.17 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015818, upper bound: 0.0015214
time: 1.14 seconds

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

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016839, upper bound: 0.0017293
time: 1.21 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016839, upper bound: 0.0017299
time: 1.23 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.96 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 0, lower bound: -0.0015840, upper bound: 0.0015082
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 0, lower bound: -0.0015818, upper bound: 0.0015214
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 0, lower bound: -0.0016839, upper bound: 0.0017293
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.96
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

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015688, upper bound: 0.0015068
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015688, upper bound: 0.0015068
time: 1.48 seconds

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

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015688, upper bound: 0.0015214
time: 1.20 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015688, upper bound: 0.0015214
time: 1.20 seconds

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

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

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
time: 1.35 seconds

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

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015082, upper bound: 0.0015988
time: 1.66 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015214, upper bound: 0.0015988
time: 1.26 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.43 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 0, lower bound: -0.0015688, upper bound: 0.0015068
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 0, lower bound: -0.0015688, upper bound: 0.0015068
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 0, lower bound: -0.0015688, upper bound: 0.0015214
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 0, lower bound: -0.0015688, upper bound: 0.0015214
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 0, lower bound: -0.0015082, upper bound: 0.0015840
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 0, lower bound: -0.0015214, upper bound: 0.0015818
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 0, lower bound: -0.0015082, upper bound: 0.0015988
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.43
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

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015291, upper bound: 0.0015082
time: 1.38 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015291, upper bound: 0.0015082
time: 1.70 seconds

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

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015291, upper bound: 0.0015082
time: 1.27 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015291, upper bound: 0.0015082
time: 1.48 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9888023, 0.9919065, 0.9889213, 0.9924843, -0.0030184, 0.0023646
1: -0.0040541, -0.0032806, -0.0040245, -0.0031367, -0.0007521, 0.0005892
2: 0.0073317, 0.0114307, 0.0065688, 0.0112736, -0.0031224, 0.0039858
3: -0.0064759, -0.0046102, -0.0064044, -0.0042629, -0.0018142, 0.0014212
4: 0.0019469, 0.0027403, 0.0017993, 0.0027099, -0.0006043, 0.0007714
5: 0.0081808, 0.0133363, 0.0072212, 0.0131387, -0.0039271, 0.0050131
6: -0.0018441, -0.0005355, -0.0017939, -0.0002920, -0.0012724, 0.0009967
7: -0.0079088, -0.0045232, -0.0077790, -0.0038931, -0.0032920, 0.0025789
8: -0.0037233, -0.0019429, -0.0036551, -0.0016115, -0.0017313, 0.0013562
9: 0.0003890, 0.0024535, 0.0000047, 0.0023744, -0.0015726, 0.0020075

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015214
time: 1.18 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015214
time: 1.41 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9888738, 0.9923024, 0.9889213, 0.9924843, -0.0027236, 0.0025839
1: -0.0040363, -0.0031820, -0.0040245, -0.0031367, -0.0006787, 0.0006438
2: 0.0068090, 0.0113364, 0.0065688, 0.0112736, -0.0034120, 0.0035965
3: -0.0064329, -0.0043723, -0.0064044, -0.0042629, -0.0016370, 0.0015530
4: 0.0018457, 0.0027220, 0.0017993, 0.0027099, -0.0006604, 0.0006961
5: 0.0075233, 0.0132176, 0.0072212, 0.0131387, -0.0042914, 0.0045235
6: -0.0018139, -0.0003687, -0.0017939, -0.0002920, -0.0011481, 0.0010892
7: -0.0078308, -0.0040915, -0.0077790, -0.0038931, -0.0029705, 0.0028181
8: -0.0036823, -0.0017158, -0.0036551, -0.0016115, -0.0015622, 0.0014820
9: 0.0001257, 0.0024060, 0.0000047, 0.0023744, -0.0017185, 0.0018114

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015068
time: 1.53 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015068
time: 1.56 seconds

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

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015688
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015818
time: 1.22 seconds

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

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015214, upper bound: 0.0015688
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015214, upper bound: 0.0015818
time: 1.15 seconds

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

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015315, upper bound: 0.0015788
time: 1.27 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015315, upper bound: 0.0015988
time: 1.24 seconds

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

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013376, upper bound: 0.0011768
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014852, upper bound: 0.0015255
time: 1.17 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.69 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0015291, upper bound: 0.0015082
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0015291, upper bound: 0.0015082
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0015291, upper bound: 0.0015082
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0015291, upper bound: 0.0015082
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015214
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015214
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015068
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015068
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015688
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0015068, upper bound: 0.0015818
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0015214, upper bound: 0.0015688
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0015214, upper bound: 0.0015818
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0015315, upper bound: 0.0015788
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0015315, upper bound: 0.0015988
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0013376, upper bound: 0.0011768
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 0, lower bound: -0.0014852, upper bound: 0.0015255

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9888023, 0.9919065, 0.9888023, 0.9919065, -0.0024064, 0.0024064
1: -0.0040541, -0.0032806, -0.0040541, -0.0032806, -0.0005996, 0.0005996
2: 0.0073317, 0.0114307, 0.0073317, 0.0114307, -0.0031777, 0.0031777
3: -0.0064759, -0.0046102, -0.0064759, -0.0046102, -0.0014463, 0.0014463
4: 0.0019469, 0.0027403, 0.0019469, 0.0027403, -0.0006150, 0.0006150
5: 0.0081808, 0.0133363, 0.0081808, 0.0133363, -0.0039967, 0.0039967
6: -0.0018441, -0.0005355, -0.0018441, -0.0005355, -0.0010144, 0.0010144
7: -0.0079088, -0.0045232, -0.0079088, -0.0045232, -0.0026246, 0.0026246
8: -0.0037233, -0.0019429, -0.0037233, -0.0019429, -0.0013802, 0.0013802
9: 0.0003890, 0.0024535, 0.0003890, 0.0024535, -0.0016005, 0.0016005

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014912
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015003, upper bound: 0.0014996
time: 1.27 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9888023, 0.9919065, 0.9888360, 0.9921069, -0.0026350, 0.0024470
1: -0.0040541, -0.0032806, -0.0040457, -0.0032307, -0.0006566, 0.0006097
2: 0.0073317, 0.0114307, 0.0070670, 0.0113862, -0.0032313, 0.0034795
3: -0.0064759, -0.0046102, -0.0064556, -0.0044897, -0.0015837, 0.0014707
4: 0.0019469, 0.0027403, 0.0018957, 0.0027317, -0.0006254, 0.0006734
5: 0.0081808, 0.0133363, 0.0078479, 0.0132802, -0.0040641, 0.0043762
6: -0.0018441, -0.0005355, -0.0018298, -0.0004511, -0.0011107, 0.0010315
7: -0.0079088, -0.0045232, -0.0078720, -0.0043047, -0.0028738, 0.0026688
8: -0.0037233, -0.0019429, -0.0037039, -0.0018279, -0.0015113, 0.0014035
9: 0.0003890, 0.0024535, 0.0002557, 0.0024310, -0.0016274, 0.0017524

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014912
time: 1.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015003, upper bound: 0.0014996
time: 1.73 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9888738, 0.9923024, 0.9888023, 0.9919065, -0.0023969, 0.0028552
1: -0.0040363, -0.0031820, -0.0040541, -0.0032806, -0.0005972, 0.0007115
2: 0.0068090, 0.0113364, 0.0073317, 0.0114307, -0.0037703, 0.0031651
3: -0.0064329, -0.0043723, -0.0064759, -0.0046102, -0.0014406, 0.0017161
4: 0.0018457, 0.0027220, 0.0019469, 0.0027403, -0.0007297, 0.0006126
5: 0.0075233, 0.0132176, 0.0081808, 0.0133363, -0.0047421, 0.0039809
6: -0.0018139, -0.0003687, -0.0018441, -0.0005355, -0.0010104, 0.0012036
7: -0.0078308, -0.0040915, -0.0079088, -0.0045232, -0.0026142, 0.0031141
8: -0.0036823, -0.0017158, -0.0037233, -0.0019429, -0.0013748, 0.0016376
9: 0.0001257, 0.0024060, 0.0003890, 0.0024535, -0.0018989, 0.0015941

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015028, upper bound: 0.0014660
time: 1.24 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015030, upper bound: 0.0014812
time: 1.29 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888738, 0.9923024, 0.9888360, 0.9921069, -0.0026254, 0.0028958
1: -0.0040363, -0.0031820, -0.0040457, -0.0032307, -0.0006542, 0.0007216
2: 0.0068090, 0.0113364, 0.0070670, 0.0113862, -0.0038239, 0.0034669
3: -0.0064329, -0.0043723, -0.0064556, -0.0044897, -0.0015780, 0.0017405
4: 0.0018457, 0.0027220, 0.0018957, 0.0027317, -0.0007401, 0.0006710
5: 0.0075233, 0.0132176, 0.0078479, 0.0132802, -0.0048095, 0.0043604
6: -0.0018139, -0.0003687, -0.0018298, -0.0004511, -0.0011067, 0.0012207
7: -0.0078308, -0.0040915, -0.0078720, -0.0043047, -0.0028634, 0.0031583
8: -0.0036823, -0.0017158, -0.0037039, -0.0018279, -0.0015058, 0.0016609
9: 0.0001257, 0.0024060, 0.0002557, 0.0024310, -0.0019259, 0.0017461

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015028, upper bound: 0.0014660
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015030, upper bound: 0.0014812
time: 1.28 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9888023, 0.9919065, 0.9888738, 0.9923024, -0.0028552, 0.0023969
1: -0.0040541, -0.0032806, -0.0040363, -0.0031820, -0.0007115, 0.0005972
2: 0.0073317, 0.0114307, 0.0068090, 0.0113364, -0.0031651, 0.0037703
3: -0.0064759, -0.0046102, -0.0064329, -0.0043723, -0.0017161, 0.0014406
4: 0.0019469, 0.0027403, 0.0018457, 0.0027220, -0.0006126, 0.0007297
5: 0.0081808, 0.0133363, 0.0075233, 0.0132176, -0.0039809, 0.0047421
6: -0.0018441, -0.0005355, -0.0018139, -0.0003687, -0.0012036, 0.0010104
7: -0.0079088, -0.0045232, -0.0078308, -0.0040915, -0.0031141, 0.0026142
8: -0.0037233, -0.0019429, -0.0036823, -0.0017158, -0.0016376, 0.0013748
9: 0.0003890, 0.0024535, 0.0001257, 0.0024060, -0.0015941, 0.0018989

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014811, upper bound: 0.0014876
time: 1.20 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014812, upper bound: 0.0015030
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9888023, 0.9919065, 0.9889241, 0.9924735, -0.0030231, 0.0023617
1: -0.0040541, -0.0032806, -0.0040238, -0.0031393, -0.0007533, 0.0005885
2: 0.0073317, 0.0114307, 0.0065828, 0.0112698, -0.0031186, 0.0039920
3: -0.0064759, -0.0046102, -0.0064027, -0.0042693, -0.0018170, 0.0014194
4: 0.0019469, 0.0027403, 0.0018020, 0.0027091, -0.0006036, 0.0007727
5: 0.0081808, 0.0133363, 0.0072389, 0.0131340, -0.0039224, 0.0050209
6: -0.0018441, -0.0005355, -0.0017927, -0.0002965, -0.0012744, 0.0009955
7: -0.0079088, -0.0045232, -0.0077759, -0.0039047, -0.0032972, 0.0025758
8: -0.0037233, -0.0019429, -0.0036534, -0.0016176, -0.0017340, 0.0013546
9: 0.0003890, 0.0024535, 0.0000118, 0.0023725, -0.0015707, 0.0020106

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014811, upper bound: 0.0014876
time: 1.21 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014812, upper bound: 0.0015030
time: 1.63 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9888738, 0.9923024, 0.9888738, 0.9923024, -0.0025359, 0.0025359
1: -0.0040363, -0.0031820, -0.0040363, -0.0031820, -0.0006319, 0.0006319
2: 0.0068090, 0.0113364, 0.0068090, 0.0113364, -0.0033486, 0.0033486
3: -0.0064329, -0.0043723, -0.0064329, -0.0043723, -0.0015241, 0.0015241
4: 0.0018457, 0.0027220, 0.0018457, 0.0027220, -0.0006481, 0.0006481
5: 0.0075233, 0.0132176, 0.0075233, 0.0132176, -0.0042117, 0.0042117
6: -0.0018139, -0.0003687, -0.0018139, -0.0003687, -0.0010690, 0.0010690
7: -0.0078308, -0.0040915, -0.0078308, -0.0040915, -0.0027657, 0.0027657
8: -0.0036823, -0.0017158, -0.0036823, -0.0017158, -0.0014545, 0.0014545
9: 0.0001257, 0.0024060, 0.0001257, 0.0024060, -0.0016865, 0.0016865

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014944, upper bound: 0.0014651
time: 1.24 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014946, upper bound: 0.0014797
time: 1.29 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888738, 0.9923024, 0.9889241, 0.9924735, -0.0027657, 0.0025805
1: -0.0040363, -0.0031820, -0.0040238, -0.0031393, -0.0006891, 0.0006430
2: 0.0068090, 0.0113364, 0.0065828, 0.0112698, -0.0034075, 0.0036521
3: -0.0064329, -0.0043723, -0.0064027, -0.0042693, -0.0016623, 0.0015509
4: 0.0018457, 0.0027220, 0.0018020, 0.0027091, -0.0006595, 0.0007068
5: 0.0075233, 0.0132176, 0.0072389, 0.0131340, -0.0042857, 0.0045933
6: -0.0018139, -0.0003687, -0.0017927, -0.0002965, -0.0011658, 0.0010878
7: -0.0078308, -0.0040915, -0.0077759, -0.0039047, -0.0030164, 0.0028144
8: -0.0036823, -0.0017158, -0.0036534, -0.0016176, -0.0015863, 0.0014800
9: 0.0001257, 0.0024060, 0.0000118, 0.0023725, -0.0017162, 0.0018394

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014944, upper bound: 0.0014651
time: 1.20 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014946, upper bound: 0.0014797
time: 1.30 seconds

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

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011018, upper bound: 0.0014001
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014321, upper bound: 0.0015051
time: 1.15 seconds

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

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011018, upper bound: 0.0014001
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014321, upper bound: 0.0015111
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9889241, 0.9924735, 0.9888023, 0.9919065, -0.0023617, 0.0030232
1: -0.0040238, -0.0031393, -0.0040541, -0.0032806, -0.0005885, 0.0007533
2: 0.0065828, 0.0112698, 0.0073317, 0.0114307, -0.0039920, 0.0031186
3: -0.0064027, -0.0042693, -0.0064759, -0.0046102, -0.0014194, 0.0018170
4: 0.0018020, 0.0027091, 0.0019469, 0.0027403, -0.0007727, 0.0006036
5: 0.0072389, 0.0131340, 0.0081808, 0.0133363, -0.0050209, 0.0039224
6: -0.0017927, -0.0002965, -0.0018441, -0.0005355, -0.0009955, 0.0012744
7: -0.0077759, -0.0039047, -0.0079088, -0.0045232, -0.0025758, 0.0032972
8: -0.0036534, -0.0016176, -0.0037233, -0.0019429, -0.0013546, 0.0017340
9: 0.0000118, 0.0023725, 0.0003890, 0.0024535, -0.0020106, 0.0015707

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014945, upper bound: 0.0015273
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014946, upper bound: 0.0015400
time: 1.42 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9889241, 0.9924735, 0.9888738, 0.9923024, -0.0025805, 0.0027657
1: -0.0040238, -0.0031393, -0.0040363, -0.0031820, -0.0006430, 0.0006891
2: 0.0065828, 0.0112698, 0.0068090, 0.0113364, -0.0036521, 0.0034075
3: -0.0064027, -0.0042693, -0.0064329, -0.0043723, -0.0015509, 0.0016623
4: 0.0018020, 0.0027091, 0.0018457, 0.0027220, -0.0007068, 0.0006595
5: 0.0072389, 0.0131340, 0.0075233, 0.0132176, -0.0045933, 0.0042857
6: -0.0017927, -0.0002965, -0.0018139, -0.0003687, -0.0010878, 0.0011658
7: -0.0077759, -0.0039047, -0.0078308, -0.0040915, -0.0028144, 0.0030164
8: -0.0036534, -0.0016176, -0.0036823, -0.0017158, -0.0014800, 0.0015863
9: 0.0000118, 0.0023725, 0.0001257, 0.0024060, -0.0018394, 0.0017162

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014945, upper bound: 0.0015273
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014946, upper bound: 0.0015399
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

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012036, upper bound: 0.0014415
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014525, upper bound: 0.0015106
time: 1.28 seconds

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

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012036, upper bound: 0.0014415
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014525, upper bound: 0.0015255
time: 1.25 seconds

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
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0011494
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0011768
time: 0.91 seconds

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

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011406, upper bound: 0.0013728
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011406, upper bound: 0.0015255
time: 1.27 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.89 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014912
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0015003, upper bound: 0.0014996
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014912
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0015003, upper bound: 0.0014996
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0015028, upper bound: 0.0014660
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0015030, upper bound: 0.0014812
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0015028, upper bound: 0.0014660
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0015030, upper bound: 0.0014812
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014811, upper bound: 0.0014876
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014812, upper bound: 0.0015030
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014811, upper bound: 0.0014876
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014812, upper bound: 0.0015030
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014944, upper bound: 0.0014651
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014946, upper bound: 0.0014797
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014944, upper bound: 0.0014651
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014946, upper bound: 0.0014797
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0011018, upper bound: 0.0014001
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014321, upper bound: 0.0015051
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0011018, upper bound: 0.0014001
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014321, upper bound: 0.0015111
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014945, upper bound: 0.0015273
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014946, upper bound: 0.0015400
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014945, upper bound: 0.0015273
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014946, upper bound: 0.0015399
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0012036, upper bound: 0.0014415
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014525, upper bound: 0.0015106
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0012036, upper bound: 0.0014415
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0014525, upper bound: 0.0015255
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0011494
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0011768
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0011406, upper bound: 0.0013728
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.0011406, upper bound: 0.0015255

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9892468, 0.9921501, 0.9889489, 0.9918969, -0.0018235, 0.0021499
1: -0.0039434, -0.0032200, -0.0040176, -0.0032830, -0.0004544, 0.0005357
2: 0.0070101, 0.0108438, 0.0073442, 0.0112372, -0.0028390, 0.0024080
3: -0.0062088, -0.0044638, -0.0063878, -0.0046159, -0.0010960, 0.0012922
4: 0.0018847, 0.0026267, 0.0019494, 0.0027028, -0.0005495, 0.0004661
5: 0.0077763, 0.0125981, 0.0081966, 0.0130929, -0.0035707, 0.0030286
6: -0.0016567, -0.0004329, -0.0017823, -0.0005395, -0.0007687, 0.0009063
7: -0.0074240, -0.0042576, -0.0077489, -0.0045336, -0.0019888, 0.0023448
8: -0.0034684, -0.0018032, -0.0036392, -0.0019483, -0.0010459, 0.0012331
9: 0.0002270, 0.0021579, 0.0003953, 0.0023560, -0.0014298, 0.0012128

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014275
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014916
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888493, 0.9919039, 0.9888023, 0.9919065, -0.0019362, 0.0023912
1: -0.0040424, -0.0032813, -0.0040541, -0.0032806, -0.0004825, 0.0005958
2: 0.0073351, 0.0113688, 0.0073317, 0.0114307, -0.0031576, 0.0025568
3: -0.0064477, -0.0046117, -0.0064759, -0.0046102, -0.0011637, 0.0014372
4: 0.0019476, 0.0027283, 0.0019469, 0.0027403, -0.0006111, 0.0004949
5: 0.0081850, 0.0132584, 0.0081808, 0.0133363, -0.0039714, 0.0032157
6: -0.0018243, -0.0005366, -0.0018441, -0.0005355, -0.0008162, 0.0010080
7: -0.0078576, -0.0045260, -0.0079088, -0.0045232, -0.0021117, 0.0026080
8: -0.0036964, -0.0019443, -0.0037233, -0.0019429, -0.0011105, 0.0013715
9: 0.0003907, 0.0024223, 0.0003890, 0.0024535, -0.0015903, 0.0012877

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014916, upper bound: 0.0014275
time: 1.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014916, upper bound: 0.0015003
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9892468, 0.9921501, 0.9889867, 0.9920977, -0.0020513, 0.0022136
1: -0.0039434, -0.0032200, -0.0040082, -0.0032330, -0.0005111, 0.0005516
2: 0.0070101, 0.0108438, 0.0070791, 0.0111873, -0.0029231, 0.0027087
3: -0.0062088, -0.0044638, -0.0063651, -0.0044952, -0.0012329, 0.0013305
4: 0.0018847, 0.0026267, 0.0018980, 0.0026932, -0.0005658, 0.0005243
5: 0.0077763, 0.0125981, 0.0078632, 0.0130301, -0.0036765, 0.0034069
6: -0.0016567, -0.0004329, -0.0017663, -0.0004549, -0.0008647, 0.0009331
7: -0.0074240, -0.0042576, -0.0077077, -0.0043146, -0.0022373, 0.0024143
8: -0.0034684, -0.0018032, -0.0036175, -0.0018332, -0.0011766, 0.0012697
9: 0.0002270, 0.0021579, 0.0002618, 0.0023309, -0.0014722, 0.0013643

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014737, upper bound: 0.0014275
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014737, upper bound: 0.0014912
time: 1.22 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9888493, 0.9919039, 0.9888360, 0.9921069, -0.0022795, 0.0024318
1: -0.0040424, -0.0032813, -0.0040457, -0.0032307, -0.0005680, 0.0006059
2: 0.0073351, 0.0113688, 0.0070670, 0.0113862, -0.0032112, 0.0030100
3: -0.0064477, -0.0046117, -0.0064556, -0.0044897, -0.0013700, 0.0014616
4: 0.0019476, 0.0027283, 0.0018957, 0.0027317, -0.0006215, 0.0005826
5: 0.0081850, 0.0132584, 0.0078479, 0.0132802, -0.0040388, 0.0037858
6: -0.0018243, -0.0005366, -0.0018298, -0.0004511, -0.0009609, 0.0010251
7: -0.0078576, -0.0045260, -0.0078720, -0.0043047, -0.0024861, 0.0026523
8: -0.0036964, -0.0019443, -0.0037039, -0.0018279, -0.0013074, 0.0013948
9: 0.0003907, 0.0024223, 0.0002557, 0.0024310, -0.0016173, 0.0015160

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015379, upper bound: 0.0014275
time: 1.27 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015379, upper bound: 0.0014995
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9888825, 0.9922152, 0.9888041, 0.9918770, -0.0023416, 0.0027620
1: -0.0040341, -0.0032037, -0.0040537, -0.0032880, -0.0005835, 0.0006882
2: 0.0069241, 0.0113249, 0.0073706, 0.0114284, -0.0036472, 0.0030920
3: -0.0064277, -0.0044247, -0.0064748, -0.0046279, -0.0014074, 0.0016601
4: 0.0018680, 0.0027198, 0.0019545, 0.0027398, -0.0007059, 0.0005985
5: 0.0076681, 0.0132032, 0.0082298, 0.0133334, -0.0045873, 0.0038890
6: -0.0018103, -0.0004054, -0.0018433, -0.0005480, -0.0009871, 0.0011643
7: -0.0078214, -0.0041866, -0.0079069, -0.0045554, -0.0025538, 0.0030124
8: -0.0036773, -0.0017658, -0.0037223, -0.0019598, -0.0013430, 0.0015842
9: 0.0001837, 0.0024002, 0.0004086, 0.0024523, -0.0018369, 0.0015573

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014876, upper bound: 0.0014660
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014876, upper bound: 0.0014660
time: 1.40 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888770, 0.9922507, 0.9888034, 0.9918913, -0.0023525, 0.0027847
1: -0.0040355, -0.0031949, -0.0040538, -0.0032844, -0.0005862, 0.0006939
2: 0.0068772, 0.0113321, 0.0073517, 0.0114292, -0.0036772, 0.0031065
3: -0.0064310, -0.0044033, -0.0064752, -0.0046193, -0.0014139, 0.0016737
4: 0.0018590, 0.0027212, 0.0019508, 0.0027400, -0.0007117, 0.0006012
5: 0.0076092, 0.0132123, 0.0082059, 0.0133343, -0.0046250, 0.0039071
6: -0.0018126, -0.0003905, -0.0018436, -0.0005419, -0.0009917, 0.0011739
7: -0.0078273, -0.0041478, -0.0079075, -0.0045397, -0.0025657, 0.0030372
8: -0.0036805, -0.0017455, -0.0037226, -0.0019516, -0.0013493, 0.0015972
9: 0.0001601, 0.0024038, 0.0003991, 0.0024527, -0.0018520, 0.0015646

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010587, upper bound: 0.0010650
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009877, upper bound: 0.0009446
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9888825, 0.9922152, 0.9888378, 0.9920785, -0.0025718, 0.0028013
1: -0.0040341, -0.0032037, -0.0040453, -0.0032378, -0.0006408, 0.0006980
2: 0.0069241, 0.0113249, 0.0071046, 0.0113839, -0.0036991, 0.0033961
3: -0.0064277, -0.0044247, -0.0064546, -0.0045068, -0.0015457, 0.0016837
4: 0.0018680, 0.0027198, 0.0019030, 0.0027312, -0.0007160, 0.0006573
5: 0.0076681, 0.0132032, 0.0078952, 0.0132774, -0.0046526, 0.0042714
6: -0.0018103, -0.0004054, -0.0018291, -0.0004630, -0.0010841, 0.0011809
7: -0.0078214, -0.0041866, -0.0078701, -0.0043357, -0.0028050, 0.0030553
8: -0.0036773, -0.0017658, -0.0037029, -0.0018442, -0.0014751, 0.0016067
9: 0.0001837, 0.0024002, 0.0002746, 0.0024299, -0.0018631, 0.0017104

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015388, upper bound: 0.0014660
time: 1.42 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015388, upper bound: 0.0014660
time: 1.26 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9888770, 0.9922507, 0.9888372, 0.9920906, -0.0025798, 0.0028406
1: -0.0040355, -0.0031949, -0.0040454, -0.0032348, -0.0006428, 0.0007078
2: 0.0068772, 0.0113321, 0.0070886, 0.0113846, -0.0037510, 0.0034066
3: -0.0064310, -0.0044033, -0.0064549, -0.0044996, -0.0015506, 0.0017073
4: 0.0018590, 0.0027212, 0.0018999, 0.0027314, -0.0007260, 0.0006594
5: 0.0076092, 0.0132123, 0.0078751, 0.0132783, -0.0047178, 0.0042847
6: -0.0018126, -0.0003905, -0.0018293, -0.0004579, -0.0010875, 0.0011974
7: -0.0078273, -0.0041478, -0.0078707, -0.0043225, -0.0028137, 0.0030981
8: -0.0036805, -0.0017455, -0.0037033, -0.0018373, -0.0014797, 0.0016293
9: 0.0001601, 0.0024038, 0.0002666, 0.0024303, -0.0018892, 0.0017158

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011210, upper bound: 0.0010650
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010503, upper bound: 0.0009451
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9888144, 0.9918199, 0.9888752, 0.9922723, -0.0027990, 0.0023009
1: -0.0040511, -0.0033022, -0.0040359, -0.0031895, -0.0006974, 0.0005733
2: 0.0074460, 0.0114149, 0.0068487, 0.0113344, -0.0030383, 0.0036960
3: -0.0064687, -0.0046622, -0.0064320, -0.0043903, -0.0016823, 0.0013829
4: 0.0019690, 0.0027372, 0.0018534, 0.0027216, -0.0005881, 0.0007154
5: 0.0083245, 0.0133164, 0.0075733, 0.0132151, -0.0038214, 0.0046486
6: -0.0018390, -0.0005720, -0.0018133, -0.0003813, -0.0011799, 0.0009699
7: -0.0078957, -0.0046176, -0.0078292, -0.0041243, -0.0030527, 0.0025095
8: -0.0037164, -0.0019925, -0.0036814, -0.0017331, -0.0016054, 0.0013197
9: 0.0004466, 0.0024455, 0.0001457, 0.0024050, -0.0015303, 0.0018615

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011242, upper bound: 0.0012078
time: 1.19 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009229, upper bound: 0.0009209
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888064, 0.9918550, 0.9888747, 0.9922870, -0.0028083, 0.0023233
1: -0.0040531, -0.0032935, -0.0040361, -0.0031858, -0.0006998, 0.0005789
2: 0.0073997, 0.0114253, 0.0068292, 0.0113351, -0.0030679, 0.0037083
3: -0.0064734, -0.0046411, -0.0064324, -0.0043815, -0.0016879, 0.0013964
4: 0.0019601, 0.0027392, 0.0018497, 0.0027218, -0.0005938, 0.0007177
5: 0.0082663, 0.0133295, 0.0075488, 0.0132161, -0.0038587, 0.0046641
6: -0.0018423, -0.0005572, -0.0018135, -0.0003751, -0.0011838, 0.0009794
7: -0.0079043, -0.0045794, -0.0078298, -0.0041082, -0.0030628, 0.0025339
8: -0.0037209, -0.0019724, -0.0036818, -0.0017246, -0.0016107, 0.0013326
9: 0.0004232, 0.0024508, 0.0001359, 0.0024053, -0.0015452, 0.0018677

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011379, upper bound: 0.0012360
time: 1.08 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009446, upper bound: 0.0009877
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9888144, 0.9918199, 0.9889256, 0.9924444, -0.0029661, 0.0022650
1: -0.0040511, -0.0033022, -0.0040234, -0.0031466, -0.0007391, 0.0005644
2: 0.0074460, 0.0114149, 0.0066215, 0.0112679, -0.0029908, 0.0039167
3: -0.0064687, -0.0046622, -0.0064018, -0.0042869, -0.0017827, 0.0013613
4: 0.0019690, 0.0027372, 0.0018095, 0.0027088, -0.0005789, 0.0007581
5: 0.0083245, 0.0133164, 0.0072875, 0.0131315, -0.0037617, 0.0049262
6: -0.0018390, -0.0005720, -0.0017921, -0.0003088, -0.0012503, 0.0009548
7: -0.0078957, -0.0046176, -0.0077743, -0.0039366, -0.0032350, 0.0024702
8: -0.0037164, -0.0019925, -0.0036526, -0.0016344, -0.0017012, 0.0012991
9: 0.0004466, 0.0024455, 0.0000313, 0.0023715, -0.0015063, 0.0019727

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012251, upper bound: 0.0012078
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011349, upper bound: 0.0010399
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9888064, 0.9918550, 0.9889251, 0.9924576, -0.0029767, 0.0022980
1: -0.0040531, -0.0032935, -0.0040235, -0.0031433, -0.0007417, 0.0005726
2: 0.0073997, 0.0114253, 0.0066040, 0.0112686, -0.0030345, 0.0039307
3: -0.0064734, -0.0046411, -0.0064021, -0.0042790, -0.0017891, 0.0013812
4: 0.0019601, 0.0027392, 0.0018061, 0.0027089, -0.0005873, 0.0007608
5: 0.0082663, 0.0133295, 0.0072656, 0.0131324, -0.0038166, 0.0049438
6: -0.0018423, -0.0005572, -0.0017923, -0.0003032, -0.0012548, 0.0009687
7: -0.0079043, -0.0045794, -0.0077749, -0.0039222, -0.0032465, 0.0025063
8: -0.0037209, -0.0019724, -0.0036529, -0.0016268, -0.0017073, 0.0013180
9: 0.0004232, 0.0024508, 0.0000225, 0.0023719, -0.0015283, 0.0019797

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012324, upper bound: 0.0012360
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011435, upper bound: 0.0010740
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9888825, 0.9922152, 0.9888752, 0.9922723, -0.0024766, 0.0024400
1: -0.0040341, -0.0032037, -0.0040359, -0.0031895, -0.0006171, 0.0006080
2: 0.0069241, 0.0113249, 0.0068487, 0.0113344, -0.0032220, 0.0032703
3: -0.0064277, -0.0044247, -0.0064320, -0.0043903, -0.0014885, 0.0014665
4: 0.0018680, 0.0027198, 0.0018534, 0.0027216, -0.0006236, 0.0006330
5: 0.0076681, 0.0132032, 0.0075733, 0.0132151, -0.0040524, 0.0041131
6: -0.0018103, -0.0004054, -0.0018133, -0.0003813, -0.0010440, 0.0010285
7: -0.0078214, -0.0041866, -0.0078292, -0.0041243, -0.0027010, 0.0026612
8: -0.0036773, -0.0017658, -0.0036814, -0.0017331, -0.0014204, 0.0013995
9: 0.0001837, 0.0024002, 0.0001457, 0.0024050, -0.0016228, 0.0016471

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014803, upper bound: 0.0014651
time: 1.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014803, upper bound: 0.0014651
time: 1.29 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888770, 0.9922507, 0.9888747, 0.9922870, -0.0024901, 0.0024589
1: -0.0040355, -0.0031949, -0.0040361, -0.0031858, -0.0006205, 0.0006127
2: 0.0068772, 0.0113321, 0.0068292, 0.0113351, -0.0032470, 0.0032882
3: -0.0064310, -0.0044033, -0.0064324, -0.0043815, -0.0014966, 0.0014779
4: 0.0018590, 0.0027212, 0.0018497, 0.0027218, -0.0006285, 0.0006364
5: 0.0076092, 0.0132123, 0.0075488, 0.0132161, -0.0040839, 0.0041357
6: -0.0018126, -0.0003905, -0.0018135, -0.0003751, -0.0010497, 0.0010365
7: -0.0078273, -0.0041478, -0.0078298, -0.0041082, -0.0027159, 0.0026818
8: -0.0036805, -0.0017455, -0.0036818, -0.0017246, -0.0014282, 0.0014103
9: 0.0001601, 0.0024038, 0.0001359, 0.0024053, -0.0016354, 0.0016561

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009649, upper bound: 0.0010172
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007828, upper bound: 0.0007828
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9888825, 0.9922152, 0.9889256, 0.9924444, -0.0027083, 0.0024835
1: -0.0040341, -0.0032037, -0.0040234, -0.0031466, -0.0006748, 0.0006188
2: 0.0069241, 0.0113249, 0.0066215, 0.0112679, -0.0032794, 0.0035762
3: -0.0064277, -0.0044247, -0.0064018, -0.0042869, -0.0016278, 0.0014926
4: 0.0018680, 0.0027198, 0.0018095, 0.0027088, -0.0006347, 0.0006922
5: 0.0076681, 0.0132032, 0.0072875, 0.0131315, -0.0041246, 0.0044980
6: -0.0018103, -0.0004054, -0.0017921, -0.0003088, -0.0011416, 0.0010469
7: -0.0078214, -0.0041866, -0.0077743, -0.0039366, -0.0029538, 0.0027086
8: -0.0036773, -0.0017658, -0.0036526, -0.0016344, -0.0015534, 0.0014244
9: 0.0001837, 0.0024002, 0.0000313, 0.0023715, -0.0016517, 0.0018012

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015380, upper bound: 0.0014651
time: 1.48 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015380, upper bound: 0.0014651
time: 1.55 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9888770, 0.9922507, 0.9889251, 0.9924576, -0.0027189, 0.0025203
1: -0.0040355, -0.0031949, -0.0040235, -0.0031433, -0.0006775, 0.0006280
2: 0.0068772, 0.0113321, 0.0066040, 0.0112686, -0.0033280, 0.0035903
3: -0.0064310, -0.0044033, -0.0064021, -0.0042790, -0.0016342, 0.0015148
4: 0.0018590, 0.0027212, 0.0018061, 0.0027089, -0.0006441, 0.0006949
5: 0.0076092, 0.0132123, 0.0072656, 0.0131324, -0.0041857, 0.0045157
6: -0.0018126, -0.0003905, -0.0017923, -0.0003032, -0.0011461, 0.0010624
7: -0.0078273, -0.0041478, -0.0077749, -0.0039222, -0.0029654, 0.0027487
8: -0.0036805, -0.0017455, -0.0036529, -0.0016268, -0.0015595, 0.0014455
9: 0.0001601, 0.0024038, 0.0000225, 0.0023719, -0.0016762, 0.0018083

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010538, upper bound: 0.0010176
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009236, upper bound: 0.0008455
time: 0.85 seconds

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

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014738
time: 1.25 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0015379
time: 1.25 seconds

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

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014912, upper bound: 0.0014738
time: 1.26 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014912, upper bound: 0.0015464
time: 1.41 seconds

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

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010629, upper bound: 0.0013692
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010764, upper bound: 0.0013694
time: 1.37 seconds

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

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013904, upper bound: 0.0014821
time: 1.29 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014055, upper bound: 0.0014827
time: 1.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9889333, 0.9923872, 0.9888041, 0.9918770, -0.0023142, 0.0029284
1: -0.0040215, -0.0031609, -0.0040537, -0.0032880, -0.0005766, 0.0007297
2: 0.0066969, 0.0112579, 0.0073706, 0.0114284, -0.0038669, 0.0030559
3: -0.0063972, -0.0043213, -0.0064748, -0.0046279, -0.0013909, 0.0017600
4: 0.0018241, 0.0027068, 0.0019545, 0.0027398, -0.0007484, 0.0005915
5: 0.0073824, 0.0131189, 0.0082298, 0.0133334, -0.0048635, 0.0038435
6: -0.0017889, -0.0003329, -0.0018433, -0.0005480, -0.0009755, 0.0012344
7: -0.0077660, -0.0039989, -0.0079069, -0.0045554, -0.0025240, 0.0031938
8: -0.0036482, -0.0016671, -0.0037223, -0.0019598, -0.0013273, 0.0016796
9: 0.0000693, 0.0023664, 0.0004086, 0.0024523, -0.0019476, 0.0015391

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011188, upper bound: 0.0012204
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010648, upper bound: 0.0011227
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9889273, 0.9924192, 0.9888034, 0.9918913, -0.0023190, 0.0029502
1: -0.0040230, -0.0031529, -0.0040538, -0.0032844, -0.0005778, 0.0007351
2: 0.0066546, 0.0112657, 0.0073517, 0.0114292, -0.0038957, 0.0030623
3: -0.0064008, -0.0043020, -0.0064752, -0.0046193, -0.0013938, 0.0017732
4: 0.0018159, 0.0027083, 0.0019508, 0.0027400, -0.0007540, 0.0005927
5: 0.0073292, 0.0131287, 0.0082059, 0.0133343, -0.0048998, 0.0038515
6: -0.0017914, -0.0003194, -0.0018436, -0.0005419, -0.0009776, 0.0012436
7: -0.0077725, -0.0039640, -0.0079075, -0.0045397, -0.0025292, 0.0032176
8: -0.0036516, -0.0016488, -0.0037226, -0.0019516, -0.0013301, 0.0016921
9: 0.0000480, 0.0023704, 0.0003991, 0.0024527, -0.0019621, 0.0015423

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011279, upper bound: 0.0012365
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010740, upper bound: 0.0011435
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9889333, 0.9923872, 0.9888752, 0.9922723, -0.0025348, 0.0026740
1: -0.0040215, -0.0031609, -0.0040359, -0.0031895, -0.0006316, 0.0006663
2: 0.0066969, 0.0112579, 0.0068487, 0.0113344, -0.0035310, 0.0033472
3: -0.0063972, -0.0043213, -0.0064320, -0.0043903, -0.0015235, 0.0016072
4: 0.0018241, 0.0027068, 0.0018534, 0.0027216, -0.0006834, 0.0006478
5: 0.0073824, 0.0131189, 0.0075733, 0.0132151, -0.0044411, 0.0042099
6: -0.0017889, -0.0003329, -0.0018133, -0.0003813, -0.0010685, 0.0011272
7: -0.0077660, -0.0039989, -0.0078292, -0.0041243, -0.0027646, 0.0029164
8: -0.0036482, -0.0016671, -0.0036814, -0.0017331, -0.0014539, 0.0015337
9: 0.0000693, 0.0023664, 0.0001457, 0.0024050, -0.0017784, 0.0016858

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010226, upper bound: 0.0011692
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008167, upper bound: 0.0008593
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9889273, 0.9924192, 0.9888747, 0.9922870, -0.0025377, 0.0026941
1: -0.0040230, -0.0031529, -0.0040361, -0.0031858, -0.0006323, 0.0006713
2: 0.0066546, 0.0112657, 0.0068292, 0.0113351, -0.0035575, 0.0033510
3: -0.0064008, -0.0043020, -0.0064324, -0.0043815, -0.0015252, 0.0016192
4: 0.0018159, 0.0027083, 0.0018497, 0.0027218, -0.0006885, 0.0006486
5: 0.0073292, 0.0131287, 0.0075488, 0.0132161, -0.0044744, 0.0042146
6: -0.0017914, -0.0003194, -0.0018135, -0.0003751, -0.0010697, 0.0011356
7: -0.0077725, -0.0039640, -0.0078298, -0.0041082, -0.0027677, 0.0029383
8: -0.0036516, -0.0016488, -0.0036818, -0.0017246, -0.0014555, 0.0015452
9: 0.0000480, 0.0023704, 0.0001359, 0.0024053, -0.0017917, 0.0016877

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010320, upper bound: 0.0011900
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008455, upper bound: 0.0009236
time: 0.79 seconds

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

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0014738
time: 1.26 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0015380
time: 1.25 seconds

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

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014928, upper bound: 0.0014738
time: 1.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014928, upper bound: 0.0015464
time: 1.20 seconds

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

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011831, upper bound: 0.0012334
time: 1.04 seconds

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

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0012350
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0012350
time: 1.06 seconds

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

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
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
time: 0.93 seconds

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

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0014523
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0014523
time: 0.94 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.39 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014275
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014916
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014916, upper bound: 0.0014275
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014916, upper bound: 0.0015003
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014737, upper bound: 0.0014275
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014737, upper bound: 0.0014912
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0015379, upper bound: 0.0014275
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0015379, upper bound: 0.0014995
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014876, upper bound: 0.0014660
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014876, upper bound: 0.0014660
NS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0010587, upper bound: 0.0010650
NS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0009877, upper bound: 0.0009446
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0015388, upper bound: 0.0014660
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0015388, upper bound: 0.0014660
NS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0011210, upper bound: 0.0010650
NS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0010503, upper bound: 0.0009451
NS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0011242, upper bound: 0.0012078
NS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0009229, upper bound: 0.0009209
NS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0011379, upper bound: 0.0012360
NS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0009446, upper bound: 0.0009877
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0012251, upper bound: 0.0012078
NS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0011349, upper bound: 0.0010399
NS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0012324, upper bound: 0.0012360
NS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0011435, upper bound: 0.0010740
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014803, upper bound: 0.0014651
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014803, upper bound: 0.0014651
NS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0009649, upper bound: 0.0010172
NS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0007828, upper bound: 0.0007828
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0015380, upper bound: 0.0014651
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0015380, upper bound: 0.0014651
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0010538, upper bound: 0.0010176
NS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0009236, upper bound: 0.0008455
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0014738
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014275, upper bound: 0.0015379
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014912, upper bound: 0.0014738
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014912, upper bound: 0.0015464
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0010629, upper bound: 0.0013692
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0010764, upper bound: 0.0013694
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0013904, upper bound: 0.0014821
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014055, upper bound: 0.0014827
NS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0011188, upper bound: 0.0012204
NS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0010648, upper bound: 0.0011227
NS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0011279, upper bound: 0.0012365
NS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0010740, upper bound: 0.0011435
NS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0010226, upper bound: 0.0011692
NS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0008167, upper bound: 0.0008593
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0010320, upper bound: 0.0011900
NS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0008455, upper bound: 0.0009236
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0014738
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0015380
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014928, upper bound: 0.0014738
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0014928, upper bound: 0.0015464
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0011831, upper bound: 0.0012334
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0011831, upper bound: 0.0014415
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0012350
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0012350
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0013728
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0011494
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0014523
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 0, lower bound: -0.0011186, upper bound: 0.0014523

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

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

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0013840
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014031
time: 1.37 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

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

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014510
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014668
time: 1.36 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

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

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014650, upper bound: 0.0013840
time: 1.24 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014668, upper bound: 0.0014031
time: 1.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

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

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014650, upper bound: 0.0014100
time: 1.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014668, upper bound: 0.0014311
time: 1.32 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

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

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014502, upper bound: 0.0013840
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014505, upper bound: 0.0014031
time: 1.34 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

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

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014502, upper bound: 0.0014503
time: 1.28 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014505, upper bound: 0.0014664
time: 1.27 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

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

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015110, upper bound: 0.0013840
time: 1.42 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015129, upper bound: 0.0014031
time: 1.27 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

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

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015110, upper bound: 0.0014100
time: 1.27 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015129, upper bound: 0.0014311
time: 1.67 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9888825, 0.9922152, 0.9888144, 0.9918199, -0.0022841, 0.0027425
1: -0.0040341, -0.0032037, -0.0040511, -0.0033022, -0.0005691, 0.0006834
2: 0.0069241, 0.0113249, 0.0074460, 0.0114149, -0.0036215, 0.0030161
3: -0.0064277, -0.0044247, -0.0064687, -0.0046622, -0.0013728, 0.0016483
4: 0.0018680, 0.0027198, 0.0019690, 0.0027372, -0.0007009, 0.0005838
5: 0.0076681, 0.0132032, 0.0083245, 0.0133164, -0.0045548, 0.0037935
6: -0.0018103, -0.0004054, -0.0018390, -0.0005720, -0.0009628, 0.0011561
7: -0.0078214, -0.0041866, -0.0078957, -0.0046176, -0.0024911, 0.0029911
8: -0.0036773, -0.0017658, -0.0037164, -0.0019925, -0.0013101, 0.0015730
9: 0.0001837, 0.0024002, 0.0004466, 0.0024455, -0.0018240, 0.0015191

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013394, upper bound: 0.0012472
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012428, upper bound: 0.0012002
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9888825, 0.9922152, 0.9888064, 0.9918550, -0.0023088, 0.0027427
1: -0.0040341, -0.0032037, -0.0040531, -0.0032935, -0.0005753, 0.0006834
2: 0.0069241, 0.0113249, 0.0073997, 0.0114253, -0.0036217, 0.0030488
3: -0.0064277, -0.0044247, -0.0064734, -0.0046411, -0.0013877, 0.0016485
4: 0.0018680, 0.0027198, 0.0019601, 0.0027392, -0.0007010, 0.0005901
5: 0.0076681, 0.0132032, 0.0082663, 0.0133295, -0.0045552, 0.0038346
6: -0.0018103, -0.0004054, -0.0018423, -0.0005572, -0.0009733, 0.0011562
7: -0.0078214, -0.0041866, -0.0079043, -0.0045794, -0.0025181, 0.0029913
8: -0.0036773, -0.0017658, -0.0037209, -0.0019724, -0.0013243, 0.0015731
9: 0.0001837, 0.0024002, 0.0004232, 0.0024508, -0.0018241, 0.0015355

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013394, upper bound: 0.0012472
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012428, upper bound: 0.0012002
time: 1.41 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9888825, 0.9922152, 0.9888443, 0.9920229, -0.0025160, 0.0027965
1: -0.0040341, -0.0032037, -0.0040437, -0.0032517, -0.0006269, 0.0006968
2: 0.0069241, 0.0113249, 0.0071781, 0.0113753, -0.0036927, 0.0033224
3: -0.0064277, -0.0044247, -0.0064506, -0.0045403, -0.0015122, 0.0016808
4: 0.0018680, 0.0027198, 0.0019172, 0.0027295, -0.0007147, 0.0006430
5: 0.0076681, 0.0132032, 0.0079876, 0.0132666, -0.0046445, 0.0041787
6: -0.0018103, -0.0004054, -0.0018264, -0.0004865, -0.0010606, 0.0011788
7: -0.0078214, -0.0041866, -0.0078630, -0.0043963, -0.0027441, 0.0030500
8: -0.0036773, -0.0017658, -0.0036992, -0.0018761, -0.0014431, 0.0016039
9: 0.0001837, 0.0024002, 0.0003116, 0.0024256, -0.0018598, 0.0016733

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014225, upper bound: 0.0012472
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013175, upper bound: 0.0012002
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9888825, 0.9922152, 0.9888401, 0.9920518, -0.0025333, 0.0027864
1: -0.0040341, -0.0032037, -0.0040447, -0.0032444, -0.0006312, 0.0006943
2: 0.0069241, 0.0113249, 0.0071398, 0.0113808, -0.0036794, 0.0033453
3: -0.0064277, -0.0044247, -0.0064532, -0.0045228, -0.0015226, 0.0016747
4: 0.0018680, 0.0027198, 0.0019098, 0.0027306, -0.0007121, 0.0006475
5: 0.0076681, 0.0132032, 0.0079394, 0.0132735, -0.0046278, 0.0042075
6: -0.0018103, -0.0004054, -0.0018281, -0.0004743, -0.0010679, 0.0011746
7: -0.0078214, -0.0041866, -0.0078675, -0.0043647, -0.0027630, 0.0030390
8: -0.0036773, -0.0017658, -0.0037016, -0.0018595, -0.0014530, 0.0015982
9: 0.0001837, 0.0024002, 0.0002923, 0.0024284, -0.0018532, 0.0016848

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014225, upper bound: 0.0012472
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013175, upper bound: 0.0012002
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9888825, 0.9922152, 0.9888825, 0.9922152, -0.0024193, 0.0024193
1: -0.0040341, -0.0032037, -0.0040341, -0.0032037, -0.0006028, 0.0006028
2: 0.0069241, 0.0113249, 0.0069241, 0.0113249, -0.0031946, 0.0031946
3: -0.0064277, -0.0044247, -0.0064277, -0.0044247, -0.0014540, 0.0014540
4: 0.0018680, 0.0027198, 0.0018680, 0.0027198, -0.0006183, 0.0006183
5: 0.0076681, 0.0132032, 0.0076681, 0.0132032, -0.0040180, 0.0040180
6: -0.0018103, -0.0004054, -0.0018103, -0.0004054, -0.0010198, 0.0010198
7: -0.0078214, -0.0041866, -0.0078214, -0.0041866, -0.0026385, 0.0026385
8: -0.0036773, -0.0017658, -0.0036773, -0.0017658, -0.0013876, 0.0013876
9: 0.0001837, 0.0024002, 0.0001837, 0.0024002, -0.0016090, 0.0016090

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 56

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012550, upper bound: 0.0012032
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011471, upper bound: 0.0011471
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9888825, 0.9922152, 0.9888770, 0.9922507, -0.0024444, 0.0024208
1: -0.0040341, -0.0032037, -0.0040355, -0.0031949, -0.0006091, 0.0006032
2: 0.0069241, 0.0113249, 0.0068772, 0.0113321, -0.0031967, 0.0032278
3: -0.0064277, -0.0044247, -0.0064310, -0.0044033, -0.0014691, 0.0014550
4: 0.0018680, 0.0027198, 0.0018590, 0.0027212, -0.0006187, 0.0006247
5: 0.0076681, 0.0132032, 0.0076092, 0.0132123, -0.0040206, 0.0040597
6: -0.0018103, -0.0004054, -0.0018126, -0.0003905, -0.0010304, 0.0010205
7: -0.0078214, -0.0041866, -0.0078273, -0.0041478, -0.0026659, 0.0026402
8: -0.0036773, -0.0017658, -0.0036805, -0.0017455, -0.0014020, 0.0013885
9: 0.0001837, 0.0024002, 0.0001601, 0.0024038, -0.0016100, 0.0016257

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 56

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012550, upper bound: 0.0012032
time: 1.31 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011471, upper bound: 0.0011471
time: 1.30 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9888825, 0.9922152, 0.9889333, 0.9923872, -0.0026533, 0.0024775
1: -0.0040341, -0.0032037, -0.0040215, -0.0031609, -0.0006611, 0.0006173
2: 0.0069241, 0.0113249, 0.0066969, 0.0112579, -0.0032716, 0.0035036
3: -0.0064277, -0.0044247, -0.0063972, -0.0043213, -0.0015947, 0.0014891
4: 0.0018680, 0.0027198, 0.0018241, 0.0027068, -0.0006332, 0.0006781
5: 0.0076681, 0.0132032, 0.0073824, 0.0131189, -0.0041148, 0.0044066
6: -0.0018103, -0.0004054, -0.0017889, -0.0003329, -0.0011185, 0.0010444
7: -0.0078214, -0.0041866, -0.0077660, -0.0039989, -0.0028938, 0.0027021
8: -0.0036773, -0.0017658, -0.0036482, -0.0016671, -0.0015218, 0.0014210
9: 0.0001837, 0.0024002, 0.0000693, 0.0023664, -0.0016477, 0.0017646

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 56

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013864, upper bound: 0.0012082
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012754, upper bound: 0.0011545
time: 1.15 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9888825, 0.9922152, 0.9889273, 0.9924192, -0.0026702, 0.0024684
1: -0.0040341, -0.0032037, -0.0040230, -0.0031529, -0.0006653, 0.0006150
2: 0.0069241, 0.0113249, 0.0066546, 0.0112657, -0.0032594, 0.0035259
3: -0.0064277, -0.0044247, -0.0064008, -0.0043020, -0.0016049, 0.0014836
4: 0.0018680, 0.0027198, 0.0018159, 0.0027083, -0.0006309, 0.0006824
5: 0.0076681, 0.0132032, 0.0073292, 0.0131287, -0.0040995, 0.0044347
6: -0.0018103, -0.0004054, -0.0017914, -0.0003194, -0.0011256, 0.0010405
7: -0.0078214, -0.0041866, -0.0077725, -0.0039640, -0.0029122, 0.0026921
8: -0.0036773, -0.0017658, -0.0036516, -0.0016488, -0.0015315, 0.0014157
9: 0.0001837, 0.0024002, 0.0000480, 0.0023704, -0.0016416, 0.0017758

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 56

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013864, upper bound: 0.0012081
time: 1.18 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012754, upper bound: 0.0011545
time: 1.16 seconds

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

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014351
time: 1.37 seconds

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

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014978
time: 1.37 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0015129
time: 1.40 seconds

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

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014645, upper bound: 0.0014351
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014664, upper bound: 0.0014505
time: 1.35 seconds

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

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014645, upper bound: 0.0014594
time: 1.19 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014664, upper bound: 0.0014759
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9892818, 0.9923234, 0.9890327, 0.9922054, -0.0022746, 0.0024493
1: -0.0039347, -0.0031767, -0.0039967, -0.0032062, -0.0005668, 0.0006103
2: 0.0067811, 0.0107976, 0.0069369, 0.0111264, -0.0032343, 0.0030036
3: -0.0061877, -0.0043596, -0.0063374, -0.0044305, -0.0013671, 0.0014721
4: 0.0018403, 0.0026177, 0.0018705, 0.0026814, -0.0006260, 0.0005813
5: 0.0074882, 0.0125400, 0.0076843, 0.0129536, -0.0040678, 0.0037777
6: -0.0016420, -0.0003598, -0.0017469, -0.0004095, -0.0009588, 0.0010325
7: -0.0073859, -0.0040684, -0.0076575, -0.0041972, -0.0024808, 0.0026713
8: -0.0034483, -0.0017037, -0.0035911, -0.0017714, -0.0013046, 0.0014048
9: 0.0001117, 0.0021346, 0.0001902, 0.0023002, -0.0016289, 0.0015128

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010629, upper bound: 0.0013595
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010629, upper bound: 0.0013692
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9892812, 0.9923385, 0.9890256, 0.9922410, -0.0022989, 0.0024660
1: -0.0039348, -0.0031730, -0.0039985, -0.0031973, -0.0005728, 0.0006145
2: 0.0067613, 0.0107984, 0.0068899, 0.0111359, -0.0032563, 0.0030357
3: -0.0061881, -0.0043506, -0.0063417, -0.0044091, -0.0013817, 0.0014821
4: 0.0018365, 0.0026179, 0.0018614, 0.0026832, -0.0006303, 0.0005876
5: 0.0074634, 0.0125410, 0.0076251, 0.0129655, -0.0040956, 0.0038181
6: -0.0016422, -0.0003535, -0.0017499, -0.0003945, -0.0009691, 0.0010395
7: -0.0073865, -0.0040521, -0.0076653, -0.0041583, -0.0025073, 0.0026895
8: -0.0034486, -0.0016951, -0.0035952, -0.0017510, -0.0013186, 0.0014144
9: 0.0001017, 0.0021350, 0.0001665, 0.0023050, -0.0016400, 0.0015289

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010764, upper bound: 0.0013595
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010764, upper bound: 0.0013694
time: 1.13 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9888822, 0.9920760, 0.9888825, 0.9922152, -0.0024146, 0.0025575
1: -0.0040342, -0.0032384, -0.0040341, -0.0032037, -0.0006017, 0.0006373
2: 0.0071079, 0.0113251, 0.0069241, 0.0113249, -0.0033771, 0.0031885
3: -0.0064278, -0.0045083, -0.0064277, -0.0044247, -0.0014512, 0.0015371
4: 0.0019036, 0.0027198, 0.0018680, 0.0027198, -0.0006536, 0.0006171
5: 0.0078993, 0.0132035, 0.0076681, 0.0132032, -0.0042475, 0.0040102
6: -0.0018103, -0.0004641, -0.0018103, -0.0004054, -0.0010178, 0.0010781
7: -0.0078215, -0.0043384, -0.0078214, -0.0041866, -0.0026335, 0.0027893
8: -0.0036774, -0.0018456, -0.0036773, -0.0017658, -0.0013849, 0.0014669
9: 0.0002763, 0.0024003, 0.0001837, 0.0024002, -0.0017009, 0.0016059

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013904, upper bound: 0.0014662
time: 1.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013904, upper bound: 0.0014821
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888818, 0.9920881, 0.9888770, 0.9922507, -0.0024291, 0.0025656
1: -0.0040343, -0.0032354, -0.0040355, -0.0031949, -0.0006053, 0.0006393
2: 0.0070919, 0.0113259, 0.0068772, 0.0113321, -0.0033878, 0.0032075
3: -0.0064282, -0.0045010, -0.0064310, -0.0044033, -0.0014599, 0.0015420
4: 0.0019005, 0.0027200, 0.0018590, 0.0027212, -0.0006557, 0.0006208
5: 0.0078791, 0.0132044, 0.0076092, 0.0132123, -0.0042610, 0.0040342
6: -0.0018106, -0.0004590, -0.0018126, -0.0003905, -0.0010239, 0.0010815
7: -0.0078222, -0.0043251, -0.0078273, -0.0041478, -0.0026492, 0.0027981
8: -0.0036777, -0.0018387, -0.0036805, -0.0017455, -0.0013932, 0.0014715
9: 0.0002682, 0.0024007, 0.0001601, 0.0024038, -0.0017063, 0.0016155

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010066, upper bound: 0.0010583
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008854, upper bound: 0.0009879
time: 0.87 seconds

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

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014084, upper bound: 0.0014353
time: 1.36 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014085, upper bound: 0.0014508
time: 1.32 seconds

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

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014084, upper bound: 0.0014981
time: 1.26 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014085, upper bound: 0.0015131
time: 1.27 seconds

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

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014666, upper bound: 0.0014353
time: 1.29 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014682, upper bound: 0.0014508
time: 1.30 seconds

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

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014666, upper bound: 0.0014594
time: 1.22 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014682, upper bound: 0.0014759
time: 1.23 seconds

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

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011541, upper bound: 0.0011935
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011539, upper bound: 0.0014139
time: 1.39 seconds

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

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0011944
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0012086
time: 1.20 seconds

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

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0014181
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0014365
time: 1.19 seconds

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

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

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

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014568, upper bound: 0.0014085
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014568, upper bound: 0.0014249
time: 1.23 seconds

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

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014568, upper bound: 0.0014085
time: 1.27 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014568, upper bound: 0.0014249
time: 1.34 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.21 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0013840
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014031
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014510
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014668
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014650, upper bound: 0.0013840
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014668, upper bound: 0.0014031
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014650, upper bound: 0.0014100
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014668, upper bound: 0.0014311
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014502, upper bound: 0.0013840
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014505, upper bound: 0.0014031
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014502, upper bound: 0.0014503
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014505, upper bound: 0.0014664
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0015110, upper bound: 0.0013840
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0015129, upper bound: 0.0014031
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0015110, upper bound: 0.0014100
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0015129, upper bound: 0.0014311
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0013394, upper bound: 0.0012472
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0012428, upper bound: 0.0012002
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0013394, upper bound: 0.0012472
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0012428, upper bound: 0.0012002
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014225, upper bound: 0.0012472
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0013175, upper bound: 0.0012002
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014225, upper bound: 0.0012472
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0013175, upper bound: 0.0012002
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0012550, upper bound: 0.0012032
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0011471, upper bound: 0.0011471
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0012550, upper bound: 0.0012032
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0011471, upper bound: 0.0011471
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0013864, upper bound: 0.0012082
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0012754, upper bound: 0.0011545
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0013864, upper bound: 0.0012081
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0012754, upper bound: 0.0011545
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014351
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014505
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0014978
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014031, upper bound: 0.0015129
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014645, upper bound: 0.0014351
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014664, upper bound: 0.0014505
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014645, upper bound: 0.0014594
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014664, upper bound: 0.0014759
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0010629, upper bound: 0.0013595
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0010629, upper bound: 0.0013692
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0010764, upper bound: 0.0013595
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0010764, upper bound: 0.0013694
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0013904, upper bound: 0.0014662
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0013904, upper bound: 0.0014821
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0010066, upper bound: 0.0010583
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0008854, upper bound: 0.0009879
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014084, upper bound: 0.0014353
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014085, upper bound: 0.0014508
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014084, upper bound: 0.0014981
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014085, upper bound: 0.0015131
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014666, upper bound: 0.0014353
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014682, upper bound: 0.0014508
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014666, upper bound: 0.0014594
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014682, upper bound: 0.0014759
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0011541, upper bound: 0.0011935
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0011539, upper bound: 0.0014139
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0011944
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0012086
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0014181
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0014365
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0011137, upper bound: 0.0013402
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0011137, upper bound: 0.0013440
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014568, upper bound: 0.0014085
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014568, upper bound: 0.0014249
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014568, upper bound: 0.0014085
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.0014568, upper bound: 0.0014249

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

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

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013840, upper bound: 0.0013840
time: 1.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013840, upper bound: 0.0013840
time: 1.80 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

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

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013840, upper bound: 0.0014031
time: 1.27 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013840, upper bound: 0.0014031
time: 1.61 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

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

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013840, upper bound: 0.0014510
time: 1.37 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013840, upper bound: 0.0014510
time: 1.27 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

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

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013840, upper bound: 0.0014650
time: 1.42 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013840, upper bound: 0.0014668
time: 1.89 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9888603, 0.9918175, 0.9892485, 0.9921194, -0.0023135, 0.0017227
1: -0.0040397, -0.0033028, -0.0039429, -0.0032276, -0.0005765, 0.0004293
2: 0.0074493, 0.0113541, 0.0070506, 0.0108415, -0.0022748, 0.0030549
3: -0.0064410, -0.0046637, -0.0062077, -0.0044822, -0.0013905, 0.0010354
4: 0.0019697, 0.0027255, 0.0018925, 0.0026262, -0.0004403, 0.0005913
5: 0.0083287, 0.0132399, 0.0078273, 0.0125952, -0.0028611, 0.0038423
6: -0.0018196, -0.0005731, -0.0016560, -0.0004458, -0.0009752, 0.0007262
7: -0.0078455, -0.0046204, -0.0074221, -0.0042911, -0.0025232, 0.0018789
8: -0.0036900, -0.0019940, -0.0034674, -0.0018208, -0.0013269, 0.0009881
9: 0.0004482, 0.0024149, 0.0002474, 0.0021567, -0.0011457, 0.0015386

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012046, upper bound: 0.0010530
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0013840
time: 1.38 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0013840
time: 1.37 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888533, 0.9918526, 0.9892479, 0.9921356, -0.0023387, 0.0017244
1: -0.0040414, -0.0032941, -0.0039431, -0.0032236, -0.0005827, 0.0004297
2: 0.0074029, 0.0113634, 0.0070292, 0.0108423, -0.0022771, 0.0030882
3: -0.0064452, -0.0046426, -0.0062081, -0.0044725, -0.0014056, 0.0010364
4: 0.0019607, 0.0027272, 0.0018884, 0.0026264, -0.0004407, 0.0005977
5: 0.0082704, 0.0132516, 0.0078004, 0.0125962, -0.0028640, 0.0038842
6: -0.0018226, -0.0005583, -0.0016562, -0.0004390, -0.0009858, 0.0007269
7: -0.0078531, -0.0045821, -0.0074228, -0.0042734, -0.0025507, 0.0018808
8: -0.0036940, -0.0019738, -0.0034677, -0.0018115, -0.0013414, 0.0009891
9: 0.0004249, 0.0024196, 0.0002367, 0.0021571, -0.0011469, 0.0015554

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014031
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014031
time: 1.76 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

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
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013378, upper bound: 0.0012846
time: 1.37 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013261, upper bound: 0.0012523
time: 1.35 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

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

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013496, upper bound: 0.0013194
time: 1.27 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013397, upper bound: 0.0012898
time: 1.26 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9892615, 0.9920586, 0.9892818, 0.9923234, -0.0021087, 0.0018132
1: -0.0039397, -0.0032427, -0.0039347, -0.0031767, -0.0005254, 0.0004518
2: 0.0071307, 0.0108243, 0.0067811, 0.0107976, -0.0023943, 0.0027845
3: -0.0061999, -0.0045187, -0.0061877, -0.0043596, -0.0012674, 0.0010898
4: 0.0019080, 0.0026229, 0.0018403, 0.0026177, -0.0004634, 0.0005389
5: 0.0079280, 0.0125736, 0.0074882, 0.0125400, -0.0030114, 0.0035021
6: -0.0016505, -0.0004714, -0.0016420, -0.0003598, -0.0008889, 0.0007643
7: -0.0074079, -0.0043572, -0.0073859, -0.0040684, -0.0022998, 0.0019776
8: -0.0034599, -0.0018556, -0.0034483, -0.0017037, -0.0012094, 0.0010400
9: 0.0002878, 0.0021481, 0.0001117, 0.0021346, -0.0012059, 0.0014024

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014351, upper bound: 0.0013840
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014351, upper bound: 0.0013840
time: 1.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9892507, 0.9921016, 0.9892812, 0.9923385, -0.0021348, 0.0018345
1: -0.0039424, -0.0032320, -0.0039348, -0.0031730, -0.0005319, 0.0004571
2: 0.0070741, 0.0108385, 0.0067613, 0.0107984, -0.0024224, 0.0028190
3: -0.0062063, -0.0044929, -0.0061881, -0.0043506, -0.0012831, 0.0011026
4: 0.0018971, 0.0026257, 0.0018365, 0.0026179, -0.0004688, 0.0005456
5: 0.0078568, 0.0125914, 0.0074634, 0.0125410, -0.0030467, 0.0035455
6: -0.0016550, -0.0004533, -0.0016422, -0.0003535, -0.0008999, 0.0007733
7: -0.0074196, -0.0043105, -0.0073865, -0.0040521, -0.0023283, 0.0020007
8: -0.0034661, -0.0018310, -0.0034486, -0.0016951, -0.0012244, 0.0010522
9: 0.0002593, 0.0021552, 0.0001017, 0.0021350, -0.0012200, 0.0014198

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014351, upper bound: 0.0014031
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014351, upper bound: 0.0014031
time: 1.75 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

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

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014351, upper bound: 0.0014504
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014351, upper bound: 0.0014503
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

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

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014351, upper bound: 0.0014645
time: 1.28 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014351, upper bound: 0.0014664
time: 1.58 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9888603, 0.9918175, 0.9892818, 0.9923234, -0.0026802, 0.0018193
1: -0.0040397, -0.0033028, -0.0039347, -0.0031767, -0.0006678, 0.0004533
2: 0.0074493, 0.0113541, 0.0067811, 0.0107976, -0.0024024, 0.0035392
3: -0.0064410, -0.0046637, -0.0061877, -0.0043596, -0.0016109, 0.0010934
4: 0.0019697, 0.0027255, 0.0018403, 0.0026177, -0.0004650, 0.0006850
5: 0.0083287, 0.0132399, 0.0074882, 0.0125400, -0.0030215, 0.0044514
6: -0.0018196, -0.0005731, -0.0016420, -0.0003598, -0.0011298, 0.0007669
7: -0.0078455, -0.0046204, -0.0073859, -0.0040684, -0.0029231, 0.0019842
8: -0.0036900, -0.0019940, -0.0034483, -0.0017037, -0.0015373, 0.0010435
9: 0.0004482, 0.0024149, 0.0001117, 0.0021346, -0.0012100, 0.0017825

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012567, upper bound: 0.0010530
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010925, upper bound: 0.0007861
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888533, 0.9918526, 0.9892812, 0.9923385, -0.0026912, 0.0018424
1: -0.0040414, -0.0032941, -0.0039348, -0.0031730, -0.0006706, 0.0004591
2: 0.0074029, 0.0113634, 0.0067613, 0.0107984, -0.0024328, 0.0035537
3: -0.0064452, -0.0046426, -0.0061881, -0.0043506, -0.0016175, 0.0011073
4: 0.0019607, 0.0027272, 0.0018365, 0.0026179, -0.0004709, 0.0006878
5: 0.0082704, 0.0132516, 0.0074634, 0.0125410, -0.0030598, 0.0044697
6: -0.0018226, -0.0005583, -0.0016422, -0.0003535, -0.0011344, 0.0007766
7: -0.0078531, -0.0045821, -0.0073865, -0.0040521, -0.0029352, 0.0020093
8: -0.0036940, -0.0019738, -0.0034486, -0.0016951, -0.0015436, 0.0010567
9: 0.0004249, 0.0024196, 0.0001017, 0.0021350, -0.0012253, 0.0017898

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014978, upper bound: 0.0014031
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014978, upper bound: 0.0014031
time: 1.22 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9888603, 0.9918175, 0.9888822, 0.9920760, -0.0021963, 0.0019087
1: -0.0040397, -0.0033028, -0.0040342, -0.0032384, -0.0005473, 0.0004756
2: 0.0074493, 0.0113541, 0.0071079, 0.0113251, -0.0025204, 0.0029002
3: -0.0064410, -0.0046637, -0.0064278, -0.0045083, -0.0013201, 0.0011472
4: 0.0019697, 0.0027255, 0.0019036, 0.0027198, -0.0004878, 0.0005613
5: 0.0083287, 0.0132399, 0.0078993, 0.0132035, -0.0031700, 0.0036477
6: -0.0018196, -0.0005731, -0.0018103, -0.0004641, -0.0009258, 0.0008046
7: -0.0078455, -0.0046204, -0.0078215, -0.0043384, -0.0023954, 0.0020817
8: -0.0036900, -0.0019940, -0.0036774, -0.0018456, -0.0012597, 0.0010947
9: 0.0004482, 0.0024149, 0.0002763, 0.0024003, -0.0012694, 0.0014607

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013855, upper bound: 0.0012846
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013790, upper bound: 0.0012523
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9888533, 0.9918526, 0.9888818, 0.9920881, -0.0022229, 0.0019208
1: -0.0040414, -0.0032941, -0.0040343, -0.0032354, -0.0005539, 0.0004786
2: 0.0074029, 0.0113634, 0.0070919, 0.0113259, -0.0025364, 0.0029353
3: -0.0064452, -0.0046426, -0.0064282, -0.0045010, -0.0013360, 0.0011544
4: 0.0019607, 0.0027272, 0.0019005, 0.0027200, -0.0004909, 0.0005681
5: 0.0082704, 0.0132516, 0.0078791, 0.0132044, -0.0031901, 0.0036918
6: -0.0018226, -0.0005583, -0.0018106, -0.0004590, -0.0009370, 0.0008097
7: -0.0078531, -0.0045821, -0.0078222, -0.0043251, -0.0024244, 0.0020949
8: -0.0036940, -0.0019738, -0.0036777, -0.0018387, -0.0012750, 0.0011017
9: 0.0004249, 0.0024196, 0.0002682, 0.0024007, -0.0012775, 0.0014784

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013983, upper bound: 0.0013194
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013936, upper bound: 0.0012898
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9888905, 0.9922062, 0.9888144, 0.9918199, -0.0022759, 0.0027299
1: -0.0040322, -0.0032060, -0.0040511, -0.0033022, -0.0005671, 0.0006802
2: 0.0069360, 0.0113143, 0.0074460, 0.0114149, -0.0036048, 0.0030053
3: -0.0064229, -0.0044301, -0.0064687, -0.0046622, -0.0013679, 0.0016408
4: 0.0018703, 0.0027178, 0.0019690, 0.0027372, -0.0006977, 0.0005817
5: 0.0076831, 0.0131899, 0.0083245, 0.0133164, -0.0045340, 0.0037799
6: -0.0018069, -0.0004092, -0.0018390, -0.0005720, -0.0009594, 0.0011508
7: -0.0078126, -0.0041964, -0.0078957, -0.0046176, -0.0024822, 0.0029774
8: -0.0036727, -0.0017710, -0.0037164, -0.0019925, -0.0013054, 0.0015658
9: 0.0001897, 0.0023949, 0.0004466, 0.0024455, -0.0018156, 0.0015136

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012428, upper bound: 0.0012002
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012428, upper bound: 0.0012002
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9888905, 0.9922062, 0.9888064, 0.9918550, -0.0023007, 0.0027302
1: -0.0040322, -0.0032060, -0.0040531, -0.0032935, -0.0005733, 0.0006803
2: 0.0069360, 0.0113143, 0.0073997, 0.0114253, -0.0036051, 0.0030380
3: -0.0064229, -0.0044301, -0.0064734, -0.0046411, -0.0013828, 0.0016409
4: 0.0018703, 0.0027178, 0.0019601, 0.0027392, -0.0006978, 0.0005880
5: 0.0076831, 0.0131899, 0.0082663, 0.0133295, -0.0045343, 0.0038210
6: -0.0018069, -0.0004092, -0.0018423, -0.0005572, -0.0009698, 0.0011509
7: -0.0078126, -0.0041964, -0.0079043, -0.0045794, -0.0025092, 0.0029776
8: -0.0036727, -0.0017710, -0.0037209, -0.0019724, -0.0013196, 0.0015659
9: 0.0001897, 0.0023949, 0.0004232, 0.0024508, -0.0018157, 0.0015301

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012565, upper bound: 0.0012002
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012565, upper bound: 0.0012002
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9888905, 0.9922062, 0.9888443, 0.9920229, -0.0025079, 0.0027839
1: -0.0040322, -0.0032060, -0.0040437, -0.0032517, -0.0006249, 0.0006937
2: 0.0069360, 0.0113143, 0.0071781, 0.0113753, -0.0036761, 0.0033116
3: -0.0064229, -0.0044301, -0.0064506, -0.0045403, -0.0015073, 0.0016732
4: 0.0018703, 0.0027178, 0.0019172, 0.0027295, -0.0007115, 0.0006410
5: 0.0076831, 0.0131899, 0.0079876, 0.0132666, -0.0046236, 0.0041651
6: -0.0018069, -0.0004092, -0.0018264, -0.0004865, -0.0010571, 0.0011735
7: -0.0078126, -0.0041964, -0.0078630, -0.0043963, -0.0027352, 0.0030362
8: -0.0036727, -0.0017710, -0.0036992, -0.0018761, -0.0014384, 0.0015967
9: 0.0001897, 0.0023949, 0.0003116, 0.0024256, -0.0018515, 0.0016679

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013175, upper bound: 0.0012002
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013175, upper bound: 0.0012002
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888216, 0.9922175, 0.9888515, 0.9920161, -0.0025571, 0.0028080
1: -0.0040493, -0.0032031, -0.0040419, -0.0032533, -0.0006372, 0.0006997
2: 0.0069209, 0.0114053, 0.0071869, 0.0113657, -0.0037079, 0.0033766
3: -0.0064643, -0.0044232, -0.0064463, -0.0045443, -0.0015369, 0.0016877
4: 0.0018674, 0.0027354, 0.0019189, 0.0027277, -0.0007177, 0.0006535
5: 0.0076642, 0.0133043, 0.0079986, 0.0132545, -0.0046636, 0.0042469
6: -0.0018359, -0.0004044, -0.0018233, -0.0004893, -0.0010779, 0.0011837
7: -0.0078878, -0.0041840, -0.0078551, -0.0044036, -0.0027889, 0.0030625
8: -0.0037122, -0.0017645, -0.0036951, -0.0018800, -0.0014667, 0.0016106
9: 0.0001821, 0.0024407, 0.0003161, 0.0024207, -0.0018675, 0.0017007

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012847, upper bound: 0.0011903
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012968, upper bound: 0.0011836
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9888905, 0.9922062, 0.9888401, 0.9920518, -0.0025252, 0.0027738
1: -0.0040322, -0.0032060, -0.0040447, -0.0032444, -0.0006292, 0.0006912
2: 0.0069360, 0.0113143, 0.0071398, 0.0113808, -0.0036628, 0.0033344
3: -0.0064229, -0.0044301, -0.0064532, -0.0045228, -0.0015177, 0.0016672
4: 0.0018703, 0.0027178, 0.0019098, 0.0027306, -0.0007089, 0.0006454
5: 0.0076831, 0.0131899, 0.0079394, 0.0132735, -0.0046069, 0.0041939
6: -0.0018069, -0.0004092, -0.0018281, -0.0004743, -0.0010644, 0.0011693
7: -0.0078126, -0.0041964, -0.0078675, -0.0043647, -0.0027540, 0.0030253
8: -0.0036727, -0.0017710, -0.0037016, -0.0018595, -0.0014483, 0.0015910
9: 0.0001897, 0.0023949, 0.0002923, 0.0024284, -0.0018448, 0.0016794

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013286, upper bound: 0.0012002
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013286, upper bound: 0.0012002
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9888216, 0.9922175, 0.9888474, 0.9920451, -0.0025745, 0.0027978
1: -0.0040493, -0.0032031, -0.0040429, -0.0032461, -0.0006415, 0.0006971
2: 0.0069209, 0.0114053, 0.0071486, 0.0113712, -0.0036945, 0.0033996
3: -0.0064643, -0.0044232, -0.0064488, -0.0045269, -0.0015474, 0.0016816
4: 0.0018674, 0.0027354, 0.0019115, 0.0027288, -0.0007151, 0.0006580
5: 0.0076642, 0.0133043, 0.0079506, 0.0132614, -0.0046467, 0.0042759
6: -0.0018359, -0.0004044, -0.0018250, -0.0004771, -0.0010853, 0.0011794
7: -0.0078878, -0.0041840, -0.0078596, -0.0043720, -0.0028079, 0.0030514
8: -0.0037122, -0.0017645, -0.0036974, -0.0018634, -0.0014766, 0.0016047
9: 0.0001821, 0.0024407, 0.0002968, 0.0024235, -0.0018607, 0.0017122

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012943, upper bound: 0.0011903
time: 1.28 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013103, upper bound: 0.0011836
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9888905, 0.9922062, 0.9889333, 0.9923872, -0.0026452, 0.0024655
1: -0.0040322, -0.0032060, -0.0040215, -0.0031609, -0.0006591, 0.0006143
2: 0.0069360, 0.0113143, 0.0066969, 0.0112579, -0.0032557, 0.0034929
3: -0.0064229, -0.0044301, -0.0063972, -0.0043213, -0.0015898, 0.0014819
4: 0.0018703, 0.0027178, 0.0018241, 0.0027068, -0.0006301, 0.0006760
5: 0.0076831, 0.0131899, 0.0073824, 0.0131189, -0.0040948, 0.0043931
6: -0.0018069, -0.0004092, -0.0017889, -0.0003329, -0.0011150, 0.0010393
7: -0.0078126, -0.0041964, -0.0077660, -0.0039989, -0.0028849, 0.0026890
8: -0.0036727, -0.0017710, -0.0036482, -0.0016671, -0.0015171, 0.0014141
9: 0.0001897, 0.0023949, 0.0000693, 0.0023664, -0.0016397, 0.0017592

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012754, upper bound: 0.0011545
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012754, upper bound: 0.0011545
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888216, 0.9922175, 0.9889402, 0.9923803, -0.0026889, 0.0024904
1: -0.0040493, -0.0032031, -0.0040197, -0.0031626, -0.0006700, 0.0006205
2: 0.0069209, 0.0114053, 0.0067061, 0.0112485, -0.0032886, 0.0035507
3: -0.0064643, -0.0044232, -0.0063930, -0.0043254, -0.0016161, 0.0014968
4: 0.0018674, 0.0027354, 0.0018258, 0.0027050, -0.0006365, 0.0006872
5: 0.0076642, 0.0133043, 0.0073939, 0.0131071, -0.0041362, 0.0044658
6: -0.0018359, -0.0004044, -0.0017859, -0.0003358, -0.0011335, 0.0010498
7: -0.0078878, -0.0041840, -0.0077583, -0.0040065, -0.0029326, 0.0027162
8: -0.0037122, -0.0017645, -0.0036442, -0.0016711, -0.0015422, 0.0014284
9: 0.0001821, 0.0024407, 0.0000739, 0.0023617, -0.0016563, 0.0017883

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012402, upper bound: 0.0011442
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012522, upper bound: 0.0011366
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9888905, 0.9922062, 0.9889273, 0.9924192, -0.0026620, 0.0024563
1: -0.0040322, -0.0032060, -0.0040230, -0.0031529, -0.0006633, 0.0006121
2: 0.0069360, 0.0113143, 0.0066546, 0.0112657, -0.0032436, 0.0035152
3: -0.0064229, -0.0044301, -0.0064008, -0.0043020, -0.0016000, 0.0014763
4: 0.0018703, 0.0027178, 0.0018159, 0.0027083, -0.0006278, 0.0006804
5: 0.0076831, 0.0131899, 0.0073292, 0.0131287, -0.0040796, 0.0044212
6: -0.0018069, -0.0004092, -0.0017914, -0.0003194, -0.0011221, 0.0010354
7: -0.0078126, -0.0041964, -0.0077725, -0.0039640, -0.0029033, 0.0026790
8: -0.0036727, -0.0017710, -0.0036516, -0.0016488, -0.0015268, 0.0014089
9: 0.0001897, 0.0023949, 0.0000480, 0.0023704, -0.0016336, 0.0017704

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012817, upper bound: 0.0011545
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012817, upper bound: 0.0011545
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9888216, 0.9922175, 0.9889345, 0.9924122, -0.0027057, 0.0024811
1: -0.0040493, -0.0032031, -0.0040212, -0.0031546, -0.0006742, 0.0006182
2: 0.0069209, 0.0114053, 0.0066639, 0.0112563, -0.0032762, 0.0035728
3: -0.0064643, -0.0044232, -0.0063965, -0.0043062, -0.0016262, 0.0014912
4: 0.0018674, 0.0027354, 0.0018177, 0.0027065, -0.0006341, 0.0006915
5: 0.0076642, 0.0133043, 0.0073408, 0.0131169, -0.0041206, 0.0044937
6: -0.0018359, -0.0004044, -0.0017884, -0.0003223, -0.0011405, 0.0010459
7: -0.0078878, -0.0041840, -0.0077647, -0.0039716, -0.0029509, 0.0027059
8: -0.0037122, -0.0017645, -0.0036475, -0.0016528, -0.0015519, 0.0014230
9: 0.0001821, 0.0024407, 0.0000526, 0.0023656, -0.0016501, 0.0017995

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012460, upper bound: 0.0011442
time: 1.08 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012594, upper bound: 0.0011364
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9892911, 0.9922644, 0.9892485, 0.9921194, -0.0018584, 0.0020905
1: -0.0039323, -0.0031914, -0.0039429, -0.0032276, -0.0004631, 0.0005209
2: 0.0068589, 0.0107853, 0.0070506, 0.0108415, -0.0027605, 0.0024540
3: -0.0061821, -0.0043950, -0.0062077, -0.0044822, -0.0011170, 0.0012565
4: 0.0018554, 0.0026154, 0.0018925, 0.0026262, -0.0005343, 0.0004750
5: 0.0075861, 0.0125245, 0.0078273, 0.0125952, -0.0034720, 0.0030865
6: -0.0016380, -0.0003846, -0.0016560, -0.0004458, -0.0007834, 0.0008812
7: -0.0073757, -0.0041327, -0.0074221, -0.0042911, -0.0020269, 0.0022800
8: -0.0034430, -0.0017375, -0.0034674, -0.0018208, -0.0010659, 0.0011990
9: 0.0001509, 0.0021284, 0.0002474, 0.0021567, -0.0013903, 0.0012360

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008972, upper bound: 0.0010617
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013840, upper bound: 0.0014351
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013840, upper bound: 0.0014351
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9892840, 0.9923024, 0.9892479, 0.9921356, -0.0018855, 0.0020931
1: -0.0039341, -0.0031820, -0.0039431, -0.0032236, -0.0004698, 0.0005215
2: 0.0068089, 0.0107946, 0.0070292, 0.0108423, -0.0027639, 0.0024898
3: -0.0061864, -0.0043722, -0.0062081, -0.0044725, -0.0011332, 0.0012580
4: 0.0018457, 0.0026172, 0.0018884, 0.0026264, -0.0005349, 0.0004819
5: 0.0075232, 0.0125362, 0.0078004, 0.0125962, -0.0034762, 0.0031315
6: -0.0016410, -0.0003686, -0.0016562, -0.0004390, -0.0007948, 0.0008823
7: -0.0073834, -0.0040914, -0.0074228, -0.0042734, -0.0020564, 0.0022828
8: -0.0034470, -0.0017158, -0.0034677, -0.0018115, -0.0010814, 0.0012005
9: 0.0001257, 0.0021331, 0.0002367, 0.0021571, -0.0013920, 0.0012540

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013840, upper bound: 0.0014502
time: 1.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013840, upper bound: 0.0014505
time: 1.36 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9892911, 0.9922644, 0.9888510, 0.9918745, -0.0018588, 0.0026407
1: -0.0039323, -0.0031914, -0.0040420, -0.0032886, -0.0004632, 0.0006580
2: 0.0068589, 0.0107853, 0.0073740, 0.0113665, -0.0034870, 0.0024545
3: -0.0061821, -0.0043950, -0.0064466, -0.0046294, -0.0011172, 0.0015871
4: 0.0018554, 0.0026154, 0.0019551, 0.0027278, -0.0006749, 0.0004751
5: 0.0075861, 0.0125245, 0.0082340, 0.0132555, -0.0043858, 0.0030871
6: -0.0016380, -0.0003846, -0.0018235, -0.0005490, -0.0007835, 0.0011132
7: -0.0073757, -0.0041327, -0.0078557, -0.0045582, -0.0020272, 0.0028801
8: -0.0034430, -0.0017375, -0.0036954, -0.0019612, -0.0010661, 0.0015146
9: 0.0001509, 0.0021284, 0.0004103, 0.0024211, -0.0017563, 0.0012362

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009102, upper bound: 0.0012174
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008091, upper bound: 0.0010741
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9892840, 0.9923024, 0.9888504, 0.9918888, -0.0018709, 0.0026652
1: -0.0039341, -0.0031820, -0.0040421, -0.0032850, -0.0004662, 0.0006641
2: 0.0068089, 0.0107946, 0.0073550, 0.0113672, -0.0035194, 0.0024705
3: -0.0061864, -0.0043722, -0.0064470, -0.0046208, -0.0011245, 0.0016019
4: 0.0018457, 0.0026172, 0.0019514, 0.0027280, -0.0006812, 0.0004782
5: 0.0075232, 0.0125362, 0.0082101, 0.0132564, -0.0044264, 0.0031072
6: -0.0016410, -0.0003686, -0.0018238, -0.0005430, -0.0007886, 0.0011235
7: -0.0073834, -0.0040914, -0.0078563, -0.0045425, -0.0020405, 0.0029068
8: -0.0034470, -0.0017158, -0.0036957, -0.0019530, -0.0010731, 0.0015287
9: 0.0001257, 0.0021331, 0.0004008, 0.0024215, -0.0017725, 0.0012443

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009372, upper bound: 0.0012491
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008443, upper bound: 0.0011187
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9888886, 0.9920204, 0.9892485, 0.9921194, -0.0023510, 0.0019558
1: -0.0040326, -0.0032523, -0.0039429, -0.0032276, -0.0005858, 0.0004873
2: 0.0071812, 0.0113169, 0.0070506, 0.0108415, -0.0025826, 0.0031044
3: -0.0064241, -0.0045417, -0.0062077, -0.0044822, -0.0014130, 0.0011755
4: 0.0019178, 0.0027182, 0.0018925, 0.0026262, -0.0004999, 0.0006009
5: 0.0079916, 0.0131931, 0.0078273, 0.0125952, -0.0032482, 0.0039046
6: -0.0018077, -0.0004875, -0.0016560, -0.0004458, -0.0009910, 0.0008244
7: -0.0078147, -0.0043990, -0.0074221, -0.0042911, -0.0025641, 0.0021330
8: -0.0036738, -0.0018775, -0.0034674, -0.0018208, -0.0013484, 0.0011217
9: 0.0003132, 0.0023962, 0.0002474, 0.0021567, -0.0013007, 0.0015636

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012046, upper bound: 0.0011327
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014503, upper bound: 0.0014351
time: 1.21 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014503, upper bound: 0.0014351
time: 1.63 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888846, 0.9920495, 0.9892479, 0.9921356, -0.0023693, 0.0019585
1: -0.0040336, -0.0032450, -0.0039431, -0.0032236, -0.0005904, 0.0004880
2: 0.0071429, 0.0113221, 0.0070292, 0.0108423, -0.0025862, 0.0031287
3: -0.0064264, -0.0045243, -0.0062081, -0.0044725, -0.0014240, 0.0011771
4: 0.0019104, 0.0027193, 0.0018884, 0.0026264, -0.0005006, 0.0006055
5: 0.0079434, 0.0131997, 0.0078004, 0.0125962, -0.0032528, 0.0039350
6: -0.0018094, -0.0004753, -0.0016562, -0.0004390, -0.0009988, 0.0008256
7: -0.0078190, -0.0043673, -0.0074228, -0.0042734, -0.0025841, 0.0021361
8: -0.0036761, -0.0018609, -0.0034677, -0.0018115, -0.0013589, 0.0011233
9: 0.0002939, 0.0023988, 0.0002367, 0.0021571, -0.0013026, 0.0015758

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014504, upper bound: 0.0014502
time: 1.21 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014504, upper bound: 0.0014505
time: 1.27 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9888886, 0.9920204, 0.9888510, 0.9918745, -0.0019493, 0.0021794
1: -0.0040326, -0.0032523, -0.0040420, -0.0032886, -0.0004857, 0.0005431
2: 0.0071812, 0.0113169, 0.0073740, 0.0113665, -0.0028779, 0.0025740
3: -0.0064241, -0.0045417, -0.0064466, -0.0046294, -0.0011716, 0.0013099
4: 0.0019178, 0.0027182, 0.0019551, 0.0027278, -0.0005570, 0.0004982
5: 0.0079916, 0.0131931, 0.0082340, 0.0132555, -0.0036196, 0.0032374
6: -0.0018077, -0.0004875, -0.0018235, -0.0005490, -0.0008217, 0.0009187
7: -0.0078147, -0.0043990, -0.0078557, -0.0045582, -0.0021260, 0.0023770
8: -0.0036738, -0.0018775, -0.0036954, -0.0019612, -0.0011180, 0.0012500
9: 0.0003132, 0.0023962, 0.0004103, 0.0024211, -0.0014495, 0.0012964

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013378, upper bound: 0.0013428
time: 1.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013261, upper bound: 0.0013118
time: 1.19 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9888846, 0.9920495, 0.9888504, 0.9918888, -0.0019764, 0.0021757
1: -0.0040336, -0.0032450, -0.0040421, -0.0032850, -0.0004925, 0.0005421
2: 0.0071429, 0.0113221, 0.0073550, 0.0113672, -0.0028730, 0.0026099
3: -0.0064264, -0.0045243, -0.0064470, -0.0046208, -0.0011879, 0.0013077
4: 0.0019104, 0.0027193, 0.0019514, 0.0027280, -0.0005561, 0.0005051
5: 0.0079434, 0.0131997, 0.0082101, 0.0132564, -0.0036134, 0.0032825
6: -0.0018094, -0.0004753, -0.0018238, -0.0005430, -0.0008331, 0.0009171
7: -0.0078190, -0.0043673, -0.0078563, -0.0045425, -0.0021556, 0.0023729
8: -0.0036761, -0.0018609, -0.0036957, -0.0019530, -0.0011336, 0.0012479
9: 0.0002939, 0.0023988, 0.0004008, 0.0024215, -0.0014470, 0.0013145

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013496, upper bound: 0.0013721
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013397, upper bound: 0.0013460
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9892911, 0.9922644, 0.9890327, 0.9922054, -0.0022563, 0.0023929
1: -0.0039323, -0.0031914, -0.0039967, -0.0032062, -0.0005622, 0.0005962
2: 0.0068589, 0.0107853, 0.0069369, 0.0111264, -0.0031598, 0.0029794
3: -0.0061821, -0.0043950, -0.0063374, -0.0044305, -0.0013561, 0.0014382
4: 0.0018554, 0.0026154, 0.0018705, 0.0026814, -0.0006116, 0.0005766
5: 0.0075861, 0.0125245, 0.0076843, 0.0129536, -0.0039742, 0.0037473
6: -0.0016380, -0.0003846, -0.0017469, -0.0004095, -0.0009511, 0.0010087
7: -0.0073757, -0.0041327, -0.0076575, -0.0041972, -0.0024608, 0.0026098
8: -0.0034430, -0.0017375, -0.0035911, -0.0017714, -0.0012941, 0.0013725
9: 0.0001509, 0.0021284, 0.0001902, 0.0023002, -0.0015914, 0.0015006

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010492, upper bound: 0.0013485
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010552, upper bound: 0.0013499
time: 1.13 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9892840, 0.9923024, 0.9890327, 0.9922054, -0.0022565, 0.0024210
1: -0.0039341, -0.0031820, -0.0039967, -0.0032062, -0.0005623, 0.0006032
2: 0.0068089, 0.0107946, 0.0069369, 0.0111264, -0.0031969, 0.0029797
3: -0.0061864, -0.0043722, -0.0063374, -0.0044305, -0.0013562, 0.0014551
4: 0.0018457, 0.0026172, 0.0018705, 0.0026814, -0.0006188, 0.0005767
5: 0.0075232, 0.0125362, 0.0076843, 0.0129536, -0.0040209, 0.0037476
6: -0.0016410, -0.0003686, -0.0017469, -0.0004095, -0.0009512, 0.0010205
7: -0.0073834, -0.0040914, -0.0076575, -0.0041972, -0.0024610, 0.0026404
8: -0.0034470, -0.0017158, -0.0035911, -0.0017714, -0.0012942, 0.0013886
9: 0.0001257, 0.0021331, 0.0001902, 0.0023002, -0.0016101, 0.0015007

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010492, upper bound: 0.0013551
time: 1.08 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010552, upper bound: 0.0013562
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9892911, 0.9922644, 0.9890256, 0.9922410, -0.0022737, 0.0023970
1: -0.0039323, -0.0031914, -0.0039985, -0.0031973, -0.0005665, 0.0005973
2: 0.0068589, 0.0107853, 0.0068899, 0.0111359, -0.0031652, 0.0030024
3: -0.0061821, -0.0043950, -0.0063417, -0.0044091, -0.0013666, 0.0014406
4: 0.0018554, 0.0026154, 0.0018614, 0.0026832, -0.0006126, 0.0005811
5: 0.0075861, 0.0125245, 0.0076251, 0.0129655, -0.0039809, 0.0037762
6: -0.0016380, -0.0003846, -0.0017499, -0.0003945, -0.0009584, 0.0010104
7: -0.0073757, -0.0041327, -0.0076653, -0.0041583, -0.0024798, 0.0026142
8: -0.0034430, -0.0017375, -0.0035952, -0.0017510, -0.0013041, 0.0013748
9: 0.0001509, 0.0021284, 0.0001665, 0.0023050, -0.0015941, 0.0015122

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010492, upper bound: 0.0013458
time: 1.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010552, upper bound: 0.0013464
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9892840, 0.9923024, 0.9890256, 0.9922410, -0.0022835, 0.0024169
1: -0.0039341, -0.0031820, -0.0039985, -0.0031973, -0.0005690, 0.0006022
2: 0.0068089, 0.0107946, 0.0068899, 0.0111359, -0.0031915, 0.0030154
3: -0.0061864, -0.0043722, -0.0063417, -0.0044091, -0.0013725, 0.0014526
4: 0.0018457, 0.0026172, 0.0018614, 0.0026832, -0.0006177, 0.0005836
5: 0.0075232, 0.0125362, 0.0076251, 0.0129655, -0.0040141, 0.0037926
6: -0.0016410, -0.0003686, -0.0017499, -0.0003945, -0.0009626, 0.0010188
7: -0.0073834, -0.0040914, -0.0076653, -0.0041583, -0.0024905, 0.0026360
8: -0.0034470, -0.0017158, -0.0035952, -0.0017510, -0.0013097, 0.0013862
9: 0.0001257, 0.0021331, 0.0001665, 0.0023050, -0.0016074, 0.0015187

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.50 + 598.32 = 601.82 seconds
