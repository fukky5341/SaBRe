## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000586925


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0033858, 0.0033858)
1: (-0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009546, 0.0009546)
2: (-0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0070432, 0.0070432)
3: (0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009321, 0.0009321)
4: (0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0052637, 0.0052637)
5: (0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014624, 0.0014624)
6: (0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0013274, 0.0013274)
7: (-0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0049537, 0.0049537)
8: (-0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0038555, 0.0038555)
9: (-0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003326, 0.0003326)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 2.21 = 3.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0006905, upper bound: 0.0006905

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006638, upper bound: 0.0006238
time: 1.26 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006637, upper bound: 0.0006638
time: 1.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.66 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.66
Output dim: 5, lower bound: -0.0006638, upper bound: 0.0006238
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.66
Output dim: 5, lower bound: -0.0006637, upper bound: 0.0006638

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0089774, -0.0052066, -0.0092176, -0.0051294, -0.0029041, 0.0029895
1: -0.0054697, -0.0044066, -0.0055375, -0.0043848, -0.0008188, 0.0008428
2: -0.0017969, 0.0060472, -0.0022967, 0.0062076, -0.0060411, 0.0062187
3: 0.0013895, 0.0024275, 0.0013234, 0.0024488, -0.0007994, 0.0008230
4: 0.0015726, 0.0074347, 0.0014527, 0.0078083, -0.0046475, 0.0045147
5: 0.9959432, 0.9975718, 0.9959099, 0.9976756, -0.0012912, 0.0012543
6: 0.0042013, 0.0056796, 0.0041710, 0.0057738, -0.0011720, 0.0011385
7: -0.0077031, -0.0021862, -0.0078160, -0.0018347, -0.0043738, 0.0042488
8: -0.0074913, -0.0031975, -0.0077649, -0.0031097, -0.0033069, 0.0034041
9: -0.0037339, -0.0033634, -0.0037414, -0.0033398, -0.0002937, 0.0002853

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006239, upper bound: 0.0006238
time: 1.27 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006239, upper bound: 0.0006236
time: 1.79 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0093042, -0.0051274, -0.0093743, -0.0051257, -0.0028370, 0.0033599
1: -0.0055619, -0.0043843, -0.0055816, -0.0043838, -0.0007999, 0.0009473
2: -0.0024768, 0.0062117, -0.0026226, 0.0062154, -0.0059016, 0.0069892
3: 0.0012995, 0.0024493, 0.0012802, 0.0024498, -0.0007810, 0.0009249
4: 0.0014496, 0.0079429, 0.0014469, 0.0080518, -0.0052233, 0.0044105
5: 0.9959090, 0.9977130, 0.9959082, 0.9977432, -0.0014512, 0.0012254
6: 0.0041702, 0.0058077, 0.0041696, 0.0058352, -0.0013172, 0.0011123
7: -0.0078189, -0.0017080, -0.0078215, -0.0016055, -0.0049157, 0.0041508
8: -0.0078635, -0.0031074, -0.0079433, -0.0031054, -0.0032306, 0.0038259
9: -0.0037416, -0.0033313, -0.0037418, -0.0033244, -0.0003301, 0.0002787

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006239, upper bound: 0.0006638
time: 1.30 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006239, upper bound: 0.0006638
time: 1.39 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.31 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.31
Output dim: 5, lower bound: -0.0006239, upper bound: 0.0006238
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.31
Output dim: 5, lower bound: -0.0006239, upper bound: 0.0006236
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.31
Output dim: 5, lower bound: -0.0006239, upper bound: 0.0006638
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.31
Output dim: 5, lower bound: -0.0006239, upper bound: 0.0006638

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0089774, -0.0052066, -0.0089774, -0.0052066, -0.0027404, 0.0027404
1: -0.0054697, -0.0044066, -0.0054697, -0.0044066, -0.0007726, 0.0007726
2: -0.0017969, 0.0060472, -0.0017969, 0.0060472, -0.0057005, 0.0057005
3: 0.0013895, 0.0024275, 0.0013895, 0.0024275, -0.0007544, 0.0007544
4: 0.0015726, 0.0074347, 0.0015726, 0.0074347, -0.0042602, 0.0042602
5: 0.9959432, 0.9975718, 0.9959432, 0.9975718, -0.0011836, 0.0011836
6: 0.0042013, 0.0056796, 0.0042013, 0.0056796, -0.0010744, 0.0010744
7: -0.0077031, -0.0021862, -0.0077031, -0.0021862, -0.0040093, 0.0040093
8: -0.0074913, -0.0031975, -0.0074913, -0.0031975, -0.0031205, 0.0031205
9: -0.0037339, -0.0033634, -0.0037339, -0.0033634, -0.0002692, 0.0002692

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006164, upper bound: 0.0005965
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006163, upper bound: 0.0006097
time: 1.33 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0089774, -0.0052066, -0.0093042, -0.0051274, -0.0029051, 0.0031578
1: -0.0054697, -0.0044066, -0.0055619, -0.0043843, -0.0008191, 0.0008903
2: -0.0017969, 0.0060472, -0.0024768, 0.0062117, -0.0060432, 0.0065688
3: 0.0013895, 0.0024275, 0.0012995, 0.0024493, -0.0007997, 0.0008693
4: 0.0015726, 0.0074347, 0.0014496, 0.0079429, -0.0049091, 0.0045163
5: 0.9959432, 0.9975718, 0.9959090, 0.9977130, -0.0013639, 0.0012548
6: 0.0042013, 0.0056796, 0.0041702, 0.0058077, -0.0012380, 0.0011390
7: -0.0077031, -0.0021862, -0.0078189, -0.0017080, -0.0046200, 0.0042504
8: -0.0074913, -0.0031975, -0.0078635, -0.0031074, -0.0033081, 0.0035958
9: -0.0037339, -0.0033634, -0.0037416, -0.0033313, -0.0003102, 0.0002854

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006164, upper bound: 0.0005964
time: 1.89 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006163, upper bound: 0.0006096
time: 1.92 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0093042, -0.0051274, -0.0089774, -0.0052066, -0.0031578, 0.0029051
1: -0.0055619, -0.0043843, -0.0054697, -0.0044066, -0.0008903, 0.0008191
2: -0.0024768, 0.0062117, -0.0017969, 0.0060472, -0.0065688, 0.0060432
3: 0.0012995, 0.0024493, 0.0013895, 0.0024275, -0.0008693, 0.0007997
4: 0.0014496, 0.0079429, 0.0015726, 0.0074347, -0.0045163, 0.0049091
5: 0.9959090, 0.9977130, 0.9959432, 0.9975718, -0.0012548, 0.0013639
6: 0.0041702, 0.0058077, 0.0042013, 0.0056796, -0.0011390, 0.0012380
7: -0.0078189, -0.0017080, -0.0077031, -0.0021862, -0.0042504, 0.0046200
8: -0.0078635, -0.0031074, -0.0074913, -0.0031975, -0.0035958, 0.0033081
9: -0.0037416, -0.0033313, -0.0037339, -0.0033634, -0.0002854, 0.0003102

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006096, upper bound: 0.0006320
time: 1.30 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006096, upper bound: 0.0006510
time: 1.35 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0093042, -0.0051274, -0.0093042, -0.0051274, -0.0028358, 0.0028358
1: -0.0055619, -0.0043843, -0.0055619, -0.0043843, -0.0007995, 0.0007995
2: -0.0024768, 0.0062117, -0.0024768, 0.0062117, -0.0058990, 0.0058990
3: 0.0012995, 0.0024493, 0.0012995, 0.0024493, -0.0007806, 0.0007806
4: 0.0014496, 0.0079429, 0.0014496, 0.0079429, -0.0044085, 0.0044085
5: 0.9959090, 0.9977130, 0.9959090, 0.9977130, -0.0012248, 0.0012248
6: 0.0041702, 0.0058077, 0.0041702, 0.0058077, -0.0011118, 0.0011118
7: -0.0078189, -0.0017080, -0.0078189, -0.0017080, -0.0041489, 0.0041489
8: -0.0078635, -0.0031074, -0.0078635, -0.0031074, -0.0032291, 0.0032291
9: -0.0037416, -0.0033313, -0.0037416, -0.0033313, -0.0002786, 0.0002786

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006096, upper bound: 0.0006320
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006096, upper bound: 0.0006509
time: 1.30 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.10 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 5, lower bound: -0.0006164, upper bound: 0.0005965
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 5, lower bound: -0.0006163, upper bound: 0.0006097
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 5, lower bound: -0.0006164, upper bound: 0.0005964
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 5, lower bound: -0.0006163, upper bound: 0.0006096
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 5, lower bound: -0.0006096, upper bound: 0.0006320
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 5, lower bound: -0.0006096, upper bound: 0.0006510
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 5, lower bound: -0.0006096, upper bound: 0.0006320
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 5, lower bound: -0.0006096, upper bound: 0.0006509

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0088058, -0.0052482, -0.0089022, -0.0052131, -0.0025600, 0.0025797
1: -0.0054213, -0.0044183, -0.0054485, -0.0044084, -0.0007218, 0.0007273
2: -0.0014399, 0.0059604, -0.0016406, 0.0060335, -0.0053254, 0.0053663
3: 0.0014367, 0.0024161, 0.0014102, 0.0024257, -0.0007047, 0.0007101
4: 0.0016374, 0.0071680, 0.0015828, 0.0073179, -0.0040104, 0.0039798
5: 0.9959612, 0.9974978, 0.9959460, 0.9975394, -0.0011142, 0.0011057
6: 0.0042176, 0.0056123, 0.0042038, 0.0056501, -0.0010114, 0.0010037
7: -0.0076421, -0.0024373, -0.0076936, -0.0022961, -0.0037743, 0.0037455
8: -0.0072959, -0.0032450, -0.0074058, -0.0032049, -0.0029151, 0.0029375
9: -0.0037298, -0.0033803, -0.0037332, -0.0033708, -0.0002534, 0.0002515

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005958, upper bound: 0.0005799
time: 1.39 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005999, upper bound: 0.0005815
time: 1.41 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0089168, -0.0052119, -0.0089566, -0.0052084, -0.0025527, 0.0027177
1: -0.0054526, -0.0044081, -0.0054639, -0.0044071, -0.0007197, 0.0007662
2: -0.0016708, 0.0060359, -0.0017536, 0.0060433, -0.0053102, 0.0056533
3: 0.0014062, 0.0024261, 0.0013952, 0.0024270, -0.0007027, 0.0007481
4: 0.0015810, 0.0073405, 0.0015755, 0.0074024, -0.0042249, 0.0039685
5: 0.9959455, 0.9975457, 0.9959439, 0.9975629, -0.0011738, 0.0011026
6: 0.0042034, 0.0056558, 0.0042020, 0.0056715, -0.0010655, 0.0010008
7: -0.0076953, -0.0022749, -0.0077005, -0.0022166, -0.0039761, 0.0037348
8: -0.0074223, -0.0032036, -0.0074677, -0.0031996, -0.0029068, 0.0030946
9: -0.0037333, -0.0033694, -0.0037337, -0.0033655, -0.0002670, 0.0002508

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005958, upper bound: 0.0005988
time: 1.56 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006001, upper bound: 0.0006002
time: 1.38 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0088058, -0.0052482, -0.0092275, -0.0051335, -0.0027248, 0.0029962
1: -0.0054213, -0.0044183, -0.0055402, -0.0043860, -0.0007682, 0.0008448
2: -0.0014399, 0.0059604, -0.0023173, 0.0061991, -0.0056682, 0.0062328
3: 0.0014367, 0.0024161, 0.0013206, 0.0024477, -0.0007501, 0.0008248
4: 0.0016374, 0.0071680, 0.0014590, 0.0078236, -0.0046580, 0.0042360
5: 0.9959612, 0.9974978, 0.9959116, 0.9976799, -0.0012941, 0.0011769
6: 0.0042176, 0.0056123, 0.0041726, 0.0057777, -0.0011747, 0.0010683
7: -0.0076421, -0.0024373, -0.0078101, -0.0018202, -0.0043837, 0.0039866
8: -0.0072959, -0.0032450, -0.0077762, -0.0031143, -0.0031028, 0.0034118
9: -0.0037298, -0.0033803, -0.0037411, -0.0033388, -0.0002944, 0.0002677

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006293, upper bound: 0.0005766
time: 1.43 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006350, upper bound: 0.0005782
time: 1.30 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0089168, -0.0052119, -0.0092837, -0.0051293, -0.0027316, 0.0031349
1: -0.0054526, -0.0044081, -0.0055561, -0.0043848, -0.0007701, 0.0008838
2: -0.0016708, 0.0060359, -0.0024341, 0.0062079, -0.0056823, 0.0065212
3: 0.0014062, 0.0024261, 0.0013052, 0.0024488, -0.0007520, 0.0008630
4: 0.0015810, 0.0073405, 0.0014524, 0.0079110, -0.0048735, 0.0042466
5: 0.9959455, 0.9975457, 0.9959098, 0.9977041, -0.0013540, 0.0011798
6: 0.0042034, 0.0056558, 0.0041710, 0.0057997, -0.0012290, 0.0010709
7: -0.0076953, -0.0022749, -0.0078162, -0.0017380, -0.0045865, 0.0039965
8: -0.0074223, -0.0032036, -0.0078402, -0.0031095, -0.0031105, 0.0035697
9: -0.0037333, -0.0033694, -0.0037415, -0.0033333, -0.0003080, 0.0002684

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006293, upper bound: 0.0005916
time: 1.62 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006351, upper bound: 0.0005929
time: 1.46 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0091279, -0.0051725, -0.0089022, -0.0052131, -0.0029712, 0.0027613
1: -0.0055121, -0.0043970, -0.0054485, -0.0044084, -0.0008377, 0.0007785
2: -0.0021100, 0.0061179, -0.0016406, 0.0060335, -0.0061808, 0.0057440
3: 0.0013481, 0.0024369, 0.0014102, 0.0024257, -0.0008179, 0.0007601
4: 0.0015197, 0.0076687, 0.0015828, 0.0073179, -0.0042927, 0.0046191
5: 0.9959284, 0.9976369, 0.9959460, 0.9975394, -0.0011926, 0.0012833
6: 0.0041879, 0.0057386, 0.0042038, 0.0056501, -0.0010826, 0.0011649
7: -0.0077529, -0.0019660, -0.0076936, -0.0022961, -0.0040399, 0.0043471
8: -0.0076627, -0.0031588, -0.0074058, -0.0032049, -0.0033834, 0.0031443
9: -0.0037372, -0.0033486, -0.0037332, -0.0033708, -0.0002713, 0.0002919

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0006126
time: 1.49 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005929, upper bound: 0.0006142
time: 1.38 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0092445, -0.0051328, -0.0089566, -0.0052084, -0.0029667, 0.0028823
1: -0.0055450, -0.0043858, -0.0054639, -0.0044071, -0.0008364, 0.0008126
2: -0.0023527, 0.0062006, -0.0017536, 0.0060433, -0.0061714, 0.0059957
3: 0.0013160, 0.0024478, 0.0013952, 0.0024270, -0.0008167, 0.0007934
4: 0.0014579, 0.0078501, 0.0015755, 0.0074024, -0.0044808, 0.0046121
5: 0.9959113, 0.9976872, 0.9959439, 0.9975629, -0.0012449, 0.0012814
6: 0.0041723, 0.0057844, 0.0042020, 0.0056715, -0.0011300, 0.0011631
7: -0.0078111, -0.0017953, -0.0077005, -0.0022166, -0.0042169, 0.0043405
8: -0.0077956, -0.0031135, -0.0074677, -0.0031996, -0.0033782, 0.0032820
9: -0.0037411, -0.0033372, -0.0037337, -0.0033655, -0.0002832, 0.0002915

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0006337
time: 1.43 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005928, upper bound: 0.0006351
time: 1.29 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0091279, -0.0051725, -0.0092275, -0.0051335, -0.0026546, 0.0026767
1: -0.0055121, -0.0043970, -0.0055402, -0.0043860, -0.0007484, 0.0007547
2: -0.0021100, 0.0061179, -0.0023173, 0.0061991, -0.0055221, 0.0055681
3: 0.0013481, 0.0024369, 0.0013206, 0.0024477, -0.0007308, 0.0007368
4: 0.0015197, 0.0076687, 0.0014590, 0.0078236, -0.0041612, 0.0041269
5: 0.9959284, 0.9976369, 0.9959116, 0.9976799, -0.0011561, 0.0011466
6: 0.0041879, 0.0057386, 0.0041726, 0.0057777, -0.0010494, 0.0010407
7: -0.0077529, -0.0019660, -0.0078101, -0.0018202, -0.0039162, 0.0038839
8: -0.0076627, -0.0031588, -0.0077762, -0.0031143, -0.0030228, 0.0030480
9: -0.0037372, -0.0033486, -0.0037411, -0.0033388, -0.0002630, 0.0002608

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0006126
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005929, upper bound: 0.0006142
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0092445, -0.0051328, -0.0092837, -0.0051293, -0.0026458, 0.0028133
1: -0.0055450, -0.0043858, -0.0055561, -0.0043848, -0.0007460, 0.0007932
2: -0.0023527, 0.0062006, -0.0024341, 0.0062079, -0.0055039, 0.0058523
3: 0.0013160, 0.0024478, 0.0013052, 0.0024488, -0.0007283, 0.0007745
4: 0.0014579, 0.0078501, 0.0014524, 0.0079110, -0.0043736, 0.0041132
5: 0.9959113, 0.9976872, 0.9959098, 0.9977041, -0.0012151, 0.0011428
6: 0.0041723, 0.0057844, 0.0041710, 0.0057997, -0.0011030, 0.0010373
7: -0.0078111, -0.0017953, -0.0078162, -0.0017380, -0.0041161, 0.0038710
8: -0.0077956, -0.0031135, -0.0078402, -0.0031095, -0.0030128, 0.0032036
9: -0.0037411, -0.0033372, -0.0037415, -0.0033333, -0.0002764, 0.0002599

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0006337
time: 1.37 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005929, upper bound: 0.0006350
time: 1.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.59 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0005958, upper bound: 0.0005799
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0005999, upper bound: 0.0005815
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0005958, upper bound: 0.0005988
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0006001, upper bound: 0.0006002
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0006293, upper bound: 0.0005766
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0006350, upper bound: 0.0005782
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0006293, upper bound: 0.0005916
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0006351, upper bound: 0.0005929
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0006126
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0005929, upper bound: 0.0006142
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0006337
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0005928, upper bound: 0.0006351
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0006126
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0005929, upper bound: 0.0006142
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0006337
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 5, lower bound: -0.0005929, upper bound: 0.0006350

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0087473, -0.0052511, -0.0087577, -0.0051142, -0.0025334, 0.0023934
1: -0.0054049, -0.0044191, -0.0054078, -0.0043805, -0.0007143, 0.0006748
2: -0.0013183, 0.0059545, -0.0013398, 0.0062393, -0.0052701, 0.0049788
3: 0.0014528, 0.0024153, 0.0014500, 0.0024530, -0.0006974, 0.0006589
4: 0.0016418, 0.0070771, 0.0014290, 0.0070932, -0.0037209, 0.0039385
5: 0.9959624, 0.9974725, 0.9959033, 0.9974769, -0.0010338, 0.0010942
6: 0.0042187, 0.0055894, 0.0041650, 0.0055935, -0.0009383, 0.0009932
7: -0.0076380, -0.0025228, -0.0078383, -0.0025077, -0.0035018, 0.0037066
8: -0.0072294, -0.0032482, -0.0072411, -0.0030923, -0.0028848, 0.0027254
9: -0.0037295, -0.0033860, -0.0037429, -0.0033850, -0.0002351, 0.0002489

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005639, upper bound: 0.0005493
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005637, upper bound: 0.0005483
time: 1.57 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0088001, -0.0052484, -0.0088538, -0.0052147, -0.0025556, 0.0024398
1: -0.0054197, -0.0044184, -0.0054349, -0.0044089, -0.0007205, 0.0006879
2: -0.0014282, 0.0059600, -0.0015398, 0.0060303, -0.0053161, 0.0050752
3: 0.0014383, 0.0024160, 0.0014235, 0.0024253, -0.0007035, 0.0006716
4: 0.0016377, 0.0071592, 0.0015852, 0.0072426, -0.0037929, 0.0039729
5: 0.9959612, 0.9974953, 0.9959467, 0.9975184, -0.0010538, 0.0011038
6: 0.0042177, 0.0056101, 0.0042044, 0.0056311, -0.0009565, 0.0010019
7: -0.0076419, -0.0024455, -0.0076913, -0.0023670, -0.0035695, 0.0037390
8: -0.0072895, -0.0032452, -0.0073506, -0.0032067, -0.0029101, 0.0027782
9: -0.0037298, -0.0033808, -0.0037331, -0.0033756, -0.0002397, 0.0002511

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005733, upper bound: 0.0005541
time: 1.38 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005734, upper bound: 0.0005543
time: 1.37 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0088545, -0.0052146, -0.0088084, -0.0051092, -0.0025214, 0.0025266
1: -0.0054351, -0.0044088, -0.0054221, -0.0043791, -0.0007109, 0.0007123
2: -0.0015414, 0.0060304, -0.0014454, 0.0062498, -0.0052450, 0.0052558
3: 0.0014233, 0.0024253, 0.0014360, 0.0024544, -0.0006941, 0.0006955
4: 0.0015851, 0.0072438, 0.0014212, 0.0071721, -0.0039278, 0.0039198
5: 0.9959466, 0.9975188, 0.9959010, 0.9974989, -0.0010913, 0.0010890
6: 0.0042044, 0.0056314, 0.0041631, 0.0056134, -0.0009905, 0.0009885
7: -0.0076914, -0.0023659, -0.0078457, -0.0024334, -0.0036965, 0.0036890
8: -0.0073515, -0.0032066, -0.0072989, -0.0030866, -0.0028711, 0.0028770
9: -0.0037331, -0.0033755, -0.0037434, -0.0033800, -0.0002482, 0.0002477

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005639, upper bound: 0.0005658
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005637, upper bound: 0.0005677
time: 1.48 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0089115, -0.0052121, -0.0089102, -0.0052099, -0.0025484, 0.0025776
1: -0.0054511, -0.0044081, -0.0054508, -0.0044075, -0.0007185, 0.0007267
2: -0.0016599, 0.0060356, -0.0016572, 0.0060403, -0.0053011, 0.0053619
3: 0.0014076, 0.0024260, 0.0014080, 0.0024266, -0.0007015, 0.0007096
4: 0.0015812, 0.0073324, 0.0015777, 0.0073304, -0.0040072, 0.0039617
5: 0.9959456, 0.9975435, 0.9959446, 0.9975428, -0.0011133, 0.0011007
6: 0.0042034, 0.0056538, 0.0042026, 0.0056533, -0.0010105, 0.0009991
7: -0.0076950, -0.0022826, -0.0076983, -0.0022844, -0.0037712, 0.0037284
8: -0.0074163, -0.0032038, -0.0074149, -0.0032013, -0.0029018, 0.0029351
9: -0.0037333, -0.0033699, -0.0037335, -0.0033700, -0.0002532, 0.0002504

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005733, upper bound: 0.0005704
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005733, upper bound: 0.0005734
time: 1.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0087473, -0.0052511, -0.0090754, -0.0050379, -0.0027003, 0.0028110
1: -0.0054049, -0.0044191, -0.0054974, -0.0043590, -0.0007613, 0.0007925
2: -0.0013183, 0.0059545, -0.0020008, 0.0063980, -0.0056171, 0.0058474
3: 0.0014528, 0.0024153, 0.0013625, 0.0024740, -0.0007433, 0.0007738
4: 0.0016418, 0.0070771, 0.0013104, 0.0075872, -0.0043700, 0.0041979
5: 0.9959624, 0.9974725, 0.9958703, 0.9976141, -0.0012141, 0.0011663
6: 0.0042187, 0.0055894, 0.0041351, 0.0057180, -0.0011020, 0.0010586
7: -0.0076380, -0.0025228, -0.0079499, -0.0020428, -0.0041126, 0.0039507
8: -0.0072294, -0.0032482, -0.0076030, -0.0030054, -0.0030748, 0.0032009
9: -0.0037295, -0.0033860, -0.0037504, -0.0033538, -0.0002762, 0.0002653

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006004, upper bound: 0.0005476
time: 1.39 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006004, upper bound: 0.0005457
time: 1.42 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0088001, -0.0052484, -0.0091813, -0.0051349, -0.0027203, 0.0028604
1: -0.0054197, -0.0044184, -0.0055272, -0.0043864, -0.0007669, 0.0008065
2: -0.0014282, 0.0059600, -0.0022212, 0.0061961, -0.0056587, 0.0059502
3: 0.0014383, 0.0024160, 0.0013334, 0.0024473, -0.0007488, 0.0007874
4: 0.0016377, 0.0071592, 0.0014613, 0.0077518, -0.0044468, 0.0042290
5: 0.9959612, 0.9974953, 0.9959122, 0.9976599, -0.0012355, 0.0011749
6: 0.0042177, 0.0056101, 0.0041732, 0.0057596, -0.0011214, 0.0010665
7: -0.0076419, -0.0024455, -0.0078079, -0.0018878, -0.0041849, 0.0039799
8: -0.0072895, -0.0032452, -0.0077236, -0.0031159, -0.0030976, 0.0032572
9: -0.0037298, -0.0033808, -0.0037409, -0.0033434, -0.0002810, 0.0002672

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006103, upper bound: 0.0005520
time: 1.42 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006101, upper bound: 0.0005511
time: 1.35 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0088545, -0.0052146, -0.0091302, -0.0050335, -0.0027088, 0.0029471
1: -0.0054351, -0.0044088, -0.0055128, -0.0043578, -0.0007637, 0.0008309
2: -0.0015414, 0.0060304, -0.0021149, 0.0064072, -0.0056349, 0.0061305
3: 0.0014233, 0.0024253, 0.0013474, 0.0024752, -0.0007457, 0.0008113
4: 0.0015851, 0.0072438, 0.0013035, 0.0076724, -0.0045816, 0.0042112
5: 0.9959466, 0.9975188, 0.9958684, 0.9976379, -0.0012729, 0.0011700
6: 0.0042044, 0.0056314, 0.0041334, 0.0057395, -0.0011554, 0.0010620
7: -0.0076914, -0.0023659, -0.0079564, -0.0019626, -0.0043118, 0.0039632
8: -0.0073515, -0.0032066, -0.0076654, -0.0030004, -0.0030846, 0.0033558
9: -0.0037331, -0.0033755, -0.0037509, -0.0033484, -0.0002895, 0.0002661

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006004, upper bound: 0.0005615
time: 1.34 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006004, upper bound: 0.0005619
time: 1.42 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0089115, -0.0052121, -0.0092376, -0.0051307, -0.0027272, 0.0030000
1: -0.0054511, -0.0044081, -0.0055431, -0.0043852, -0.0007689, 0.0008458
2: -0.0016599, 0.0060356, -0.0023383, 0.0062050, -0.0056731, 0.0062407
3: 0.0014076, 0.0024260, 0.0013179, 0.0024484, -0.0007507, 0.0008259
4: 0.0015812, 0.0073324, 0.0014546, 0.0078394, -0.0046639, 0.0042397
5: 0.9959456, 0.9975435, 0.9959103, 0.9976842, -0.0012958, 0.0011779
6: 0.0042034, 0.0056538, 0.0041715, 0.0057816, -0.0011762, 0.0010692
7: -0.0076950, -0.0022826, -0.0078142, -0.0018054, -0.0043892, 0.0039900
8: -0.0074163, -0.0032038, -0.0077877, -0.0031111, -0.0031054, 0.0034162
9: -0.0037333, -0.0033699, -0.0037413, -0.0033378, -0.0002947, 0.0002679

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006101, upper bound: 0.0005655
time: 1.77 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006100, upper bound: 0.0005665
time: 1.27 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0090664, -0.0051751, -0.0087577, -0.0051142, -0.0029466, 0.0025750
1: -0.0054948, -0.0043977, -0.0054078, -0.0043805, -0.0008308, 0.0007260
2: -0.0019820, 0.0061125, -0.0013398, 0.0062393, -0.0061295, 0.0053565
3: 0.0013650, 0.0024362, 0.0014500, 0.0024530, -0.0008111, 0.0007088
4: 0.0015238, 0.0075731, 0.0014290, 0.0070932, -0.0040031, 0.0045808
5: 0.9959296, 0.9976103, 0.9959033, 0.9974769, -0.0011122, 0.0012727
6: 0.0041889, 0.0057145, 0.0041650, 0.0055935, -0.0010095, 0.0011552
7: -0.0077491, -0.0020560, -0.0078383, -0.0025077, -0.0037673, 0.0043111
8: -0.0075927, -0.0031617, -0.0072411, -0.0030923, -0.0033553, 0.0029321
9: -0.0037370, -0.0033547, -0.0037429, -0.0033850, -0.0002530, 0.0002895

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005592, upper bound: 0.0005787
time: 1.40 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005594, upper bound: 0.0005835
time: 1.47 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0091227, -0.0051727, -0.0088538, -0.0052147, -0.0029663, 0.0026321
1: -0.0055107, -0.0043970, -0.0054349, -0.0044089, -0.0008363, 0.0007421
2: -0.0020992, 0.0061175, -0.0015398, 0.0060303, -0.0061705, 0.0054752
3: 0.0013495, 0.0024369, 0.0014235, 0.0024253, -0.0008166, 0.0007246
4: 0.0015200, 0.0076606, 0.0015852, 0.0072426, -0.0040918, 0.0046114
5: 0.9959285, 0.9976346, 0.9959467, 0.9975184, -0.0011368, 0.0012812
6: 0.0041880, 0.0057366, 0.0042044, 0.0056311, -0.0010319, 0.0011629
7: -0.0077527, -0.0019736, -0.0076913, -0.0023670, -0.0038509, 0.0043399
8: -0.0076568, -0.0031590, -0.0073506, -0.0032067, -0.0033777, 0.0029971
9: -0.0037372, -0.0033491, -0.0037331, -0.0033756, -0.0002586, 0.0002914

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005665, upper bound: 0.0005839
time: 1.32 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005665, upper bound: 0.0005894
time: 1.48 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0091813, -0.0051353, -0.0088084, -0.0051092, -0.0029385, 0.0026910
1: -0.0055272, -0.0043865, -0.0054221, -0.0043791, -0.0008285, 0.0007587
2: -0.0022212, 0.0061954, -0.0014454, 0.0062498, -0.0061126, 0.0055979
3: 0.0013334, 0.0024472, 0.0014360, 0.0024544, -0.0008089, 0.0007408
4: 0.0014618, 0.0077518, 0.0014212, 0.0071721, -0.0041835, 0.0045682
5: 0.9959124, 0.9976599, 0.9959010, 0.9974989, -0.0011623, 0.0012692
6: 0.0041733, 0.0057596, 0.0041631, 0.0056134, -0.0010550, 0.0011520
7: -0.0078074, -0.0018878, -0.0078457, -0.0024334, -0.0039371, 0.0042992
8: -0.0077236, -0.0031164, -0.0072989, -0.0030866, -0.0033460, 0.0030643
9: -0.0037409, -0.0033434, -0.0037434, -0.0033800, -0.0002644, 0.0002887

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005594, upper bound: 0.0006006
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005593, upper bound: 0.0006057
time: 1.44 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0092394, -0.0051330, -0.0089102, -0.0052099, -0.0029618, 0.0027445
1: -0.0055436, -0.0043858, -0.0054508, -0.0044075, -0.0008350, 0.0007738
2: -0.0023419, 0.0062003, -0.0016572, 0.0060403, -0.0061612, 0.0057091
3: 0.0013174, 0.0024478, 0.0014080, 0.0024266, -0.0008153, 0.0007555
4: 0.0014582, 0.0078421, 0.0015777, 0.0073304, -0.0042666, 0.0046045
5: 0.9959114, 0.9976850, 0.9959446, 0.9975428, -0.0011854, 0.0012793
6: 0.0041724, 0.0057823, 0.0042026, 0.0056533, -0.0010760, 0.0011612
7: -0.0078108, -0.0018029, -0.0076983, -0.0022844, -0.0040153, 0.0043333
8: -0.0077897, -0.0031137, -0.0074149, -0.0032013, -0.0033726, 0.0031252
9: -0.0037411, -0.0033377, -0.0037335, -0.0033700, -0.0002696, 0.0002910

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005665, upper bound: 0.0006043
time: 1.58 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005664, upper bound: 0.0006103
time: 1.48 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0090664, -0.0051751, -0.0090754, -0.0050379, -0.0026285, 0.0024883
1: -0.0054948, -0.0043977, -0.0054974, -0.0043590, -0.0007411, 0.0007015
2: -0.0019820, 0.0061125, -0.0020008, 0.0063980, -0.0054679, 0.0051761
3: 0.0013650, 0.0024362, 0.0013625, 0.0024740, -0.0007236, 0.0006850
4: 0.0015238, 0.0075731, 0.0013104, 0.0075872, -0.0038683, 0.0040863
5: 0.9959296, 0.9976103, 0.9958703, 0.9976141, -0.0010747, 0.0011353
6: 0.0041889, 0.0057145, 0.0041351, 0.0057180, -0.0009755, 0.0010305
7: -0.0077491, -0.0020560, -0.0079499, -0.0020428, -0.0036405, 0.0038457
8: -0.0075927, -0.0031617, -0.0076030, -0.0030054, -0.0029931, 0.0028334
9: -0.0037370, -0.0033547, -0.0037504, -0.0033538, -0.0002445, 0.0002582

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005598, upper bound: 0.0005786
time: 1.44 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005599, upper bound: 0.0005834
time: 1.35 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0091227, -0.0051727, -0.0091813, -0.0051349, -0.0026501, 0.0025363
1: -0.0055107, -0.0043970, -0.0055272, -0.0043864, -0.0007472, 0.0007151
2: -0.0020992, 0.0061175, -0.0022212, 0.0061961, -0.0055127, 0.0052761
3: 0.0013495, 0.0024369, 0.0013334, 0.0024473, -0.0007295, 0.0006982
4: 0.0015200, 0.0076606, 0.0014613, 0.0077518, -0.0039430, 0.0041198
5: 0.9959285, 0.9976346, 0.9959122, 0.9976599, -0.0010955, 0.0011446
6: 0.0041880, 0.0057366, 0.0041732, 0.0057596, -0.0009944, 0.0010390
7: -0.0077527, -0.0019736, -0.0078079, -0.0018878, -0.0037108, 0.0038772
8: -0.0076568, -0.0031590, -0.0077236, -0.0031159, -0.0030176, 0.0028882
9: -0.0037372, -0.0033491, -0.0037409, -0.0033434, -0.0002492, 0.0002603

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005669, upper bound: 0.0005839
time: 1.44 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005667, upper bound: 0.0005894
time: 1.48 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0091813, -0.0051353, -0.0091302, -0.0050335, -0.0026142, 0.0026207
1: -0.0055272, -0.0043865, -0.0055128, -0.0043578, -0.0007371, 0.0007389
2: -0.0022212, 0.0061954, -0.0021149, 0.0064072, -0.0054381, 0.0054515
3: 0.0013334, 0.0024472, 0.0013474, 0.0024752, -0.0007197, 0.0007214
4: 0.0014618, 0.0077518, 0.0013035, 0.0076724, -0.0040741, 0.0040641
5: 0.9959124, 0.9976599, 0.9958684, 0.9976379, -0.0011319, 0.0011291
6: 0.0041733, 0.0057596, 0.0041334, 0.0057395, -0.0010274, 0.0010249
7: -0.0078074, -0.0018878, -0.0079564, -0.0019626, -0.0038342, 0.0038248
8: -0.0077236, -0.0031164, -0.0076654, -0.0030004, -0.0029768, 0.0029842
9: -0.0037409, -0.0033434, -0.0037509, -0.0033484, -0.0002575, 0.0002568

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005599, upper bound: 0.0006006
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005598, upper bound: 0.0006058
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0092394, -0.0051330, -0.0092376, -0.0051307, -0.0026413, 0.0026769
1: -0.0055436, -0.0043858, -0.0055431, -0.0043852, -0.0007447, 0.0007547
2: -0.0023419, 0.0062003, -0.0023383, 0.0062050, -0.0054944, 0.0055686
3: 0.0013174, 0.0024478, 0.0013179, 0.0024484, -0.0007271, 0.0007369
4: 0.0014582, 0.0078421, 0.0014546, 0.0078394, -0.0041616, 0.0041062
5: 0.9959114, 0.9976850, 0.9959103, 0.9976842, -0.0011562, 0.0011408
6: 0.0041724, 0.0057823, 0.0041715, 0.0057816, -0.0010495, 0.0010355
7: -0.0078108, -0.0018029, -0.0078142, -0.0018054, -0.0039165, 0.0038644
8: -0.0077897, -0.0031137, -0.0077877, -0.0031111, -0.0030076, 0.0030482
9: -0.0037411, -0.0033377, -0.0037413, -0.0033378, -0.0002630, 0.0002595

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005669, upper bound: 0.0006043
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005669, upper bound: 0.0006102
time: 1.54 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.57 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005639, upper bound: 0.0005493
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005637, upper bound: 0.0005483
NS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005733, upper bound: 0.0005541
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005734, upper bound: 0.0005543
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005639, upper bound: 0.0005658
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005637, upper bound: 0.0005677
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005733, upper bound: 0.0005704
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005733, upper bound: 0.0005734
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0006004, upper bound: 0.0005476
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0006004, upper bound: 0.0005457
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0006103, upper bound: 0.0005520
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0006101, upper bound: 0.0005511
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0006004, upper bound: 0.0005615
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0006004, upper bound: 0.0005619
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0006101, upper bound: 0.0005655
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0006100, upper bound: 0.0005665
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005592, upper bound: 0.0005787
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005594, upper bound: 0.0005835
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005665, upper bound: 0.0005839
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005665, upper bound: 0.0005894
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005594, upper bound: 0.0006006
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005593, upper bound: 0.0006057
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005665, upper bound: 0.0006043
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005664, upper bound: 0.0006103
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005598, upper bound: 0.0005786
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005599, upper bound: 0.0005834
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005669, upper bound: 0.0005839
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005667, upper bound: 0.0005894
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005599, upper bound: 0.0006006
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005598, upper bound: 0.0006058
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005669, upper bound: 0.0006043
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005669, upper bound: 0.0006102

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0086858, -0.0052608, -0.0090614, -0.0050401, -0.0026057, 0.0027511
1: -0.0053875, -0.0044219, -0.0054934, -0.0043597, -0.0007346, 0.0007756
2: -0.0011903, 0.0059344, -0.0019718, 0.0063934, -0.0054204, 0.0057229
3: 0.0014698, 0.0024126, 0.0013664, 0.0024734, -0.0007173, 0.0007573
4: 0.0016569, 0.0069814, 0.0013139, 0.0075654, -0.0042769, 0.0040509
5: 0.9959665, 0.9974459, 0.9958712, 0.9976081, -0.0011883, 0.0011255
6: 0.0042225, 0.0055653, 0.0041360, 0.0057126, -0.0010786, 0.0010216
7: -0.0076238, -0.0026128, -0.0079466, -0.0020632, -0.0040251, 0.0038123
8: -0.0071593, -0.0032592, -0.0075871, -0.0030080, -0.0029671, 0.0031327
9: -0.0037285, -0.0033921, -0.0037502, -0.0033552, -0.0002703, 0.0002560

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005874, upper bound: 0.0005360
time: 1.43 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005361
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0086072, -0.0051672, -0.0090159, -0.0050500, -0.0026370, 0.0028319
1: -0.0053654, -0.0043955, -0.0054806, -0.0043624, -0.0007435, 0.0007984
2: -0.0010269, 0.0061290, -0.0018770, 0.0063729, -0.0054855, 0.0058909
3: 0.0014914, 0.0024384, 0.0013789, 0.0024706, -0.0007259, 0.0007796
4: 0.0015115, 0.0068593, 0.0013292, 0.0074946, -0.0044025, 0.0040995
5: 0.9959261, 0.9974119, 0.9958755, 0.9975885, -0.0012231, 0.0011390
6: 0.0041858, 0.0055345, 0.0041399, 0.0056947, -0.0011102, 0.0010338
7: -0.0077607, -0.0027278, -0.0079322, -0.0021299, -0.0041432, 0.0038581
8: -0.0070698, -0.0031527, -0.0075352, -0.0030192, -0.0030028, 0.0032247
9: -0.0037377, -0.0033998, -0.0037493, -0.0033596, -0.0002782, 0.0002591

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005342
time: 1.62 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0005342
time: 1.40 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0087390, -0.0052583, -0.0091674, -0.0051372, -0.0026277, 0.0027995
1: -0.0054025, -0.0044212, -0.0055233, -0.0043870, -0.0007408, 0.0007893
2: -0.0013012, 0.0059395, -0.0021923, 0.0061914, -0.0054661, 0.0058236
3: 0.0014551, 0.0024133, 0.0013372, 0.0024466, -0.0007234, 0.0007707
4: 0.0016531, 0.0070643, 0.0014648, 0.0077302, -0.0043522, 0.0040851
5: 0.9959655, 0.9974689, 0.9959132, 0.9976540, -0.0012092, 0.0011350
6: 0.0042215, 0.0055862, 0.0041741, 0.0057541, -0.0010976, 0.0010302
7: -0.0076274, -0.0025349, -0.0078046, -0.0019081, -0.0040959, 0.0038445
8: -0.0072200, -0.0032564, -0.0077078, -0.0031185, -0.0029922, 0.0031879
9: -0.0037288, -0.0033868, -0.0037407, -0.0033447, -0.0002750, 0.0002582

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005973, upper bound: 0.0005399
time: 1.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005400
time: 1.39 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0086623, -0.0051650, -0.0091230, -0.0051476, -0.0026578, 0.0028900
1: -0.0053809, -0.0043949, -0.0055108, -0.0043899, -0.0007493, 0.0008148
2: -0.0011415, 0.0061335, -0.0020998, 0.0061699, -0.0055289, 0.0060117
3: 0.0014762, 0.0024390, 0.0013494, 0.0024438, -0.0007317, 0.0007956
4: 0.0015081, 0.0069449, 0.0014809, 0.0076611, -0.0044928, 0.0041319
5: 0.9959252, 0.9974357, 0.9959177, 0.9976348, -0.0012482, 0.0011480
6: 0.0041850, 0.0055561, 0.0041781, 0.0057367, -0.0011330, 0.0010420
7: -0.0077639, -0.0026472, -0.0077895, -0.0019732, -0.0042282, 0.0038886
8: -0.0071325, -0.0031502, -0.0076571, -0.0031303, -0.0030265, 0.0032908
9: -0.0037379, -0.0033944, -0.0037397, -0.0033491, -0.0002839, 0.0002611

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005973, upper bound: 0.0005391
time: 1.36 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005390
time: 1.42 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0087926, -0.0052253, -0.0091163, -0.0050357, -0.0026155, 0.0028864
1: -0.0054176, -0.0044119, -0.0055089, -0.0043584, -0.0007374, 0.0008138
2: -0.0014125, 0.0060081, -0.0020860, 0.0064025, -0.0054408, 0.0060043
3: 0.0014404, 0.0024224, 0.0013513, 0.0024746, -0.0007200, 0.0007946
4: 0.0016018, 0.0071475, 0.0013071, 0.0076508, -0.0044872, 0.0040661
5: 0.9959512, 0.9974920, 0.9958693, 0.9976318, -0.0012467, 0.0011297
6: 0.0042086, 0.0056072, 0.0041343, 0.0057341, -0.0011316, 0.0010254
7: -0.0076757, -0.0024566, -0.0079531, -0.0019829, -0.0042230, 0.0038267
8: -0.0072809, -0.0032189, -0.0076496, -0.0030030, -0.0029783, 0.0032868
9: -0.0037320, -0.0033816, -0.0037507, -0.0033498, -0.0002836, 0.0002570

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005511
time: 1.43 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005511
time: 1.33 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0087225, -0.0051307, -0.0090704, -0.0050459, -0.0026462, 0.0029628
1: -0.0053979, -0.0043852, -0.0054960, -0.0043613, -0.0007460, 0.0008353
2: -0.0012667, 0.0062050, -0.0019905, 0.0063814, -0.0055045, 0.0061632
3: 0.0014597, 0.0024484, 0.0013639, 0.0024718, -0.0007284, 0.0008156
4: 0.0014546, 0.0070385, 0.0013228, 0.0075794, -0.0046060, 0.0041137
5: 0.9959103, 0.9974618, 0.9958738, 0.9976121, -0.0012797, 0.0011429
6: 0.0041715, 0.0055797, 0.0041383, 0.0057161, -0.0011616, 0.0010374
7: -0.0078142, -0.0025591, -0.0079383, -0.0020501, -0.0043348, 0.0038715
8: -0.0072011, -0.0031111, -0.0075973, -0.0030145, -0.0030132, 0.0033738
9: -0.0037413, -0.0033885, -0.0037497, -0.0033543, -0.0002911, 0.0002600

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005874, upper bound: 0.0005512
time: 1.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0005510
time: 1.60 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0088484, -0.0052230, -0.0092236, -0.0051330, -0.0026351, 0.0029399
1: -0.0054334, -0.0044112, -0.0055391, -0.0043858, -0.0007429, 0.0008289
2: -0.0015287, 0.0060129, -0.0023092, 0.0062001, -0.0054815, 0.0061155
3: 0.0014250, 0.0024230, 0.0013217, 0.0024478, -0.0007254, 0.0008093
4: 0.0015982, 0.0072343, 0.0014583, 0.0078176, -0.0045704, 0.0040965
5: 0.9959503, 0.9975162, 0.9959114, 0.9976782, -0.0012698, 0.0011381
6: 0.0042077, 0.0056291, 0.0041724, 0.0057762, -0.0011526, 0.0010331
7: -0.0076790, -0.0023749, -0.0078107, -0.0018259, -0.0043012, 0.0038553
8: -0.0073445, -0.0032163, -0.0077718, -0.0031138, -0.0030006, 0.0033477
9: -0.0037323, -0.0033761, -0.0037411, -0.0033392, -0.0002888, 0.0002589

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005973, upper bound: 0.0005546
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005545
time: 1.55 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0087793, -0.0051286, -0.0091794, -0.0051436, -0.0026633, 0.0030204
1: -0.0054139, -0.0043846, -0.0055267, -0.0043888, -0.0007509, 0.0008516
2: -0.0013849, 0.0062094, -0.0022172, 0.0061781, -0.0055402, 0.0062830
3: 0.0014440, 0.0024490, 0.0013339, 0.0024449, -0.0007332, 0.0008315
4: 0.0014513, 0.0071269, 0.0014747, 0.0077489, -0.0046955, 0.0041404
5: 0.9959095, 0.9974862, 0.9959160, 0.9976591, -0.0013046, 0.0011503
6: 0.0041707, 0.0056020, 0.0041766, 0.0057588, -0.0011841, 0.0010441
7: -0.0078173, -0.0024760, -0.0077952, -0.0018906, -0.0044190, 0.0038966
8: -0.0072658, -0.0031087, -0.0077214, -0.0031258, -0.0030327, 0.0034393
9: -0.0037415, -0.0033829, -0.0037401, -0.0033436, -0.0002967, 0.0002616

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005973, upper bound: 0.0005553
time: 1.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005554
time: 1.35 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0089932, -0.0050869, -0.0087946, -0.0052260, -0.0028808, 0.0026781
1: -0.0054742, -0.0043728, -0.0054182, -0.0044121, -0.0008122, 0.0007551
2: -0.0018298, 0.0062961, -0.0014167, 0.0060068, -0.0059927, 0.0055710
3: 0.0013851, 0.0024605, 0.0014398, 0.0024222, -0.0007930, 0.0007372
4: 0.0013866, 0.0074594, 0.0016028, 0.0071507, -0.0041634, 0.0044785
5: 0.9958914, 0.9975787, 0.9959515, 0.9974929, -0.0011567, 0.0012443
6: 0.0041543, 0.0056858, 0.0042089, 0.0056080, -0.0010500, 0.0011294
7: -0.0078782, -0.0021630, -0.0076747, -0.0024536, -0.0039182, 0.0042148
8: -0.0075094, -0.0030612, -0.0072832, -0.0032196, -0.0032804, 0.0030496
9: -0.0037456, -0.0033619, -0.0037320, -0.0033814, -0.0002631, 0.0002830

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005532, upper bound: 0.0005766
time: 1.56 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005552, upper bound: 0.0005767
time: 1.44 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0091166, -0.0051460, -0.0087949, -0.0051114, -0.0028352, 0.0026403
1: -0.0055090, -0.0043895, -0.0054183, -0.0043798, -0.0007993, 0.0007444
2: -0.0020865, 0.0061731, -0.0014173, 0.0062450, -0.0058978, 0.0054923
3: 0.0013512, 0.0024442, 0.0014397, 0.0024537, -0.0007805, 0.0007268
4: 0.0014785, 0.0076512, 0.0014247, 0.0071511, -0.0041046, 0.0044076
5: 0.9959171, 0.9976320, 0.9959021, 0.9974930, -0.0011404, 0.0012246
6: 0.0041775, 0.0057342, 0.0041640, 0.0056081, -0.0010351, 0.0011115
7: -0.0077917, -0.0019825, -0.0078423, -0.0024532, -0.0038629, 0.0041481
8: -0.0076499, -0.0031286, -0.0072836, -0.0030892, -0.0032285, 0.0030065
9: -0.0037398, -0.0033497, -0.0037432, -0.0033813, -0.0002594, 0.0002785

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005463, upper bound: 0.0005895
time: 1.40 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005484, upper bound: 0.0005894
time: 1.49 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0090503, -0.0050511, -0.0087467, -0.0051201, -0.0028561, 0.0027276
1: -0.0054903, -0.0043628, -0.0054047, -0.0043822, -0.0008052, 0.0007690
2: -0.0019487, 0.0063705, -0.0013170, 0.0062269, -0.0059413, 0.0056739
3: 0.0013694, 0.0024703, 0.0014530, 0.0024513, -0.0007862, 0.0007509
4: 0.0013309, 0.0075482, 0.0014383, 0.0070761, -0.0042403, 0.0044401
5: 0.9958761, 0.9976034, 0.9959058, 0.9974721, -0.0011781, 0.0012336
6: 0.0041403, 0.0057082, 0.0041674, 0.0055892, -0.0010694, 0.0011197
7: -0.0079306, -0.0020794, -0.0078296, -0.0025238, -0.0039906, 0.0041787
8: -0.0075744, -0.0030205, -0.0072286, -0.0030991, -0.0032523, 0.0031059
9: -0.0037491, -0.0033562, -0.0037424, -0.0033861, -0.0002680, 0.0002806

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005464, upper bound: 0.0005944
time: 1.46 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005485, upper bound: 0.0005944
time: 1.46 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0091746, -0.0051438, -0.0088964, -0.0052122, -0.0028587, 0.0026921
1: -0.0055253, -0.0043889, -0.0054469, -0.0044082, -0.0008060, 0.0007590
2: -0.0022072, 0.0061776, -0.0016286, 0.0060354, -0.0059467, 0.0056001
3: 0.0013352, 0.0024448, 0.0014118, 0.0024260, -0.0007869, 0.0007411
4: 0.0014751, 0.0077414, 0.0015814, 0.0073090, -0.0041851, 0.0044442
5: 0.9959161, 0.9976570, 0.9959456, 0.9975368, -0.0011628, 0.0012347
6: 0.0041767, 0.0057569, 0.0042035, 0.0056479, -0.0010554, 0.0011208
7: -0.0077949, -0.0018977, -0.0076949, -0.0023046, -0.0039387, 0.0041825
8: -0.0077159, -0.0031261, -0.0073992, -0.0032039, -0.0032552, 0.0030655
9: -0.0037400, -0.0033440, -0.0037333, -0.0033714, -0.0002645, 0.0002808

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005532, upper bound: 0.0005930
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005552, upper bound: 0.0005930
time: 1.48 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0091087, -0.0050491, -0.0088512, -0.0052215, -0.0028770, 0.0027798
1: -0.0055067, -0.0043622, -0.0054341, -0.0044108, -0.0008111, 0.0007837
2: -0.0020701, 0.0063746, -0.0015344, 0.0060160, -0.0059848, 0.0057825
3: 0.0013533, 0.0024709, 0.0014242, 0.0024234, -0.0007920, 0.0007652
4: 0.0013279, 0.0076390, 0.0015958, 0.0072386, -0.0043215, 0.0044726
5: 0.9958752, 0.9976285, 0.9959496, 0.9975173, -0.0012006, 0.0012426
6: 0.0041395, 0.0057311, 0.0042071, 0.0056301, -0.0010898, 0.0011279
7: -0.0079334, -0.0019940, -0.0076813, -0.0023708, -0.0040670, 0.0042093
8: -0.0076409, -0.0030183, -0.0073477, -0.0032145, -0.0032761, 0.0031653
9: -0.0037493, -0.0033505, -0.0037324, -0.0033758, -0.0002731, 0.0002826

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005531, upper bound: 0.0005984
time: 1.41 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005552, upper bound: 0.0005984
time: 1.48 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0089932, -0.0050869, -0.0091230, -0.0051476, -0.0025959, 0.0025699
1: -0.0054742, -0.0043728, -0.0055108, -0.0043899, -0.0007319, 0.0007245
2: -0.0018298, 0.0062961, -0.0020998, 0.0061699, -0.0054000, 0.0053458
3: 0.0013851, 0.0024605, 0.0013494, 0.0024438, -0.0007146, 0.0007074
4: 0.0013866, 0.0074594, 0.0014809, 0.0076611, -0.0039951, 0.0040356
5: 0.9958914, 0.9975787, 0.9959177, 0.9976348, -0.0011100, 0.0011212
6: 0.0041543, 0.0056858, 0.0041781, 0.0057367, -0.0010075, 0.0010177
7: -0.0078782, -0.0021630, -0.0077895, -0.0019732, -0.0037599, 0.0037980
8: -0.0075094, -0.0030612, -0.0076571, -0.0031303, -0.0029560, 0.0029263
9: -0.0037456, -0.0033619, -0.0037397, -0.0033491, -0.0002525, 0.0002550

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005535, upper bound: 0.0005766
time: 1.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005556, upper bound: 0.0005766
time: 1.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0091166, -0.0051460, -0.0091163, -0.0050357, -0.0025189, 0.0025630
1: -0.0055090, -0.0043895, -0.0055089, -0.0043584, -0.0007102, 0.0007226
2: -0.0020865, 0.0061731, -0.0020860, 0.0064025, -0.0052398, 0.0053315
3: 0.0013512, 0.0024442, 0.0013513, 0.0024746, -0.0006934, 0.0007055
4: 0.0014785, 0.0076512, 0.0013071, 0.0076508, -0.0039844, 0.0039159
5: 0.9959171, 0.9976320, 0.9958693, 0.9976318, -0.0011070, 0.0010879
6: 0.0041775, 0.0057342, 0.0041343, 0.0057341, -0.0010048, 0.0009875
7: -0.0077917, -0.0019825, -0.0079531, -0.0019829, -0.0037498, 0.0036853
8: -0.0076499, -0.0031286, -0.0076496, -0.0030030, -0.0028683, 0.0029184
9: -0.0037398, -0.0033497, -0.0037507, -0.0033498, -0.0002518, 0.0002475

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005468, upper bound: 0.0005895
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005895
time: 1.56 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0090503, -0.0050511, -0.0090704, -0.0050459, -0.0025603, 0.0026405
1: -0.0054903, -0.0043628, -0.0054960, -0.0043613, -0.0007219, 0.0007445
2: -0.0019487, 0.0063705, -0.0019905, 0.0063814, -0.0053260, 0.0054928
3: 0.0013694, 0.0024703, 0.0013639, 0.0024718, -0.0007048, 0.0007269
4: 0.0013309, 0.0075482, 0.0013228, 0.0075794, -0.0041050, 0.0039803
5: 0.9958761, 0.9976034, 0.9958738, 0.9976121, -0.0011405, 0.0011059
6: 0.0041403, 0.0057082, 0.0041383, 0.0057161, -0.0010352, 0.0010038
7: -0.0079306, -0.0020794, -0.0079383, -0.0020501, -0.0038633, 0.0037459
8: -0.0075744, -0.0030205, -0.0075973, -0.0030145, -0.0029155, 0.0030068
9: -0.0037491, -0.0033562, -0.0037497, -0.0033543, -0.0002594, 0.0002515

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005468, upper bound: 0.0005944
time: 2.04 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005944
time: 1.41 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0091746, -0.0051438, -0.0092236, -0.0051330, -0.0025472, 0.0026195
1: -0.0055253, -0.0043889, -0.0055391, -0.0043858, -0.0007182, 0.0007385
2: -0.0022072, 0.0061776, -0.0023092, 0.0062001, -0.0052987, 0.0054491
3: 0.0013352, 0.0024448, 0.0013217, 0.0024478, -0.0007012, 0.0007211
4: 0.0014751, 0.0077414, 0.0014583, 0.0078176, -0.0040723, 0.0039600
5: 0.9959161, 0.9976570, 0.9959114, 0.9976782, -0.0011314, 0.0011002
6: 0.0041767, 0.0057569, 0.0041724, 0.0057762, -0.0010270, 0.0009986
7: -0.0077949, -0.0018977, -0.0078107, -0.0018259, -0.0038325, 0.0037268
8: -0.0077159, -0.0031261, -0.0077718, -0.0031138, -0.0029005, 0.0029828
9: -0.0037400, -0.0033440, -0.0037411, -0.0033392, -0.0002573, 0.0002502

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005532, upper bound: 0.0005930
time: 1.39 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005554, upper bound: 0.0005930
time: 1.34 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0091087, -0.0050491, -0.0091794, -0.0051436, -0.0025897, 0.0027009
1: -0.0055067, -0.0043622, -0.0055267, -0.0043888, -0.0007301, 0.0007615
2: -0.0020701, 0.0063746, -0.0022172, 0.0061781, -0.0053870, 0.0056184
3: 0.0013533, 0.0024709, 0.0013339, 0.0024449, -0.0007129, 0.0007435
4: 0.0013279, 0.0076390, 0.0014747, 0.0077489, -0.0041988, 0.0040259
5: 0.9958752, 0.9976285, 0.9959160, 0.9976591, -0.0011666, 0.0011185
6: 0.0041395, 0.0057311, 0.0041766, 0.0057588, -0.0010589, 0.0010153
7: -0.0079334, -0.0019940, -0.0077952, -0.0018906, -0.0039516, 0.0037888
8: -0.0076409, -0.0030183, -0.0077214, -0.0031258, -0.0029489, 0.0030755
9: -0.0037493, -0.0033505, -0.0037401, -0.0033436, -0.0002653, 0.0002544

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005532, upper bound: 0.0005984
time: 1.56 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005554, upper bound: 0.0005984
time: 1.38 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.58 seconds
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005874, upper bound: 0.0005360
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005361
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005342
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0005342
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005973, upper bound: 0.0005399
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005400
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005973, upper bound: 0.0005391
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005390
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005511
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005511
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005874, upper bound: 0.0005512
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0005510
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005973, upper bound: 0.0005546
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005545
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005973, upper bound: 0.0005553
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005554
NS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005532, upper bound: 0.0005766
NS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005552, upper bound: 0.0005767
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005463, upper bound: 0.0005895
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005484, upper bound: 0.0005894
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005464, upper bound: 0.0005944
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005485, upper bound: 0.0005944
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005532, upper bound: 0.0005930
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005552, upper bound: 0.0005930
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005531, upper bound: 0.0005984
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005552, upper bound: 0.0005984
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005535, upper bound: 0.0005766
NS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005556, upper bound: 0.0005766
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005468, upper bound: 0.0005895
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005895
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005468, upper bound: 0.0005944
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005944
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005532, upper bound: 0.0005930
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005554, upper bound: 0.0005930
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005532, upper bound: 0.0005984
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 5, lower bound: -0.0005554, upper bound: 0.0005984

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0086542, -0.0052664, -0.0089841, -0.0050703, -0.0025366, 0.0026746
1: -0.0053786, -0.0044234, -0.0054716, -0.0043682, -0.0007152, 0.0007541
2: -0.0011246, 0.0059227, -0.0018110, 0.0063306, -0.0052766, 0.0055637
3: 0.0014785, 0.0024111, 0.0013876, 0.0024651, -0.0006983, 0.0007363
4: 0.0016656, 0.0069323, 0.0013607, 0.0074453, -0.0041579, 0.0039434
5: 0.9959690, 0.9974323, 0.9958843, 0.9975747, -0.0011552, 0.0010956
6: 0.0042247, 0.0055529, 0.0041478, 0.0056823, -0.0010486, 0.0009945
7: -0.0076156, -0.0026590, -0.0079025, -0.0021763, -0.0039131, 0.0037112
8: -0.0071233, -0.0032656, -0.0074990, -0.0030423, -0.0028884, 0.0030456
9: -0.0037280, -0.0033952, -0.0037473, -0.0033628, -0.0002628, 0.0002492

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005745, upper bound: 0.0005113
time: 1.42 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005687, upper bound: 0.0005163
time: 1.53 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0086574, -0.0052650, -0.0089983, -0.0050498, -0.0025689, 0.0026730
1: -0.0053795, -0.0044231, -0.0054756, -0.0043624, -0.0007243, 0.0007536
2: -0.0011313, 0.0059255, -0.0018404, 0.0063732, -0.0053438, 0.0055604
3: 0.0014776, 0.0024114, 0.0013837, 0.0024707, -0.0007072, 0.0007358
4: 0.0016635, 0.0069373, 0.0013289, 0.0074673, -0.0041555, 0.0039936
5: 0.9959684, 0.9974337, 0.9958754, 0.9975809, -0.0011545, 0.0011095
6: 0.0042242, 0.0055542, 0.0041398, 0.0056878, -0.0010480, 0.0010071
7: -0.0076176, -0.0026544, -0.0079325, -0.0021556, -0.0039108, 0.0037584
8: -0.0071270, -0.0032641, -0.0075151, -0.0030190, -0.0029252, 0.0030438
9: -0.0037281, -0.0033949, -0.0037493, -0.0033614, -0.0002626, 0.0002524

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005769, upper bound: 0.0005113
time: 1.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005711, upper bound: 0.0005162
time: 1.50 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0085752, -0.0051725, -0.0089393, -0.0050801, -0.0025665, 0.0027569
1: -0.0053563, -0.0043970, -0.0054590, -0.0043709, -0.0007236, 0.0007773
2: -0.0009604, 0.0061181, -0.0017177, 0.0063101, -0.0053389, 0.0057350
3: 0.0015002, 0.0024369, 0.0014000, 0.0024623, -0.0007065, 0.0007589
4: 0.0015196, 0.0068096, 0.0013761, 0.0073755, -0.0042860, 0.0039900
5: 0.9959285, 0.9973981, 0.9958885, 0.9975553, -0.0011908, 0.0011085
6: 0.0041879, 0.0055219, 0.0041517, 0.0056647, -0.0010809, 0.0010062
7: -0.0077530, -0.0027746, -0.0078881, -0.0022419, -0.0040336, 0.0037550
8: -0.0070334, -0.0031587, -0.0074480, -0.0030536, -0.0029225, 0.0031393
9: -0.0037372, -0.0034029, -0.0037463, -0.0033672, -0.0002708, 0.0002521

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005746, upper bound: 0.0005093
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005680, upper bound: 0.0005137
time: 1.60 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0085792, -0.0051713, -0.0089518, -0.0050595, -0.0026002, 0.0027545
1: -0.0053575, -0.0043966, -0.0054625, -0.0043651, -0.0007331, 0.0007766
2: -0.0009686, 0.0061205, -0.0017438, 0.0063531, -0.0054090, 0.0057299
3: 0.0014991, 0.0024373, 0.0013965, 0.0024680, -0.0007158, 0.0007583
4: 0.0015178, 0.0068157, 0.0013439, 0.0073951, -0.0042822, 0.0040423
5: 0.9959279, 0.9973999, 0.9958797, 0.9975608, -0.0011897, 0.0011231
6: 0.0041874, 0.0055235, 0.0041436, 0.0056696, -0.0010799, 0.0010194
7: -0.0077548, -0.0027688, -0.0079184, -0.0022236, -0.0040300, 0.0038043
8: -0.0070379, -0.0031573, -0.0074623, -0.0030300, -0.0029609, 0.0031366
9: -0.0037373, -0.0034025, -0.0037483, -0.0033659, -0.0002706, 0.0002554

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005769, upper bound: 0.0005093
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005708, upper bound: 0.0005137
time: 1.41 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0087077, -0.0052639, -0.0090923, -0.0051636, -0.0025628, 0.0027235
1: -0.0053937, -0.0044227, -0.0055021, -0.0043945, -0.0007225, 0.0007679
2: -0.0012358, 0.0059279, -0.0020359, 0.0061366, -0.0053311, 0.0056654
3: 0.0014638, 0.0024118, 0.0013579, 0.0024394, -0.0007055, 0.0007497
4: 0.0016617, 0.0070155, 0.0015058, 0.0076134, -0.0042340, 0.0039841
5: 0.9959679, 0.9974553, 0.9959246, 0.9976214, -0.0011763, 0.0011069
6: 0.0042237, 0.0055739, 0.0041844, 0.0057247, -0.0010677, 0.0010047
7: -0.0076193, -0.0025808, -0.0077661, -0.0020181, -0.0039846, 0.0037495
8: -0.0071842, -0.0032628, -0.0076222, -0.0031485, -0.0029183, 0.0031013
9: -0.0037282, -0.0033899, -0.0037381, -0.0033521, -0.0002676, 0.0002518

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005355
time: 1.88 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005400
time: 1.89 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0087104, -0.0052625, -0.0091031, -0.0051468, -0.0025908, 0.0027108
1: -0.0053945, -0.0044224, -0.0055052, -0.0043897, -0.0007305, 0.0007643
2: -0.0012416, 0.0059307, -0.0020585, 0.0061714, -0.0053895, 0.0056391
3: 0.0014630, 0.0024121, 0.0013549, 0.0024440, -0.0007132, 0.0007462
4: 0.0016596, 0.0070198, 0.0014797, 0.0076302, -0.0042143, 0.0040278
5: 0.9959673, 0.9974565, 0.9959173, 0.9976262, -0.0011709, 0.0011190
6: 0.0042232, 0.0055749, 0.0041778, 0.0057289, -0.0010628, 0.0010157
7: -0.0076212, -0.0025768, -0.0077905, -0.0020022, -0.0039661, 0.0037906
8: -0.0071874, -0.0032613, -0.0076345, -0.0031295, -0.0029502, 0.0030868
9: -0.0037284, -0.0033896, -0.0037397, -0.0033511, -0.0002663, 0.0002545

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005354
time: 1.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005399
time: 1.99 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0086306, -0.0051703, -0.0090489, -0.0051738, -0.0025914, 0.0028155
1: -0.0053719, -0.0043964, -0.0054899, -0.0043973, -0.0007306, 0.0007938
2: -0.0010755, 0.0061226, -0.0019457, 0.0061153, -0.0053906, 0.0058569
3: 0.0014850, 0.0024375, 0.0013698, 0.0024366, -0.0007134, 0.0007751
4: 0.0015162, 0.0068957, 0.0015217, 0.0075459, -0.0043771, 0.0040286
5: 0.9959275, 0.9974220, 0.9959290, 0.9976027, -0.0012161, 0.0011193
6: 0.0041870, 0.0055437, 0.0041884, 0.0057076, -0.0011038, 0.0010160
7: -0.0077562, -0.0026936, -0.0077511, -0.0020816, -0.0041193, 0.0037914
8: -0.0070965, -0.0031562, -0.0075728, -0.0031602, -0.0029508, 0.0032061
9: -0.0037374, -0.0033975, -0.0037371, -0.0033564, -0.0002766, 0.0002546

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005340
time: 1.96 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005390
time: 1.93 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0086349, -0.0051691, -0.0090574, -0.0051569, -0.0026209, 0.0028037
1: -0.0053732, -0.0043960, -0.0054923, -0.0043926, -0.0007389, 0.0007905
2: -0.0010845, 0.0061250, -0.0019634, 0.0061504, -0.0054519, 0.0058322
3: 0.0014838, 0.0024378, 0.0013675, 0.0024412, -0.0007215, 0.0007718
4: 0.0015144, 0.0069023, 0.0014954, 0.0075592, -0.0043586, 0.0040744
5: 0.9959270, 0.9974239, 0.9959217, 0.9976064, -0.0012110, 0.0011320
6: 0.0041866, 0.0055453, 0.0041818, 0.0057110, -0.0010992, 0.0010275
7: -0.0077579, -0.0026873, -0.0077758, -0.0020691, -0.0041020, 0.0038345
8: -0.0071014, -0.0031549, -0.0075825, -0.0031410, -0.0029844, 0.0031926
9: -0.0037375, -0.0033971, -0.0037387, -0.0033556, -0.0002754, 0.0002575

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005340
time: 1.95 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005390
time: 2.02 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0087602, -0.0052308, -0.0090380, -0.0050657, -0.0025505, 0.0028084
1: -0.0054085, -0.0044134, -0.0054868, -0.0043669, -0.0007191, 0.0007918
2: -0.0013451, 0.0059967, -0.0019231, 0.0063401, -0.0053056, 0.0058419
3: 0.0014493, 0.0024209, 0.0013728, 0.0024663, -0.0007021, 0.0007731
4: 0.0016103, 0.0070971, 0.0013537, 0.0075291, -0.0043659, 0.0039651
5: 0.9959536, 0.9974781, 0.9958823, 0.9975980, -0.0012130, 0.0011016
6: 0.0042108, 0.0055945, 0.0041461, 0.0057034, -0.0011010, 0.0009999
7: -0.0076677, -0.0025040, -0.0079092, -0.0020974, -0.0041088, 0.0037316
8: -0.0072440, -0.0032251, -0.0075604, -0.0030372, -0.0029043, 0.0031979
9: -0.0037315, -0.0033848, -0.0037477, -0.0033575, -0.0002759, 0.0002506

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005746, upper bound: 0.0005313
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005687, upper bound: 0.0005331
time: 1.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0087656, -0.0052294, -0.0090530, -0.0050451, -0.0025788, 0.0027950
1: -0.0054100, -0.0044130, -0.0054910, -0.0043611, -0.0007271, 0.0007880
2: -0.0013563, 0.0059997, -0.0019543, 0.0063830, -0.0053644, 0.0058143
3: 0.0014478, 0.0024213, 0.0013687, 0.0024720, -0.0007099, 0.0007694
4: 0.0016081, 0.0071055, 0.0013216, 0.0075524, -0.0043452, 0.0040090
5: 0.9959530, 0.9974805, 0.9958734, 0.9976045, -0.0012072, 0.0011138
6: 0.0042102, 0.0055966, 0.0041380, 0.0057093, -0.0010958, 0.0010110
7: -0.0076697, -0.0024961, -0.0079393, -0.0020755, -0.0040893, 0.0037729
8: -0.0072501, -0.0032235, -0.0075775, -0.0030137, -0.0029365, 0.0031827
9: -0.0037316, -0.0033842, -0.0037497, -0.0033560, -0.0002746, 0.0002533

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005769, upper bound: 0.0005313
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005711, upper bound: 0.0005330
time: 1.42 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0086903, -0.0051358, -0.0089934, -0.0050759, -0.0025782, 0.0028864
1: -0.0053888, -0.0043866, -0.0054742, -0.0043697, -0.0007269, 0.0008138
2: -0.0011997, 0.0061943, -0.0018302, 0.0063190, -0.0053632, 0.0060043
3: 0.0014685, 0.0024470, 0.0013851, 0.0024635, -0.0007097, 0.0007946
4: 0.0014626, 0.0069885, 0.0013695, 0.0074597, -0.0044873, 0.0040081
5: 0.9959125, 0.9974478, 0.9958866, 0.9975787, -0.0012467, 0.0011136
6: 0.0041735, 0.0055671, 0.0041500, 0.0056859, -0.0011316, 0.0010108
7: -0.0078067, -0.0026062, -0.0078943, -0.0021628, -0.0042230, 0.0037721
8: -0.0071644, -0.0031169, -0.0075096, -0.0030487, -0.0029358, 0.0032868
9: -0.0037408, -0.0033916, -0.0037467, -0.0033618, -0.0002836, 0.0002533

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005745, upper bound: 0.0005312
time: 1.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005682, upper bound: 0.0005328
time: 1.71 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0086948, -0.0051345, -0.0090071, -0.0050550, -0.0026096, 0.0028745
1: -0.0053901, -0.0043863, -0.0054781, -0.0043639, -0.0007357, 0.0008104
2: -0.0012091, 0.0061971, -0.0018588, 0.0063624, -0.0054284, 0.0059796
3: 0.0014673, 0.0024474, 0.0013813, 0.0024693, -0.0007184, 0.0007913
4: 0.0014606, 0.0069955, 0.0013370, 0.0074810, -0.0044687, 0.0040569
5: 0.9959121, 0.9974498, 0.9958777, 0.9975847, -0.0012415, 0.0011271
6: 0.0041730, 0.0055688, 0.0041418, 0.0056913, -0.0011270, 0.0010231
7: -0.0078086, -0.0025996, -0.0079248, -0.0021426, -0.0042056, 0.0038180
8: -0.0071696, -0.0031154, -0.0075252, -0.0030249, -0.0029715, 0.0032732
9: -0.0037410, -0.0033912, -0.0037488, -0.0033605, -0.0002824, 0.0002564

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005769, upper bound: 0.0005312
time: 1.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005709, upper bound: 0.0005328
time: 1.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0088153, -0.0052285, -0.0091474, -0.0051593, -0.0025745, 0.0028631
1: -0.0054240, -0.0044128, -0.0055177, -0.0043933, -0.0007258, 0.0008072
2: -0.0014598, 0.0060016, -0.0021506, 0.0061454, -0.0053555, 0.0059558
3: 0.0014341, 0.0024215, 0.0013427, 0.0024405, -0.0007087, 0.0007882
4: 0.0016067, 0.0071828, 0.0014992, 0.0076991, -0.0044510, 0.0040024
5: 0.9959526, 0.9975019, 0.9959227, 0.9976453, -0.0012366, 0.0011120
6: 0.0042098, 0.0056161, 0.0041827, 0.0057463, -0.0011225, 0.0010093
7: -0.0076711, -0.0024233, -0.0077723, -0.0019374, -0.0041889, 0.0037667
8: -0.0073068, -0.0032225, -0.0076850, -0.0031437, -0.0029316, 0.0032602
9: -0.0037317, -0.0033793, -0.0037385, -0.0033467, -0.0002813, 0.0002529

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005487
time: 1.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005545
time: 1.90 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0088218, -0.0052271, -0.0091599, -0.0051423, -0.0025981, 0.0028374
1: -0.0054259, -0.0044124, -0.0055212, -0.0043885, -0.0007325, 0.0008000
2: -0.0014734, 0.0060045, -0.0021765, 0.0061808, -0.0054045, 0.0059024
3: 0.0014323, 0.0024219, 0.0013393, 0.0024452, -0.0007152, 0.0007811
4: 0.0016045, 0.0071930, 0.0014727, 0.0077185, -0.0044111, 0.0040390
5: 0.9959520, 0.9975047, 0.9959154, 0.9976506, -0.0012255, 0.0011221
6: 0.0042093, 0.0056186, 0.0041761, 0.0057512, -0.0011124, 0.0010186
7: -0.0076731, -0.0024138, -0.0077972, -0.0019192, -0.0041513, 0.0038011
8: -0.0073142, -0.0032209, -0.0076992, -0.0031243, -0.0029584, 0.0032310
9: -0.0037319, -0.0033787, -0.0037402, -0.0033455, -0.0002788, 0.0002552

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005487
time: 1.95 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005546
time: 1.51 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0087472, -0.0051337, -0.0091042, -0.0051698, -0.0026012, 0.0029450
1: -0.0054048, -0.0043860, -0.0055055, -0.0043962, -0.0007334, 0.0008303
2: -0.0013181, 0.0061988, -0.0020608, 0.0061236, -0.0054109, 0.0061262
3: 0.0014529, 0.0024476, 0.0013546, 0.0024377, -0.0007161, 0.0008107
4: 0.0014593, 0.0070769, 0.0015155, 0.0076320, -0.0045784, 0.0040438
5: 0.9959117, 0.9974724, 0.9959273, 0.9976266, -0.0012720, 0.0011235
6: 0.0041727, 0.0055894, 0.0041868, 0.0057293, -0.0011546, 0.0010198
7: -0.0078098, -0.0025230, -0.0077569, -0.0020006, -0.0043087, 0.0038057
8: -0.0072292, -0.0031145, -0.0076358, -0.0031556, -0.0029620, 0.0033535
9: -0.0037410, -0.0033860, -0.0037375, -0.0033510, -0.0002893, 0.0002555

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005872, upper bound: 0.0005486
time: 1.97 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005872, upper bound: 0.0005552
time: 2.11 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0087519, -0.0051324, -0.0091150, -0.0051527, -0.0026272, 0.0029204
1: -0.0054061, -0.0043857, -0.0055085, -0.0043914, -0.0007407, 0.0008234
2: -0.0013279, 0.0062015, -0.0020832, 0.0061592, -0.0054650, 0.0060749
3: 0.0014516, 0.0024480, 0.0013516, 0.0024424, -0.0007232, 0.0008039
4: 0.0014573, 0.0070842, 0.0014888, 0.0076487, -0.0045400, 0.0040842
5: 0.9959111, 0.9974744, 0.9959198, 0.9976313, -0.0012614, 0.0011347
6: 0.0041722, 0.0055912, 0.0041801, 0.0057336, -0.0011449, 0.0010300
7: -0.0078117, -0.0025161, -0.0077820, -0.0019849, -0.0042727, 0.0038437
8: -0.0072346, -0.0031130, -0.0076480, -0.0031361, -0.0029916, 0.0033254
9: -0.0037412, -0.0033856, -0.0037392, -0.0033499, -0.0002869, 0.0002581

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005484
time: 1.87 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005554
time: 1.31 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0090835, -0.0051511, -0.0087210, -0.0051373, -0.0027601, 0.0025588
1: -0.0054996, -0.0043909, -0.0053974, -0.0043871, -0.0007782, 0.0007214
2: -0.0020176, 0.0061625, -0.0012636, 0.0061913, -0.0057415, 0.0053229
3: 0.0013603, 0.0024428, 0.0014601, 0.0024466, -0.0007598, 0.0007044
4: 0.0014864, 0.0075997, 0.0014649, 0.0070362, -0.0039780, 0.0042908
5: 0.9959192, 0.9976176, 0.9959132, 0.9974611, -0.0011052, 0.0011921
6: 0.0041795, 0.0057212, 0.0041741, 0.0055791, -0.0010032, 0.0010821
7: -0.0077843, -0.0020310, -0.0078045, -0.0025613, -0.0037437, 0.0040382
8: -0.0076122, -0.0031343, -0.0071994, -0.0031186, -0.0031429, 0.0029138
9: -0.0037393, -0.0033530, -0.0037407, -0.0033886, -0.0002514, 0.0002712

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005343, upper bound: 0.0005712
time: 1.52 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005274, upper bound: 0.0005724
time: 1.93 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0090897, -0.0051499, -0.0087306, -0.0051211, -0.0027966, 0.0025519
1: -0.0055014, -0.0043906, -0.0054002, -0.0043825, -0.0007885, 0.0007195
2: -0.0020306, 0.0061649, -0.0012836, 0.0062250, -0.0058176, 0.0053085
3: 0.0013586, 0.0024431, 0.0014574, 0.0024511, -0.0007699, 0.0007025
4: 0.0014846, 0.0076094, 0.0014397, 0.0070512, -0.0039672, 0.0043477
5: 0.9959187, 0.9976204, 0.9959062, 0.9974653, -0.0011022, 0.0012079
6: 0.0041791, 0.0057237, 0.0041677, 0.0055829, -0.0010005, 0.0010964
7: -0.0077860, -0.0020218, -0.0078282, -0.0025472, -0.0037336, 0.0040917
8: -0.0076193, -0.0031330, -0.0072104, -0.0031001, -0.0031846, 0.0029059
9: -0.0037394, -0.0033524, -0.0037423, -0.0033877, -0.0002507, 0.0002747

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005343, upper bound: 0.0005711
time: 1.98 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005299, upper bound: 0.0005725
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0090180, -0.0050560, -0.0086732, -0.0051459, -0.0027774, 0.0026495
1: -0.0054812, -0.0043641, -0.0053840, -0.0043895, -0.0007830, 0.0007470
2: -0.0018814, 0.0063603, -0.0011642, 0.0061733, -0.0057775, 0.0055115
3: 0.0013783, 0.0024690, 0.0014732, 0.0024442, -0.0007646, 0.0007294
4: 0.0013386, 0.0074979, 0.0014783, 0.0069619, -0.0041190, 0.0043177
5: 0.9958782, 0.9975894, 0.9959170, 0.9974405, -0.0011444, 0.0011996
6: 0.0041422, 0.0056955, 0.0041775, 0.0055604, -0.0010387, 0.0010889
7: -0.0079234, -0.0021268, -0.0077919, -0.0026312, -0.0038764, 0.0040635
8: -0.0075376, -0.0030261, -0.0071450, -0.0031285, -0.0031626, 0.0030170
9: -0.0037487, -0.0033594, -0.0037398, -0.0033933, -0.0002603, 0.0002729

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005342, upper bound: 0.0005756
time: 1.56 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005268, upper bound: 0.0005768
time: 1.56 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0090230, -0.0050550, -0.0086806, -0.0051295, -0.0028184, 0.0026479
1: -0.0054826, -0.0043638, -0.0053860, -0.0043849, -0.0007946, 0.0007465
2: -0.0018918, 0.0063625, -0.0011795, 0.0062073, -0.0058628, 0.0055082
3: 0.0013769, 0.0024693, 0.0014712, 0.0024487, -0.0007758, 0.0007289
4: 0.0013369, 0.0075057, 0.0014529, 0.0069734, -0.0041165, 0.0043815
5: 0.9958777, 0.9975915, 0.9959099, 0.9974437, -0.0011437, 0.0012173
6: 0.0041418, 0.0056975, 0.0041711, 0.0055632, -0.0010381, 0.0011049
7: -0.0079249, -0.0021195, -0.0078158, -0.0026204, -0.0038741, 0.0041235
8: -0.0075433, -0.0030249, -0.0071534, -0.0031098, -0.0032093, 0.0030152
9: -0.0037488, -0.0033589, -0.0037414, -0.0033926, -0.0002601, 0.0002769

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005366, upper bound: 0.0005756
time: 1.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005294, upper bound: 0.0005768
time: 1.50 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0091415, -0.0051489, -0.0088211, -0.0052426, -0.0027962, 0.0026127
1: -0.0055160, -0.0043903, -0.0054257, -0.0044168, -0.0007884, 0.0007366
2: -0.0021382, 0.0061671, -0.0014719, 0.0059721, -0.0058168, 0.0054349
3: 0.0013443, 0.0024434, 0.0014325, 0.0024176, -0.0007698, 0.0007192
4: 0.0014829, 0.0076899, 0.0016287, 0.0071918, -0.0040617, 0.0043471
5: 0.9959182, 0.9976428, 0.9959587, 0.9975044, -0.0011285, 0.0012077
6: 0.0041786, 0.0057439, 0.0042154, 0.0056183, -0.0010243, 0.0010963
7: -0.0077875, -0.0019461, -0.0076504, -0.0024148, -0.0038225, 0.0040911
8: -0.0076782, -0.0031318, -0.0073134, -0.0032386, -0.0031841, 0.0029751
9: -0.0037395, -0.0033473, -0.0037303, -0.0033788, -0.0002567, 0.0002747

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005464, upper bound: 0.0005848
time: 2.07 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005464, upper bound: 0.0005929
time: 2.06 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0091475, -0.0051477, -0.0088329, -0.0052217, -0.0028200, 0.0026024
1: -0.0055177, -0.0043900, -0.0054290, -0.0044109, -0.0007951, 0.0007337
2: -0.0021509, 0.0061695, -0.0014964, 0.0060156, -0.0058663, 0.0054136
3: 0.0013427, 0.0024437, 0.0014293, 0.0024234, -0.0007763, 0.0007164
4: 0.0014811, 0.0076993, 0.0015962, 0.0072102, -0.0040458, 0.0043841
5: 0.9959178, 0.9976453, 0.9959497, 0.9975094, -0.0011240, 0.0012180
6: 0.0041782, 0.0057463, 0.0042072, 0.0056230, -0.0010203, 0.0011056
7: -0.0077892, -0.0019372, -0.0076809, -0.0023975, -0.0038075, 0.0041259
8: -0.0076851, -0.0031305, -0.0073269, -0.0032148, -0.0032112, 0.0029634
9: -0.0037397, -0.0033467, -0.0037324, -0.0033776, -0.0002557, 0.0002770

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005464, upper bound: 0.0005848
time: 2.12 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005464, upper bound: 0.0005930
time: 2.05 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0090764, -0.0050541, -0.0087759, -0.0052516, -0.0028130, 0.0027042
1: -0.0054976, -0.0043636, -0.0054129, -0.0044193, -0.0007931, 0.0007624
2: -0.0020029, 0.0063644, -0.0013778, 0.0059534, -0.0058515, 0.0056252
3: 0.0013622, 0.0024695, 0.0014450, 0.0024151, -0.0007744, 0.0007444
4: 0.0013355, 0.0075887, 0.0016427, 0.0071215, -0.0042039, 0.0043731
5: 0.9958773, 0.9976146, 0.9959627, 0.9974848, -0.0011680, 0.0012150
6: 0.0041415, 0.0057184, 0.0042189, 0.0056006, -0.0010602, 0.0011028
7: -0.0079263, -0.0020413, -0.0076372, -0.0024810, -0.0039564, 0.0041155
8: -0.0076041, -0.0030238, -0.0072619, -0.0032488, -0.0032031, 0.0030792
9: -0.0037489, -0.0033537, -0.0037294, -0.0033832, -0.0002657, 0.0002764

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005464, upper bound: 0.0005895
time: 2.10 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005464, upper bound: 0.0005983
time: 1.98 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0090810, -0.0050530, -0.0087870, -0.0052308, -0.0028390, 0.0027006
1: -0.0054989, -0.0043633, -0.0054160, -0.0044134, -0.0008004, 0.0007614
2: -0.0020125, 0.0063666, -0.0014009, 0.0059967, -0.0059056, 0.0056178
3: 0.0013610, 0.0024698, 0.0014419, 0.0024209, -0.0007815, 0.0007434
4: 0.0013339, 0.0075959, 0.0016103, 0.0071388, -0.0041984, 0.0044135
5: 0.9958768, 0.9976165, 0.9959537, 0.9974897, -0.0011664, 0.0012262
6: 0.0041411, 0.0057202, 0.0042108, 0.0056050, -0.0010588, 0.0011130
7: -0.0079278, -0.0020346, -0.0076677, -0.0024648, -0.0039511, 0.0041536
8: -0.0076094, -0.0030226, -0.0072745, -0.0032251, -0.0032328, 0.0030752
9: -0.0037490, -0.0033532, -0.0037315, -0.0033821, -0.0002653, 0.0002789

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005484, upper bound: 0.0005897
time: 1.93 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005484, upper bound: 0.0005984
time: 1.56 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0090835, -0.0051511, -0.0090380, -0.0050657, -0.0024429, 0.0024817
1: -0.0054996, -0.0043909, -0.0054868, -0.0043669, -0.0006887, 0.0006997
2: -0.0020176, 0.0061625, -0.0019231, 0.0063401, -0.0050817, 0.0051625
3: 0.0013603, 0.0024428, 0.0013728, 0.0024663, -0.0006725, 0.0006832
4: 0.0014864, 0.0075997, 0.0013537, 0.0075291, -0.0038582, 0.0037977
5: 0.9959192, 0.9976176, 0.9958823, 0.9975980, -0.0010719, 0.0010551
6: 0.0041795, 0.0057212, 0.0041461, 0.0057034, -0.0009730, 0.0009577
7: -0.0077843, -0.0020310, -0.0079092, -0.0020974, -0.0036310, 0.0035741
8: -0.0076122, -0.0031343, -0.0075604, -0.0030372, -0.0027817, 0.0028260
9: -0.0037393, -0.0033530, -0.0037477, -0.0033575, -0.0002438, 0.0002400

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005349, upper bound: 0.0005710
time: 1.36 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005283, upper bound: 0.0005725
time: 1.46 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0090897, -0.0051499, -0.0090530, -0.0050451, -0.0024820, 0.0024640
1: -0.0055014, -0.0043906, -0.0054910, -0.0043611, -0.0006998, 0.0006947
2: -0.0020306, 0.0061649, -0.0019543, 0.0063830, -0.0051631, 0.0051257
3: 0.0013586, 0.0024431, 0.0013687, 0.0024720, -0.0006833, 0.0006783
4: 0.0014846, 0.0076094, 0.0013216, 0.0075524, -0.0038306, 0.0038586
5: 0.9959187, 0.9976204, 0.9958734, 0.9976045, -0.0010643, 0.0010720
6: 0.0041791, 0.0057237, 0.0041380, 0.0057093, -0.0009660, 0.0009731
7: -0.0077860, -0.0020218, -0.0079393, -0.0020755, -0.0036050, 0.0036314
8: -0.0076193, -0.0031330, -0.0075775, -0.0030137, -0.0028263, 0.0028058
9: -0.0037394, -0.0033524, -0.0037497, -0.0033560, -0.0002421, 0.0002438

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005374, upper bound: 0.0005711
time: 1.40 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005306, upper bound: 0.0005725
time: 1.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0090180, -0.0050560, -0.0089934, -0.0050759, -0.0024841, 0.0025635
1: -0.0054812, -0.0043641, -0.0054742, -0.0043697, -0.0007004, 0.0007228
2: -0.0018814, 0.0063603, -0.0018302, 0.0063190, -0.0051675, 0.0053326
3: 0.0013783, 0.0024690, 0.0013851, 0.0024635, -0.0006838, 0.0007057
4: 0.0013386, 0.0074979, 0.0013695, 0.0074597, -0.0039853, 0.0038619
5: 0.9958782, 0.9975894, 0.9958866, 0.9975787, -0.0011072, 0.0010729
6: 0.0041422, 0.0056955, 0.0041500, 0.0056859, -0.0010050, 0.0009739
7: -0.0079234, -0.0021268, -0.0078943, -0.0021628, -0.0037506, 0.0036344
8: -0.0075376, -0.0030261, -0.0075096, -0.0030487, -0.0028287, 0.0029191
9: -0.0037487, -0.0033594, -0.0037467, -0.0033618, -0.0002518, 0.0002440

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005351, upper bound: 0.0005756
time: 1.44 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005283, upper bound: 0.0005768
time: 1.45 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0090230, -0.0050550, -0.0090071, -0.0050550, -0.0025240, 0.0025500
1: -0.0054826, -0.0043638, -0.0054781, -0.0043639, -0.0007116, 0.0007189
2: -0.0018918, 0.0063625, -0.0018588, 0.0063624, -0.0052504, 0.0053046
3: 0.0013769, 0.0024693, 0.0013813, 0.0024693, -0.0006948, 0.0007020
4: 0.0013369, 0.0075057, 0.0013370, 0.0074810, -0.0039643, 0.0039238
5: 0.9958777, 0.9975915, 0.9958777, 0.9975847, -0.0011014, 0.0010902
6: 0.0041418, 0.0056975, 0.0041418, 0.0056913, -0.0009997, 0.0009895
7: -0.0079249, -0.0021195, -0.0079248, -0.0021426, -0.0037309, 0.0036928
8: -0.0075433, -0.0030249, -0.0075252, -0.0030249, -0.0028741, 0.0029037
9: -0.0037488, -0.0033589, -0.0037488, -0.0033605, -0.0002505, 0.0002480

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005374, upper bound: 0.0005755
time: 1.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005308, upper bound: 0.0005768
time: 1.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0091415, -0.0051489, -0.0091474, -0.0051593, -0.0024833, 0.0025406
1: -0.0055160, -0.0043903, -0.0055177, -0.0043933, -0.0007001, 0.0007163
2: -0.0021382, 0.0061671, -0.0021506, 0.0061454, -0.0051657, 0.0052850
3: 0.0013443, 0.0024434, 0.0013427, 0.0024405, -0.0006836, 0.0006994
4: 0.0014829, 0.0076899, 0.0014992, 0.0076991, -0.0039497, 0.0038605
5: 0.9959182, 0.9976428, 0.9959227, 0.9976453, -0.0010973, 0.0010726
6: 0.0041786, 0.0057439, 0.0041827, 0.0057463, -0.0009961, 0.0009736
7: -0.0077875, -0.0019461, -0.0077723, -0.0019374, -0.0037171, 0.0036332
8: -0.0076782, -0.0031318, -0.0076850, -0.0031437, -0.0028277, 0.0028930
9: -0.0037395, -0.0033473, -0.0037385, -0.0033467, -0.0002496, 0.0002440

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005467, upper bound: 0.0005847
time: 1.88 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005467, upper bound: 0.0005929
time: 1.56 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0091475, -0.0051477, -0.0091599, -0.0051423, -0.0025094, 0.0025114
1: -0.0055177, -0.0043900, -0.0055212, -0.0043885, -0.0007075, 0.0007081
2: -0.0021509, 0.0061695, -0.0021765, 0.0061808, -0.0052201, 0.0052242
3: 0.0013427, 0.0024437, 0.0013393, 0.0024452, -0.0006908, 0.0006913
4: 0.0014811, 0.0076993, 0.0014727, 0.0077185, -0.0039043, 0.0039011
5: 0.9959178, 0.9976453, 0.9959154, 0.9976506, -0.0010847, 0.0010839
6: 0.0041782, 0.0057463, 0.0041761, 0.0057512, -0.0009846, 0.0009838
7: -0.0077892, -0.0019372, -0.0077972, -0.0019192, -0.0036744, 0.0036714
8: -0.0076851, -0.0031305, -0.0076992, -0.0031243, -0.0028575, 0.0028598
9: -0.0037397, -0.0033467, -0.0037402, -0.0033455, -0.0002467, 0.0002465

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005848
time: 1.88 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005930
time: 1.35 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0090764, -0.0050541, -0.0091042, -0.0051698, -0.0025249, 0.0026254
1: -0.0054976, -0.0043636, -0.0055055, -0.0043962, -0.0007119, 0.0007402
2: -0.0020029, 0.0063644, -0.0020608, 0.0061236, -0.0052523, 0.0054614
3: 0.0013622, 0.0024695, 0.0013546, 0.0024377, -0.0006951, 0.0007227
4: 0.0013355, 0.0075887, 0.0015155, 0.0076320, -0.0040815, 0.0039253
5: 0.9958773, 0.9976146, 0.9959273, 0.9976266, -0.0011340, 0.0010906
6: 0.0041415, 0.0057184, 0.0041868, 0.0057293, -0.0010293, 0.0009899
7: -0.0079263, -0.0020413, -0.0077569, -0.0020006, -0.0038411, 0.0036941
8: -0.0076041, -0.0030238, -0.0076358, -0.0031556, -0.0028751, 0.0029896
9: -0.0037489, -0.0033537, -0.0037375, -0.0033510, -0.0002579, 0.0002481

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005470, upper bound: 0.0005897
time: 1.90 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005470, upper bound: 0.0005983
time: 1.93 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0090810, -0.0050530, -0.0091150, -0.0051527, -0.0025524, 0.0025991
1: -0.0054989, -0.0043633, -0.0055085, -0.0043914, -0.0007196, 0.0007328
2: -0.0020125, 0.0063666, -0.0020832, 0.0061592, -0.0053095, 0.0054066
3: 0.0013610, 0.0024698, 0.0013516, 0.0024424, -0.0007026, 0.0007155
4: 0.0013339, 0.0075959, 0.0014888, 0.0076487, -0.0040405, 0.0039680
5: 0.9958768, 0.9976165, 0.9959198, 0.9976313, -0.0011226, 0.0011024
6: 0.0041411, 0.0057202, 0.0041801, 0.0057336, -0.0010190, 0.0010007
7: -0.0079278, -0.0020346, -0.0077820, -0.0019849, -0.0038026, 0.0037343
8: -0.0076094, -0.0030226, -0.0076480, -0.0031361, -0.0029064, 0.0029596
9: -0.0037490, -0.0033532, -0.0037392, -0.0033499, -0.0002553, 0.0002508

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005897
time: 1.90 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005983
time: 2.03 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.50 seconds
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005745, upper bound: 0.0005113
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005687, upper bound: 0.0005163
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005769, upper bound: 0.0005113
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005711, upper bound: 0.0005162
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005746, upper bound: 0.0005093
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005680, upper bound: 0.0005137
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005769, upper bound: 0.0005093
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005708, upper bound: 0.0005137
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005355
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005400
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005354
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005399
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005340
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005390
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005340
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005390
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005746, upper bound: 0.0005313
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005687, upper bound: 0.0005331
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005769, upper bound: 0.0005313
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005711, upper bound: 0.0005330
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005745, upper bound: 0.0005312
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005682, upper bound: 0.0005328
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005769, upper bound: 0.0005312
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005709, upper bound: 0.0005328
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005487
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005873, upper bound: 0.0005545
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005487
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005546
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005872, upper bound: 0.0005486
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005872, upper bound: 0.0005552
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005484
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005895, upper bound: 0.0005554
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005343, upper bound: 0.0005712
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005274, upper bound: 0.0005724
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005343, upper bound: 0.0005711
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005299, upper bound: 0.0005725
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005342, upper bound: 0.0005756
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005268, upper bound: 0.0005768
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005366, upper bound: 0.0005756
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005294, upper bound: 0.0005768
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005464, upper bound: 0.0005848
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005464, upper bound: 0.0005929
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005464, upper bound: 0.0005848
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005464, upper bound: 0.0005930
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005464, upper bound: 0.0005895
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005464, upper bound: 0.0005983
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005484, upper bound: 0.0005897
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005484, upper bound: 0.0005984
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005349, upper bound: 0.0005710
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005283, upper bound: 0.0005725
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005374, upper bound: 0.0005711
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005306, upper bound: 0.0005725
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005351, upper bound: 0.0005756
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005283, upper bound: 0.0005768
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005374, upper bound: 0.0005755
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005308, upper bound: 0.0005768
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005467, upper bound: 0.0005847
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005467, upper bound: 0.0005929
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005848
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005930
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005470, upper bound: 0.0005897
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005470, upper bound: 0.0005983
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005897
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005983

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0085700, -0.0051542, -0.0090923, -0.0051636, -0.0023881, 0.0028785
1: -0.0053549, -0.0043918, -0.0055021, -0.0043945, -0.0006733, 0.0008115
2: -0.0009495, 0.0061561, -0.0020359, 0.0061366, -0.0049676, 0.0059878
3: 0.0015016, 0.0024420, 0.0013579, 0.0024394, -0.0006574, 0.0007924
4: 0.0014912, 0.0068015, 0.0015058, 0.0076134, -0.0044749, 0.0037125
5: 0.9959205, 0.9973959, 0.9959246, 0.9976214, -0.0012433, 0.0010314
6: 0.0041807, 0.0055199, 0.0041844, 0.0057247, -0.0011285, 0.0009362
7: -0.0077797, -0.0027822, -0.0077661, -0.0020181, -0.0042114, 0.0034939
8: -0.0070275, -0.0031379, -0.0076222, -0.0031485, -0.0027193, 0.0032777
9: -0.0037390, -0.0034034, -0.0037381, -0.0033521, -0.0002828, 0.0002346

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005420, upper bound: 0.0004807
time: 2.10 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005732, upper bound: 0.0005221
time: 1.94 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0086647, -0.0052653, -0.0090923, -0.0051636, -0.0024319, 0.0027225
1: -0.0053816, -0.0044231, -0.0055021, -0.0043945, -0.0006856, 0.0007676
2: -0.0011464, 0.0059250, -0.0020359, 0.0061366, -0.0050588, 0.0056633
3: 0.0014756, 0.0024114, 0.0013579, 0.0024394, -0.0006694, 0.0007495
4: 0.0016639, 0.0069486, 0.0015058, 0.0076134, -0.0042324, 0.0037806
5: 0.9959685, 0.9974368, 0.9959246, 0.9976214, -0.0011759, 0.0010504
6: 0.0042243, 0.0055570, 0.0041844, 0.0057247, -0.0010674, 0.0009534
7: -0.0076172, -0.0026437, -0.0077661, -0.0020181, -0.0039832, 0.0035580
8: -0.0071353, -0.0032644, -0.0076222, -0.0031485, -0.0027692, 0.0031001
9: -0.0037281, -0.0033941, -0.0037381, -0.0033521, -0.0002675, 0.0002389

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005420, upper bound: 0.0005070
time: 2.10 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005732, upper bound: 0.0005266
time: 2.00 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0085741, -0.0051527, -0.0091031, -0.0051468, -0.0024173, 0.0028644
1: -0.0053560, -0.0043914, -0.0055052, -0.0043897, -0.0006815, 0.0008076
2: -0.0009580, 0.0061591, -0.0020585, 0.0061714, -0.0050284, 0.0059586
3: 0.0015005, 0.0024424, 0.0013549, 0.0024440, -0.0006654, 0.0007885
4: 0.0014890, 0.0068078, 0.0014797, 0.0076302, -0.0044531, 0.0037579
5: 0.9959199, 0.9973977, 0.9959173, 0.9976262, -0.0012372, 0.0010441
6: 0.0041802, 0.0055215, 0.0041778, 0.0057289, -0.0011230, 0.0009477
7: -0.0077819, -0.0027762, -0.0077905, -0.0020022, -0.0041909, 0.0035366
8: -0.0070321, -0.0031362, -0.0076345, -0.0031295, -0.0027525, 0.0032618
9: -0.0037392, -0.0034030, -0.0037397, -0.0033511, -0.0002814, 0.0002375

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005451, upper bound: 0.0004807
time: 1.78 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005752, upper bound: 0.0005222
time: 1.80 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0086665, -0.0052639, -0.0091031, -0.0051468, -0.0024645, 0.0027098
1: -0.0053821, -0.0044228, -0.0055052, -0.0043897, -0.0006948, 0.0007640
2: -0.0011503, 0.0059278, -0.0020585, 0.0061714, -0.0051268, 0.0056370
3: 0.0014751, 0.0024117, 0.0013549, 0.0024440, -0.0006784, 0.0007460
4: 0.0016618, 0.0069515, 0.0014797, 0.0076302, -0.0042128, 0.0038314
5: 0.9959680, 0.9974375, 0.9959173, 0.9976262, -0.0011704, 0.0010645
6: 0.0042238, 0.0055577, 0.0041778, 0.0057289, -0.0010624, 0.0009662
7: -0.0076192, -0.0026410, -0.0077905, -0.0020022, -0.0039647, 0.0036058
8: -0.0071374, -0.0032628, -0.0076345, -0.0031295, -0.0028064, 0.0030857
9: -0.0037282, -0.0033940, -0.0037397, -0.0033511, -0.0002662, 0.0002421

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005451, upper bound: 0.0005070
time: 1.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005752, upper bound: 0.0005266
time: 1.96 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0084920, -0.0050665, -0.0090489, -0.0051738, -0.0024200, 0.0029699
1: -0.0053329, -0.0043671, -0.0054899, -0.0043973, -0.0006823, 0.0008373
2: -0.0007873, 0.0063384, -0.0019457, 0.0061153, -0.0050341, 0.0061780
3: 0.0015231, 0.0024661, 0.0013698, 0.0024366, -0.0006662, 0.0008176
4: 0.0013549, 0.0066803, 0.0015217, 0.0075459, -0.0046171, 0.0037621
5: 0.9958827, 0.9973623, 0.9959290, 0.9976027, -0.0012828, 0.0010452
6: 0.0041464, 0.0054893, 0.0041884, 0.0057076, -0.0011644, 0.0009488
7: -0.0079080, -0.0028963, -0.0077511, -0.0020816, -0.0043452, 0.0035406
8: -0.0069387, -0.0030381, -0.0075728, -0.0031602, -0.0027557, 0.0033818
9: -0.0037476, -0.0034111, -0.0037371, -0.0033564, -0.0002918, 0.0002377

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005668, upper bound: 0.0005229
time: 1.70 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005680, upper bound: 0.0005135
time: 2.06 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0085893, -0.0051716, -0.0090489, -0.0051738, -0.0024658, 0.0028145
1: -0.0053603, -0.0043967, -0.0054899, -0.0043973, -0.0006952, 0.0007935
2: -0.0009897, 0.0061200, -0.0019457, 0.0061153, -0.0051293, 0.0058547
3: 0.0014963, 0.0024372, 0.0013698, 0.0024366, -0.0006788, 0.0007748
4: 0.0015182, 0.0068315, 0.0015217, 0.0075459, -0.0043754, 0.0038333
5: 0.9959280, 0.9974043, 0.9959290, 0.9976027, -0.0012156, 0.0010650
6: 0.0041875, 0.0055275, 0.0041884, 0.0057076, -0.0011034, 0.0009667
7: -0.0077544, -0.0027540, -0.0077511, -0.0020816, -0.0041177, 0.0036076
8: -0.0070495, -0.0031576, -0.0075728, -0.0031602, -0.0028078, 0.0032048
9: -0.0037373, -0.0034015, -0.0037371, -0.0033564, -0.0002765, 0.0002422

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005666, upper bound: 0.0005281
time: 2.15 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005680, upper bound: 0.0005209
time: 2.15 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0084953, -0.0050653, -0.0090574, -0.0051569, -0.0024486, 0.0029570
1: -0.0053338, -0.0043668, -0.0054923, -0.0043926, -0.0006904, 0.0008337
2: -0.0007941, 0.0063410, -0.0019634, 0.0061504, -0.0050936, 0.0061512
3: 0.0015222, 0.0024664, 0.0013675, 0.0024412, -0.0006741, 0.0008140
4: 0.0013530, 0.0066853, 0.0014954, 0.0075592, -0.0045970, 0.0038067
5: 0.9958822, 0.9973637, 0.9959217, 0.9976064, -0.0012772, 0.0010576
6: 0.0041459, 0.0054906, 0.0041818, 0.0057110, -0.0011593, 0.0009600
7: -0.0079098, -0.0028915, -0.0077758, -0.0020691, -0.0043263, 0.0035825
8: -0.0069424, -0.0030366, -0.0075825, -0.0031410, -0.0027883, 0.0033672
9: -0.0037477, -0.0034108, -0.0037387, -0.0033556, -0.0002905, 0.0002406

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005696, upper bound: 0.0005229
time: 2.01 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005709, upper bound: 0.0005134
time: 2.25 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0085933, -0.0051704, -0.0090574, -0.0051569, -0.0024985, 0.0028026
1: -0.0053614, -0.0043964, -0.0054923, -0.0043926, -0.0007044, 0.0007902
2: -0.0009980, 0.0061224, -0.0019634, 0.0061504, -0.0051974, 0.0058299
3: 0.0014952, 0.0024375, 0.0013675, 0.0024412, -0.0006878, 0.0007715
4: 0.0015164, 0.0068377, 0.0014954, 0.0075592, -0.0043569, 0.0038842
5: 0.9959275, 0.9974059, 0.9959217, 0.9976064, -0.0012105, 0.0010792
6: 0.0041871, 0.0055290, 0.0041818, 0.0057110, -0.0010988, 0.0009795
7: -0.0077560, -0.0027481, -0.0077758, -0.0020691, -0.0041004, 0.0036555
8: -0.0070540, -0.0031563, -0.0075825, -0.0031410, -0.0028451, 0.0031913
9: -0.0037374, -0.0034011, -0.0037387, -0.0033556, -0.0002753, 0.0002455

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005696, upper bound: 0.0005281
time: 2.08 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005709, upper bound: 0.0005209
time: 2.25 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0086731, -0.0051288, -0.0091474, -0.0051593, -0.0023944, 0.0030151
1: -0.0053839, -0.0043847, -0.0055177, -0.0043933, -0.0006751, 0.0008501
2: -0.0011640, 0.0062089, -0.0021506, 0.0061454, -0.0049808, 0.0062720
3: 0.0014733, 0.0024490, 0.0013427, 0.0024405, -0.0006591, 0.0008300
4: 0.0014517, 0.0069618, 0.0014992, 0.0076991, -0.0046873, 0.0037223
5: 0.9959095, 0.9974405, 0.9959227, 0.9976453, -0.0013023, 0.0010342
6: 0.0041708, 0.0055603, 0.0041827, 0.0057463, -0.0011821, 0.0009387
7: -0.0078169, -0.0026313, -0.0077723, -0.0019374, -0.0044113, 0.0035031
8: -0.0071449, -0.0031089, -0.0076850, -0.0031437, -0.0027265, 0.0034333
9: -0.0037415, -0.0033933, -0.0037385, -0.0033467, -0.0002962, 0.0002352

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005487
time: 1.43 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005486
time: 1.94 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0087736, -0.0052297, -0.0091474, -0.0051593, -0.0024469, 0.0028623
1: -0.0054123, -0.0044131, -0.0055177, -0.0043933, -0.0006899, 0.0008070
2: -0.0013731, 0.0059990, -0.0021506, 0.0061454, -0.0050901, 0.0059542
3: 0.0014456, 0.0024212, 0.0013427, 0.0024405, -0.0006736, 0.0007879
4: 0.0016086, 0.0071180, 0.0014992, 0.0076991, -0.0044498, 0.0038040
5: 0.9959531, 0.9974838, 0.9959227, 0.9976453, -0.0012363, 0.0010569
6: 0.0042103, 0.0055997, 0.0041827, 0.0057463, -0.0011222, 0.0009593
7: -0.0076693, -0.0024843, -0.0077723, -0.0019374, -0.0041877, 0.0035800
8: -0.0072593, -0.0032238, -0.0076850, -0.0031437, -0.0027863, 0.0032593
9: -0.0037316, -0.0033834, -0.0037385, -0.0033467, -0.0002812, 0.0002404

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005544
time: 1.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005546
time: 1.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0086786, -0.0051273, -0.0091599, -0.0051423, -0.0024201, 0.0029878
1: -0.0053855, -0.0043842, -0.0055212, -0.0043885, -0.0006823, 0.0008424
2: -0.0011755, 0.0062120, -0.0021765, 0.0061808, -0.0050343, 0.0062152
3: 0.0014717, 0.0024494, 0.0013393, 0.0024452, -0.0006662, 0.0008225
4: 0.0014494, 0.0069703, 0.0014727, 0.0077185, -0.0046448, 0.0037623
5: 0.9959089, 0.9974428, 0.9959154, 0.9976506, -0.0012905, 0.0010453
6: 0.0041702, 0.0055625, 0.0041761, 0.0057512, -0.0011714, 0.0009488
7: -0.0078191, -0.0026233, -0.0077972, -0.0019192, -0.0043713, 0.0035407
8: -0.0071512, -0.0031073, -0.0076992, -0.0031243, -0.0027558, 0.0034022
9: -0.0037417, -0.0033928, -0.0037402, -0.0033455, -0.0002935, 0.0002378

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005702, upper bound: 0.0005486
time: 1.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005702, upper bound: 0.0005486
time: 1.91 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0087804, -0.0052283, -0.0091599, -0.0051423, -0.0024744, 0.0028366
1: -0.0054142, -0.0044127, -0.0055212, -0.0043885, -0.0006976, 0.0007997
2: -0.0013872, 0.0060019, -0.0021765, 0.0061808, -0.0051473, 0.0059007
3: 0.0014437, 0.0024216, 0.0013393, 0.0024452, -0.0006812, 0.0007809
4: 0.0016064, 0.0071285, 0.0014727, 0.0077185, -0.0044098, 0.0038468
5: 0.9959526, 0.9974867, 0.9959154, 0.9976506, -0.0012252, 0.0010687
6: 0.0042098, 0.0056024, 0.0041761, 0.0057512, -0.0011121, 0.0009701
7: -0.0076713, -0.0024744, -0.0077972, -0.0019192, -0.0041502, 0.0036202
8: -0.0072671, -0.0032223, -0.0076992, -0.0031243, -0.0028176, 0.0032301
9: -0.0037317, -0.0033828, -0.0037402, -0.0033455, -0.0002787, 0.0002431

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005702, upper bound: 0.0005545
time: 1.94 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005702, upper bound: 0.0005546
time: 1.95 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0086021, -0.0050346, -0.0091042, -0.0051698, -0.0024233, 0.0030989
1: -0.0053639, -0.0043581, -0.0055055, -0.0043962, -0.0006832, 0.0008737
2: -0.0010163, 0.0064048, -0.0020608, 0.0061236, -0.0050410, 0.0064463
3: 0.0014928, 0.0024749, 0.0013546, 0.0024377, -0.0006671, 0.0008531
4: 0.0013053, 0.0068514, 0.0015155, 0.0076320, -0.0048176, 0.0037673
5: 0.9958689, 0.9974097, 0.9959273, 0.9976266, -0.0013385, 0.0010467
6: 0.0041339, 0.0055325, 0.0041868, 0.0057293, -0.0012149, 0.0009501
7: -0.0079547, -0.0027352, -0.0077569, -0.0020006, -0.0045339, 0.0035454
8: -0.0070641, -0.0030017, -0.0076358, -0.0031556, -0.0027594, 0.0035287
9: -0.0037508, -0.0034003, -0.0037375, -0.0033510, -0.0003044, 0.0002381

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005486
time: 1.96 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005484
time: 1.98 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0087059, -0.0051348, -0.0091042, -0.0051698, -0.0024767, 0.0029441
1: -0.0053932, -0.0043864, -0.0055055, -0.0043962, -0.0006983, 0.0008301
2: -0.0012321, 0.0061964, -0.0020608, 0.0061236, -0.0051521, 0.0061244
3: 0.0014642, 0.0024473, 0.0013546, 0.0024377, -0.0006818, 0.0008105
4: 0.0014611, 0.0070127, 0.0015155, 0.0076320, -0.0045770, 0.0038504
5: 0.9959121, 0.9974546, 0.9959273, 0.9976266, -0.0012716, 0.0010697
6: 0.0041731, 0.0055732, 0.0041868, 0.0057293, -0.0011542, 0.0009710
7: -0.0078081, -0.0025834, -0.0077569, -0.0020006, -0.0043074, 0.0036236
8: -0.0071822, -0.0031158, -0.0076358, -0.0031556, -0.0028203, 0.0033525
9: -0.0037409, -0.0033901, -0.0037375, -0.0033510, -0.0002892, 0.0002433

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005553
time: 1.50 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005554
time: 2.02 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0086065, -0.0050333, -0.0091150, -0.0051527, -0.0024492, 0.0030733
1: -0.0053651, -0.0043577, -0.0055085, -0.0043914, -0.0006905, 0.0008665
2: -0.0010253, 0.0064075, -0.0020832, 0.0061592, -0.0050948, 0.0063931
3: 0.0014916, 0.0024752, 0.0013516, 0.0024424, -0.0006742, 0.0008460
4: 0.0013033, 0.0068581, 0.0014888, 0.0076487, -0.0047778, 0.0038075
5: 0.9958683, 0.9974116, 0.9959198, 0.9976313, -0.0013274, 0.0010578
6: 0.0041333, 0.0055342, 0.0041801, 0.0057336, -0.0012049, 0.0009602
7: -0.0079566, -0.0027289, -0.0077820, -0.0019849, -0.0044964, 0.0035833
8: -0.0070690, -0.0030002, -0.0076480, -0.0031361, -0.0027889, 0.0034996
9: -0.0037509, -0.0033999, -0.0037392, -0.0033499, -0.0003019, 0.0002406

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005701, upper bound: 0.0005485
time: 1.51 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005701, upper bound: 0.0005485
time: 1.85 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0087108, -0.0051335, -0.0091150, -0.0051527, -0.0025054, 0.0029195
1: -0.0053946, -0.0043860, -0.0055085, -0.0043914, -0.0007064, 0.0008231
2: -0.0012423, 0.0061991, -0.0020832, 0.0061592, -0.0052117, 0.0060731
3: 0.0014629, 0.0024476, 0.0013516, 0.0024424, -0.0006897, 0.0008037
4: 0.0014591, 0.0070203, 0.0014888, 0.0076487, -0.0045386, 0.0038949
5: 0.9959116, 0.9974567, 0.9959198, 0.9976313, -0.0012610, 0.0010821
6: 0.0041726, 0.0055751, 0.0041801, 0.0057336, -0.0011446, 0.0009822
7: -0.0078100, -0.0025762, -0.0077820, -0.0019849, -0.0042714, 0.0036656
8: -0.0071878, -0.0031143, -0.0076480, -0.0031361, -0.0028529, 0.0033244
9: -0.0037410, -0.0033896, -0.0037392, -0.0033499, -0.0002868, 0.0002461

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005701, upper bound: 0.0005554
time: 1.90 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005701, upper bound: 0.0005552
time: 1.34 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0091009, -0.0051501, -0.0088211, -0.0052426, -0.0026542, 0.0026118
1: -0.0055046, -0.0043907, -0.0054257, -0.0044168, -0.0007483, 0.0007364
2: -0.0020540, 0.0061647, -0.0014719, 0.0059721, -0.0055213, 0.0054331
3: 0.0013555, 0.0024431, 0.0014325, 0.0024176, -0.0007307, 0.0007190
4: 0.0014848, 0.0076269, 0.0016287, 0.0071918, -0.0040603, 0.0041262
5: 0.9959188, 0.9976252, 0.9959587, 0.9975044, -0.0011281, 0.0011464
6: 0.0041791, 0.0057281, 0.0042154, 0.0056183, -0.0010240, 0.0010406
7: -0.0077858, -0.0020054, -0.0076504, -0.0024148, -0.0038212, 0.0038833
8: -0.0076321, -0.0031332, -0.0073134, -0.0032386, -0.0030223, 0.0029741
9: -0.0037394, -0.0033513, -0.0037303, -0.0033788, -0.0002566, 0.0002608

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005331, upper bound: 0.0005928
time: 1.93 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005334, upper bound: 0.0005930
time: 1.91 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0091067, -0.0051489, -0.0088329, -0.0052217, -0.0026884, 0.0026015
1: -0.0055062, -0.0043903, -0.0054290, -0.0044109, -0.0007580, 0.0007335
2: -0.0020659, 0.0061671, -0.0014964, 0.0060156, -0.0055925, 0.0054117
3: 0.0013539, 0.0024434, 0.0014293, 0.0024234, -0.0007401, 0.0007162
4: 0.0014830, 0.0076358, 0.0015962, 0.0072102, -0.0040444, 0.0041795
5: 0.9959182, 0.9976277, 0.9959497, 0.9975094, -0.0011236, 0.0011612
6: 0.0041786, 0.0057303, 0.0042072, 0.0056230, -0.0010199, 0.0010540
7: -0.0077875, -0.0019970, -0.0076809, -0.0023975, -0.0038062, 0.0039334
8: -0.0076386, -0.0031318, -0.0073269, -0.0032148, -0.0030613, 0.0029624
9: -0.0037395, -0.0033507, -0.0037324, -0.0033776, -0.0002556, 0.0002641

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005337, upper bound: 0.0005930
time: 1.83 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005337, upper bound: 0.0005929
time: 1.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0089267, -0.0049563, -0.0087759, -0.0052516, -0.0026363, 0.0028621
1: -0.0054554, -0.0043360, -0.0054129, -0.0044193, -0.0007433, 0.0008069
2: -0.0016916, 0.0065678, -0.0013778, 0.0059534, -0.0054841, 0.0059537
3: 0.0014034, 0.0024964, 0.0014450, 0.0024151, -0.0007257, 0.0007879
4: 0.0011835, 0.0073560, 0.0016427, 0.0071215, -0.0044494, 0.0040984
5: 0.9958351, 0.9975500, 0.9959627, 0.9974848, -0.0012362, 0.0011387
6: 0.0041031, 0.0056598, 0.0042189, 0.0056006, -0.0011221, 0.0010336
7: -0.0080693, -0.0022603, -0.0076372, -0.0024810, -0.0041874, 0.0038571
8: -0.0074337, -0.0029125, -0.0072619, -0.0032488, -0.0030020, 0.0032590
9: -0.0037585, -0.0033684, -0.0037294, -0.0033832, -0.0002812, 0.0002590

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005332, upper bound: 0.0005897
time: 1.57 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005334, upper bound: 0.0005897
time: 1.49 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0090363, -0.0050552, -0.0087759, -0.0052516, -0.0026739, 0.0027032
1: -0.0054863, -0.0043639, -0.0054129, -0.0044193, -0.0007539, 0.0007621
2: -0.0019196, 0.0063621, -0.0013778, 0.0059534, -0.0055622, 0.0056233
3: 0.0013733, 0.0024692, 0.0014450, 0.0024151, -0.0007361, 0.0007442
4: 0.0013373, 0.0075264, 0.0016427, 0.0071215, -0.0042025, 0.0041569
5: 0.9958777, 0.9975973, 0.9959627, 0.9974848, -0.0011676, 0.0011549
6: 0.0041419, 0.0057027, 0.0042189, 0.0056006, -0.0010598, 0.0010483
7: -0.0079246, -0.0020999, -0.0076372, -0.0024810, -0.0039550, 0.0039121
8: -0.0075585, -0.0030251, -0.0072619, -0.0032488, -0.0030448, 0.0030782
9: -0.0037487, -0.0033576, -0.0037294, -0.0033832, -0.0002656, 0.0002627

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005332, upper bound: 0.0005983
time: 4.92 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005334, upper bound: 0.0005982
time: 1.98 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0089324, -0.0049552, -0.0087870, -0.0052308, -0.0026628, 0.0028567
1: -0.0054570, -0.0043357, -0.0054160, -0.0044134, -0.0007507, 0.0008054
2: -0.0017033, 0.0065701, -0.0014009, 0.0059967, -0.0055391, 0.0059426
3: 0.0014019, 0.0024967, 0.0014419, 0.0024209, -0.0007330, 0.0007864
4: 0.0011818, 0.0073648, 0.0016103, 0.0071388, -0.0044411, 0.0041396
5: 0.9958346, 0.9975524, 0.9959537, 0.9974897, -0.0012339, 0.0011501
6: 0.0041027, 0.0056620, 0.0042108, 0.0056050, -0.0011200, 0.0010439
7: -0.0080709, -0.0022520, -0.0076677, -0.0024648, -0.0041796, 0.0038958
8: -0.0074401, -0.0029113, -0.0072745, -0.0032251, -0.0030321, 0.0032530
9: -0.0037586, -0.0033678, -0.0037315, -0.0033821, -0.0002807, 0.0002616

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005339, upper bound: 0.0005896
time: 2.01 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005333, upper bound: 0.0005895
time: 2.03 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0090404, -0.0050541, -0.0087870, -0.0052308, -0.0027115, 0.0026997
1: -0.0054875, -0.0043636, -0.0054160, -0.0044134, -0.0007645, 0.0007611
2: -0.0019280, 0.0063643, -0.0014009, 0.0059967, -0.0056406, 0.0056158
3: 0.0013721, 0.0024695, 0.0014419, 0.0024209, -0.0007464, 0.0007432
4: 0.0013356, 0.0075328, 0.0016103, 0.0071388, -0.0041969, 0.0042154
5: 0.9958773, 0.9975991, 0.9959537, 0.9974897, -0.0011660, 0.0011712
6: 0.0041415, 0.0057043, 0.0042108, 0.0056050, -0.0010584, 0.0010631
7: -0.0079262, -0.0020940, -0.0076677, -0.0024648, -0.0039498, 0.0039672
8: -0.0075631, -0.0030239, -0.0072745, -0.0032251, -0.0030876, 0.0030741
9: -0.0037488, -0.0033572, -0.0037315, -0.0033821, -0.0002652, 0.0002664

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005339, upper bound: 0.0005984
time: 2.05 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005333, upper bound: 0.0005984
time: 2.12 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0091009, -0.0051501, -0.0091474, -0.0051593, -0.0023409, 0.0025398
1: -0.0055046, -0.0043907, -0.0055177, -0.0043933, -0.0006600, 0.0007161
2: -0.0020540, 0.0061647, -0.0021506, 0.0061454, -0.0048696, 0.0052833
3: 0.0013555, 0.0024431, 0.0013427, 0.0024405, -0.0006444, 0.0006992
4: 0.0014848, 0.0076269, 0.0014992, 0.0076991, -0.0039484, 0.0036392
5: 0.9959188, 0.9976252, 0.9959227, 0.9976453, -0.0010970, 0.0010111
6: 0.0041791, 0.0057281, 0.0041827, 0.0057463, -0.0009957, 0.0009178
7: -0.0077858, -0.0020054, -0.0077723, -0.0019374, -0.0037159, 0.0034249
8: -0.0076321, -0.0031332, -0.0076850, -0.0031437, -0.0026656, 0.0028921
9: -0.0037394, -0.0033513, -0.0037385, -0.0033467, -0.0002495, 0.0002300

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005342, upper bound: 0.0005930
time: 2.08 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005342, upper bound: 0.0005929
time: 1.96 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0091067, -0.0051489, -0.0091599, -0.0051423, -0.0023800, 0.0025106
1: -0.0055062, -0.0043903, -0.0055212, -0.0043885, -0.0006710, 0.0007078
2: -0.0020659, 0.0061671, -0.0021765, 0.0061808, -0.0049510, 0.0052226
3: 0.0013539, 0.0024434, 0.0013393, 0.0024452, -0.0006552, 0.0006911
4: 0.0014830, 0.0076358, 0.0014727, 0.0077185, -0.0039030, 0.0037001
5: 0.9959182, 0.9976277, 0.9959154, 0.9976506, -0.0010844, 0.0010280
6: 0.0041786, 0.0057303, 0.0041761, 0.0057512, -0.0009843, 0.0009331
7: -0.0077875, -0.0019970, -0.0077972, -0.0019192, -0.0036732, 0.0034822
8: -0.0076386, -0.0031318, -0.0076992, -0.0031243, -0.0027102, 0.0028589
9: -0.0037395, -0.0033507, -0.0037402, -0.0033455, -0.0002466, 0.0002338

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005344, upper bound: 0.0005930
time: 1.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005344, upper bound: 0.0005929
time: 2.03 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0089267, -0.0049563, -0.0091042, -0.0051698, -0.0023451, 0.0027824
1: -0.0054554, -0.0043360, -0.0055055, -0.0043962, -0.0006612, 0.0007845
2: -0.0016916, 0.0065678, -0.0020608, 0.0061236, -0.0048783, 0.0057880
3: 0.0014034, 0.0024964, 0.0013546, 0.0024377, -0.0006456, 0.0007659
4: 0.0011835, 0.0073560, 0.0015155, 0.0076320, -0.0043256, 0.0036458
5: 0.9958351, 0.9975500, 0.9959273, 0.9976266, -0.0012018, 0.0010129
6: 0.0041031, 0.0056598, 0.0041868, 0.0057293, -0.0010908, 0.0009194
7: -0.0080693, -0.0022603, -0.0077569, -0.0020006, -0.0040708, 0.0034311
8: -0.0074337, -0.0029125, -0.0076358, -0.0031556, -0.0026704, 0.0031683
9: -0.0037585, -0.0033684, -0.0037375, -0.0033510, -0.0002733, 0.0002304

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005342, upper bound: 0.0005897
time: 1.55 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005342, upper bound: 0.0005897
time: 1.95 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0090363, -0.0050552, -0.0091042, -0.0051698, -0.0023829, 0.0026245
1: -0.0054863, -0.0043639, -0.0055055, -0.0043962, -0.0006718, 0.0007400
2: -0.0019196, 0.0063621, -0.0020608, 0.0061236, -0.0049570, 0.0054595
3: 0.0013733, 0.0024692, 0.0013546, 0.0024377, -0.0006560, 0.0007225
4: 0.0013373, 0.0075264, 0.0015155, 0.0076320, -0.0040801, 0.0037045
5: 0.9958777, 0.9975973, 0.9959273, 0.9976266, -0.0011336, 0.0010292
6: 0.0041419, 0.0057027, 0.0041868, 0.0057293, -0.0010289, 0.0009342
7: -0.0079246, -0.0020999, -0.0077569, -0.0020006, -0.0038398, 0.0034864
8: -0.0075585, -0.0030251, -0.0076358, -0.0031556, -0.0027135, 0.0029886
9: -0.0037487, -0.0033576, -0.0037375, -0.0033510, -0.0002578, 0.0002341

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005342, upper bound: 0.0005983
time: 1.94 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005342, upper bound: 0.0005984
time: 2.06 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0089324, -0.0049552, -0.0091150, -0.0051527, -0.0023730, 0.0027537
1: -0.0054570, -0.0043357, -0.0055085, -0.0043914, -0.0006690, 0.0007764
2: -0.0017033, 0.0065701, -0.0020832, 0.0061592, -0.0049364, 0.0057283
3: 0.0014019, 0.0024967, 0.0013516, 0.0024424, -0.0006533, 0.0007580
4: 0.0011818, 0.0073648, 0.0014888, 0.0076487, -0.0042810, 0.0036891
5: 0.9958346, 0.9975524, 0.9959198, 0.9976313, -0.0011894, 0.0010250
6: 0.0041027, 0.0056620, 0.0041801, 0.0057336, -0.0010796, 0.0009303
7: -0.0080709, -0.0022520, -0.0077820, -0.0019849, -0.0040289, 0.0034719
8: -0.0074401, -0.0029113, -0.0076480, -0.0031361, -0.0027022, 0.0031357
9: -0.0037586, -0.0033678, -0.0037392, -0.0033499, -0.0002705, 0.0002331

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005346, upper bound: 0.0005897
time: 1.55 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005344, upper bound: 0.0005894
time: 1.85 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0090404, -0.0050541, -0.0091150, -0.0051527, -0.0024214, 0.0025982
1: -0.0054875, -0.0043636, -0.0055085, -0.0043914, -0.0006827, 0.0007325
2: -0.0019280, 0.0063643, -0.0020832, 0.0061592, -0.0050371, 0.0054047
3: 0.0013721, 0.0024695, 0.0013516, 0.0024424, -0.0006666, 0.0007152
4: 0.0013356, 0.0075328, 0.0014888, 0.0076487, -0.0040392, 0.0037644
5: 0.9958773, 0.9975991, 0.9959198, 0.9976313, -0.0011222, 0.0010459
6: 0.0041415, 0.0057043, 0.0041801, 0.0057336, -0.0010186, 0.0009493
7: -0.0079262, -0.0020940, -0.0077820, -0.0019849, -0.0038013, 0.0035427
8: -0.0075631, -0.0030239, -0.0076480, -0.0031361, -0.0027573, 0.0029586
9: -0.0037488, -0.0033572, -0.0037392, -0.0033499, -0.0002552, 0.0002379

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005346, upper bound: 0.0005982
time: 2.03 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005344, upper bound: 0.0005983
time: 1.88 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 5.64 seconds
NS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005420, upper bound: 0.0004807
NS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005732, upper bound: 0.0005221
NS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005420, upper bound: 0.0005070
NS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005732, upper bound: 0.0005266
NS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005451, upper bound: 0.0004807
NS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005752, upper bound: 0.0005222
NS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005451, upper bound: 0.0005070
NS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005752, upper bound: 0.0005266
NS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005668, upper bound: 0.0005229
NS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005680, upper bound: 0.0005135
NS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005666, upper bound: 0.0005281
NS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005680, upper bound: 0.0005209
NS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005696, upper bound: 0.0005229
NS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005709, upper bound: 0.0005134
NS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005696, upper bound: 0.0005281
NS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005709, upper bound: 0.0005209
NS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005487
NS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005486
NS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005544
NS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005546
NS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005702, upper bound: 0.0005486
NS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005702, upper bound: 0.0005486
NS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005702, upper bound: 0.0005545
NS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005702, upper bound: 0.0005546
NS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005486
NS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005484
NS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005553
NS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005554
NS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005701, upper bound: 0.0005485
NS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005701, upper bound: 0.0005485
NS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005701, upper bound: 0.0005554
NS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005701, upper bound: 0.0005552
NS_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005331, upper bound: 0.0005928
NS_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005334, upper bound: 0.0005930
NS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005337, upper bound: 0.0005930
NS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005337, upper bound: 0.0005929
NS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005332, upper bound: 0.0005897
NS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005334, upper bound: 0.0005897
NS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005332, upper bound: 0.0005983
NS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005334, upper bound: 0.0005982
NS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005339, upper bound: 0.0005896
NS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005333, upper bound: 0.0005895
NS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005339, upper bound: 0.0005984
NS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005333, upper bound: 0.0005984
NS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005342, upper bound: 0.0005930
NS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005342, upper bound: 0.0005929
NS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005344, upper bound: 0.0005930
NS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005344, upper bound: 0.0005929
NS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005342, upper bound: 0.0005897
NS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005342, upper bound: 0.0005897
NS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005342, upper bound: 0.0005983
NS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005342, upper bound: 0.0005984
NS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005346, upper bound: 0.0005897
NS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005344, upper bound: 0.0005894
NS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005346, upper bound: 0.0005982
NS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.64
Output dim: 5, lower bound: -0.0005344, upper bound: 0.0005983

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0091009, -0.0051501, -0.0086729, -0.0052717, -0.0027137, 0.0024695
1: -0.0055046, -0.0043907, -0.0053839, -0.0044249, -0.0007651, 0.0006963
2: -0.0020540, 0.0061647, -0.0011635, 0.0059117, -0.0056450, 0.0051371
3: 0.0013555, 0.0024431, 0.0014733, 0.0024096, -0.0007470, 0.0006798
4: 0.0014848, 0.0076269, 0.0016738, 0.0069614, -0.0038392, 0.0042188
5: 0.9959188, 0.9976252, 0.9959713, 0.9974404, -0.0010666, 0.0011721
6: 0.0041791, 0.0057281, 0.0042268, 0.0055602, -0.0009682, 0.0010639
7: -0.0077858, -0.0020054, -0.0076079, -0.0026317, -0.0036131, 0.0039703
8: -0.0076321, -0.0031332, -0.0071446, -0.0032716, -0.0030901, 0.0028121
9: -0.0037394, -0.0033513, -0.0037275, -0.0033933, -0.0002426, 0.0002666

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005277, upper bound: 0.0005745
time: 1.53 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005207, upper bound: 0.0005763
time: 1.46 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0091009, -0.0051501, -0.0087807, -0.0052463, -0.0026506, 0.0024735
1: -0.0055046, -0.0043907, -0.0054143, -0.0044178, -0.0007473, 0.0006974
2: -0.0020540, 0.0061647, -0.0013877, 0.0059645, -0.0055137, 0.0051453
3: 0.0013555, 0.0024431, 0.0014437, 0.0024166, -0.0007297, 0.0006809
4: 0.0014848, 0.0076269, 0.0016344, 0.0071290, -0.0038453, 0.0041206
5: 0.9959188, 0.9976252, 0.9959604, 0.9974869, -0.0010683, 0.0011448
6: 0.0041791, 0.0057281, 0.0042168, 0.0056025, -0.0009697, 0.0010392
7: -0.0077858, -0.0020054, -0.0076450, -0.0024740, -0.0036189, 0.0038779
8: -0.0076321, -0.0031332, -0.0072674, -0.0032427, -0.0030182, 0.0028166
9: -0.0037394, -0.0033513, -0.0037300, -0.0033827, -0.0002430, 0.0002604

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005277, upper bound: 0.0005745
time: 2.08 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005207, upper bound: 0.0005763
time: 1.66 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0091067, -0.0051489, -0.0086767, -0.0052622, -0.0027467, 0.0024580
1: -0.0055062, -0.0043903, -0.0053849, -0.0044223, -0.0007744, 0.0006930
2: -0.0020659, 0.0061671, -0.0011715, 0.0059314, -0.0057138, 0.0051132
3: 0.0013539, 0.0024434, 0.0014723, 0.0024122, -0.0007561, 0.0006766
4: 0.0014830, 0.0076358, 0.0016591, 0.0069674, -0.0038213, 0.0042701
5: 0.9959182, 0.9976277, 0.9959672, 0.9974420, -0.0010617, 0.0011864
6: 0.0041786, 0.0057303, 0.0042231, 0.0055617, -0.0009637, 0.0010769
7: -0.0077875, -0.0019970, -0.0076217, -0.0026261, -0.0035962, 0.0040186
8: -0.0076386, -0.0031318, -0.0071490, -0.0032609, -0.0031277, 0.0027990
9: -0.0037395, -0.0033507, -0.0037284, -0.0033930, -0.0002415, 0.0002698

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005276, upper bound: 0.0005745
time: 2.12 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005209, upper bound: 0.0005762
time: 1.55 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0091067, -0.0051489, -0.0087943, -0.0052254, -0.0026848, 0.0024697
1: -0.0055062, -0.0043903, -0.0054181, -0.0044119, -0.0007570, 0.0006963
2: -0.0020659, 0.0061671, -0.0014160, 0.0060080, -0.0055850, 0.0051375
3: 0.0013539, 0.0024434, 0.0014399, 0.0024224, -0.0007391, 0.0006799
4: 0.0014830, 0.0076358, 0.0016019, 0.0071501, -0.0038395, 0.0041739
5: 0.9959182, 0.9976277, 0.9959512, 0.9974927, -0.0010667, 0.0011596
6: 0.0041786, 0.0057303, 0.0042086, 0.0056078, -0.0009683, 0.0010526
7: -0.0077875, -0.0019970, -0.0076756, -0.0024541, -0.0036134, 0.0039281
8: -0.0076386, -0.0031318, -0.0072829, -0.0032189, -0.0030572, 0.0028123
9: -0.0037395, -0.0033507, -0.0037320, -0.0033814, -0.0002426, 0.0002638

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005281, upper bound: 0.0005745
time: 1.53 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005209, upper bound: 0.0005763
time: 1.49 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0089267, -0.0049563, -0.0086237, -0.0052793, -0.0026943, 0.0027081
1: -0.0054554, -0.0043360, -0.0053700, -0.0044271, -0.0007596, 0.0007635
2: -0.0016916, 0.0065678, -0.0010611, 0.0058958, -0.0056048, 0.0056334
3: 0.0014034, 0.0024964, 0.0014869, 0.0024075, -0.0007417, 0.0007455
4: 0.0011835, 0.0073560, 0.0016857, 0.0068849, -0.0042101, 0.0041887
5: 0.9958351, 0.9975500, 0.9959745, 0.9974191, -0.0011697, 0.0011637
6: 0.0041031, 0.0056598, 0.0042298, 0.0055409, -0.0010617, 0.0010563
7: -0.0080693, -0.0022603, -0.0075967, -0.0027037, -0.0039621, 0.0039420
8: -0.0074337, -0.0029125, -0.0070886, -0.0032803, -0.0030681, 0.0030837
9: -0.0037585, -0.0033684, -0.0037267, -0.0033982, -0.0002660, 0.0002647

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005227, upper bound: 0.0005698
time: 1.66 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005133, upper bound: 0.0005708
time: 1.51 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0089267, -0.0049563, -0.0087358, -0.0052552, -0.0026326, 0.0027213
1: -0.0054554, -0.0043360, -0.0054016, -0.0044203, -0.0007422, 0.0007672
2: -0.0016916, 0.0065678, -0.0012944, 0.0059461, -0.0054764, 0.0056608
3: 0.0014034, 0.0024964, 0.0014560, 0.0024142, -0.0007247, 0.0007491
4: 0.0011835, 0.0073560, 0.0016481, 0.0070593, -0.0042305, 0.0040928
5: 0.9958351, 0.9975500, 0.9959642, 0.9974675, -0.0011754, 0.0011371
6: 0.0041031, 0.0056598, 0.0042203, 0.0055849, -0.0010669, 0.0010321
7: -0.0080693, -0.0022603, -0.0076320, -0.0025396, -0.0039814, 0.0038517
8: -0.0074337, -0.0029125, -0.0072163, -0.0032528, -0.0029978, 0.0030987
9: -0.0037585, -0.0033684, -0.0037291, -0.0033871, -0.0002673, 0.0002586

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005227, upper bound: 0.0005696
time: 2.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005134, upper bound: 0.0005708
time: 2.11 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0090363, -0.0050552, -0.0086237, -0.0052793, -0.0027344, 0.0025570
1: -0.0054863, -0.0043639, -0.0053700, -0.0044271, -0.0007709, 0.0007209
2: -0.0019196, 0.0063621, -0.0010611, 0.0058958, -0.0056881, 0.0053190
3: 0.0013733, 0.0024692, 0.0014869, 0.0024075, -0.0007527, 0.0007039
4: 0.0013373, 0.0075264, 0.0016857, 0.0068849, -0.0039751, 0.0042509
5: 0.9958777, 0.9975973, 0.9959745, 0.9974191, -0.0011044, 0.0011810
6: 0.0041419, 0.0057027, 0.0042298, 0.0055409, -0.0010025, 0.0010720
7: -0.0079246, -0.0020999, -0.0075967, -0.0027037, -0.0037410, 0.0040006
8: -0.0075585, -0.0030251, -0.0070886, -0.0032803, -0.0031137, 0.0029116
9: -0.0037487, -0.0033576, -0.0037267, -0.0033982, -0.0002512, 0.0002686

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005277, upper bound: 0.0005798
time: 1.68 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005207, upper bound: 0.0005817
time: 1.65 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0090363, -0.0050552, -0.0087358, -0.0052552, -0.0026703, 0.0025690
1: -0.0054863, -0.0043639, -0.0054016, -0.0044203, -0.0007528, 0.0007243
2: -0.0019196, 0.0063621, -0.0012944, 0.0059461, -0.0055547, 0.0053440
3: 0.0013733, 0.0024692, 0.0014560, 0.0024142, -0.0007351, 0.0007072
4: 0.0013373, 0.0075264, 0.0016481, 0.0070593, -0.0039937, 0.0041512
5: 0.9958777, 0.9975973, 0.9959642, 0.9974675, -0.0011096, 0.0011533
6: 0.0041419, 0.0057027, 0.0042203, 0.0055849, -0.0010072, 0.0010469
7: -0.0079246, -0.0020999, -0.0076320, -0.0025396, -0.0037586, 0.0039068
8: -0.0075585, -0.0030251, -0.0072163, -0.0032528, -0.0030406, 0.0029253
9: -0.0037487, -0.0033576, -0.0037291, -0.0033871, -0.0002524, 0.0002623

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005277, upper bound: 0.0005798
time: 2.25 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005207, upper bound: 0.0005817
time: 2.16 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0089324, -0.0049552, -0.0086294, -0.0052700, -0.0027180, 0.0027017
1: -0.0054570, -0.0043357, -0.0053716, -0.0044245, -0.0007663, 0.0007617
2: -0.0017033, 0.0065701, -0.0010731, 0.0059151, -0.0056541, 0.0056200
3: 0.0014019, 0.0024967, 0.0014853, 0.0024101, -0.0007482, 0.0007437
4: 0.0011818, 0.0073648, 0.0016713, 0.0068938, -0.0042000, 0.0042255
5: 0.9958346, 0.9975524, 0.9959705, 0.9974216, -0.0011669, 0.0011740
6: 0.0041027, 0.0056620, 0.0042261, 0.0055432, -0.0010592, 0.0010656
7: -0.0080709, -0.0022520, -0.0076103, -0.0026953, -0.0039527, 0.0039767
8: -0.0074401, -0.0029113, -0.0070951, -0.0032698, -0.0030951, 0.0030764
9: -0.0037586, -0.0033678, -0.0037276, -0.0033976, -0.0002654, 0.0002670

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005230, upper bound: 0.0005698
time: 1.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005137, upper bound: 0.0005709
time: 1.59 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0089324, -0.0049552, -0.0087477, -0.0052343, -0.0026591, 0.0027191
1: -0.0054570, -0.0043357, -0.0054050, -0.0044144, -0.0007497, 0.0007666
2: -0.0017033, 0.0065701, -0.0013191, 0.0059893, -0.0055315, 0.0056562
3: 0.0014019, 0.0024967, 0.0014527, 0.0024199, -0.0007320, 0.0007485
4: 0.0011818, 0.0073648, 0.0016158, 0.0070777, -0.0042271, 0.0041339
5: 0.9958346, 0.9975524, 0.9959551, 0.9974726, -0.0011744, 0.0011485
6: 0.0041027, 0.0056620, 0.0042122, 0.0055896, -0.0010660, 0.0010425
7: -0.0080709, -0.0022520, -0.0076625, -0.0025222, -0.0039782, 0.0038905
8: -0.0074401, -0.0029113, -0.0072298, -0.0032291, -0.0030279, 0.0030962
9: -0.0037586, -0.0033678, -0.0037311, -0.0033860, -0.0002671, 0.0002612

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005232, upper bound: 0.0005697
time: 2.08 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005135, upper bound: 0.0005708
time: 2.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0090404, -0.0050541, -0.0086294, -0.0052700, -0.0027704, 0.0025525
1: -0.0054875, -0.0043636, -0.0053716, -0.0044245, -0.0007811, 0.0007196
2: -0.0019280, 0.0063643, -0.0010731, 0.0059151, -0.0057631, 0.0053096
3: 0.0013721, 0.0024695, 0.0014853, 0.0024101, -0.0007626, 0.0007026
4: 0.0013356, 0.0075328, 0.0016713, 0.0068938, -0.0039681, 0.0043069
5: 0.9958773, 0.9975991, 0.9959705, 0.9974216, -0.0011024, 0.0011966
6: 0.0041415, 0.0057043, 0.0042261, 0.0055432, -0.0010007, 0.0010861
7: -0.0079262, -0.0020940, -0.0076103, -0.0026953, -0.0037344, 0.0040533
8: -0.0075631, -0.0030239, -0.0070951, -0.0032698, -0.0031547, 0.0029065
9: -0.0037488, -0.0033572, -0.0037276, -0.0033976, -0.0002508, 0.0002722

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005281, upper bound: 0.0005798
time: 1.64 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005210, upper bound: 0.0005817
time: 1.59 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.80 + 596.84 = 600.64 seconds
