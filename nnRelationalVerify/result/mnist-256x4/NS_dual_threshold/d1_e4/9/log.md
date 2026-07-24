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
execution time: IAR + RelationalAnalysis = 1.68 + 2.18 = 3.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0006905, upper bound: 0.0006905

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006638, upper bound: 0.0006238
time: 1.22 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006637, upper bound: 0.0006638
time: 1.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.67 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.67
Output dim: 5, lower bound: -0.0006638, upper bound: 0.0006238
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.67
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
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006510, upper bound: 0.0005965
time: 1.22 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006510, upper bound: 0.0006097
time: 1.33 seconds

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

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006239, upper bound: 0.0006638
time: 1.37 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006239, upper bound: 0.0006638
time: 1.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.63 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.63
Output dim: 5, lower bound: -0.0006510, upper bound: 0.0005965
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.63
Output dim: 5, lower bound: -0.0006510, upper bound: 0.0006097
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.63
Output dim: 5, lower bound: -0.0006239, upper bound: 0.0006638
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.63
Output dim: 5, lower bound: -0.0006239, upper bound: 0.0006638

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -0.0088058, -0.0052482, -0.0091402, -0.0051356, -0.0027237, 0.0028283
1: -0.0054213, -0.0044183, -0.0055156, -0.0043866, -0.0007679, 0.0007974
2: -0.0014399, 0.0059604, -0.0021356, 0.0061947, -0.0056659, 0.0058834
3: 0.0014367, 0.0024161, 0.0013447, 0.0024471, -0.0007498, 0.0007786
4: 0.0016374, 0.0071680, 0.0014623, 0.0076879, -0.0043969, 0.0042343
5: 0.9959612, 0.9974978, 0.9959125, 0.9976422, -0.0012216, 0.0011764
6: 0.0042176, 0.0056123, 0.0041734, 0.0057434, -0.0011088, 0.0010678
7: -0.0076421, -0.0024373, -0.0078070, -0.0019480, -0.0041380, 0.0039850
8: -0.0072959, -0.0032450, -0.0076767, -0.0031167, -0.0031015, 0.0032206
9: -0.0037298, -0.0033803, -0.0037408, -0.0033474, -0.0002779, 0.0002676

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006164, upper bound: 0.0005965
time: 1.55 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006164, upper bound: 0.0005964
time: 2.05 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -0.0089168, -0.0052119, -0.0091978, -0.0051312, -0.0027305, 0.0029672
1: -0.0054526, -0.0044081, -0.0055319, -0.0043853, -0.0007698, 0.0008366
2: -0.0016708, 0.0060359, -0.0022555, 0.0062038, -0.0056801, 0.0061724
3: 0.0014062, 0.0024261, 0.0013288, 0.0024483, -0.0007517, 0.0008168
4: 0.0015810, 0.0073405, 0.0014555, 0.0077775, -0.0046128, 0.0042449
5: 0.9959455, 0.9975457, 0.9959106, 0.9976670, -0.0012816, 0.0011794
6: 0.0042034, 0.0056558, 0.0041717, 0.0057660, -0.0011633, 0.0010705
7: -0.0076953, -0.0022749, -0.0078134, -0.0018636, -0.0043412, 0.0039949
8: -0.0074223, -0.0032036, -0.0077424, -0.0031117, -0.0031093, 0.0033788
9: -0.0037333, -0.0033694, -0.0037413, -0.0033418, -0.0002915, 0.0002683

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006163, upper bound: 0.0006097
time: 1.33 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006163, upper bound: 0.0006096
time: 1.88 seconds

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

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005966, upper bound: 0.0006510
time: 1.39 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006096, upper bound: 0.0006510
time: 1.29 seconds

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

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006096, upper bound: 0.0006320
time: 1.26 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006097, upper bound: 0.0006509
time: 2.00 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.00 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 5, lower bound: -0.0006164, upper bound: 0.0005965
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 5, lower bound: -0.0006164, upper bound: 0.0005964
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 5, lower bound: -0.0006163, upper bound: 0.0006097
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 5, lower bound: -0.0006163, upper bound: 0.0006096
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 5, lower bound: -0.0005966, upper bound: 0.0006510
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 5, lower bound: -0.0006096, upper bound: 0.0006510
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 5, lower bound: -0.0006096, upper bound: 0.0006320
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 5, lower bound: -0.0006097, upper bound: 0.0006509

## BFS NS instance: NS_A1_A1_B1

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

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006037, upper bound: 0.0005848
time: 1.42 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006050, upper bound: 0.0005848
time: 1.48 seconds

## BFS NS instance: NS_A1_A1_B2

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

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of NS_A1_A1_B2_A1

### Relational analysis result of NS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006049, upper bound: 0.0005841
time: 1.37 seconds

## Relational analysis of NS_A1_A1_B2_A2

### Relational analysis result of NS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006050, upper bound: 0.0005848
time: 1.76 seconds

## BFS NS instance: NS_A1_A2_B1

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

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of NS_A1_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006049, upper bound: 0.0005965
time: 1.28 seconds

## Relational analysis of NS_A1_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006049, upper bound: 0.0005987
time: 1.37 seconds

## BFS NS instance: NS_A1_A2_B2

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

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006049, upper bound: 0.0005965
time: 1.38 seconds

## Relational analysis of NS_A1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006049, upper bound: 0.0005987
time: 1.83 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0092275, -0.0051335, -0.0088058, -0.0052482, -0.0029962, 0.0027248
1: -0.0055402, -0.0043860, -0.0054213, -0.0044183, -0.0008448, 0.0007682
2: -0.0023173, 0.0061991, -0.0014399, 0.0059604, -0.0062328, 0.0056682
3: 0.0013206, 0.0024477, 0.0014367, 0.0024161, -0.0008248, 0.0007501
4: 0.0014590, 0.0078236, 0.0016374, 0.0071680, -0.0042360, 0.0046580
5: 0.9959116, 0.9976799, 0.9959612, 0.9974978, -0.0011769, 0.0012941
6: 0.0041726, 0.0057777, 0.0042176, 0.0056123, -0.0010683, 0.0011747
7: -0.0078101, -0.0018202, -0.0076421, -0.0024373, -0.0039866, 0.0043837
8: -0.0077762, -0.0031143, -0.0072959, -0.0032450, -0.0034118, 0.0031028
9: -0.0037411, -0.0033388, -0.0037298, -0.0033803, -0.0002677, 0.0002944

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005841, upper bound: 0.0006397
time: 1.38 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005848, upper bound: 0.0006397
time: 1.45 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0092837, -0.0051293, -0.0089168, -0.0052119, -0.0031349, 0.0027316
1: -0.0055561, -0.0043848, -0.0054526, -0.0044081, -0.0008838, 0.0007701
2: -0.0024341, 0.0062079, -0.0016708, 0.0060359, -0.0065212, 0.0056823
3: 0.0013052, 0.0024488, 0.0014062, 0.0024261, -0.0008630, 0.0007520
4: 0.0014524, 0.0079110, 0.0015810, 0.0073405, -0.0042466, 0.0048735
5: 0.9959098, 0.9977041, 0.9959455, 0.9975457, -0.0011798, 0.0013540
6: 0.0041710, 0.0057997, 0.0042034, 0.0056558, -0.0010709, 0.0012290
7: -0.0078162, -0.0017380, -0.0076953, -0.0022749, -0.0039965, 0.0045865
8: -0.0078402, -0.0031095, -0.0074223, -0.0032036, -0.0035697, 0.0031105
9: -0.0037415, -0.0033333, -0.0037333, -0.0033694, -0.0002684, 0.0003080

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005841, upper bound: 0.0006397
time: 2.05 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005986, upper bound: 0.0006396
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

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005965, upper bound: 0.0006197
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005987, upper bound: 0.0006197
time: 1.37 seconds

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

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005987, upper bound: 0.0006379
time: 1.73 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005987, upper bound: 0.0006397
time: 1.38 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.72 seconds
NS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0006037, upper bound: 0.0005848
NS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0006050, upper bound: 0.0005848
NS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0006049, upper bound: 0.0005841
NS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0006050, upper bound: 0.0005848
NS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0006049, upper bound: 0.0005965
NS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0006049, upper bound: 0.0005987
NS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0006049, upper bound: 0.0005965
NS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0006049, upper bound: 0.0005987
NS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0005841, upper bound: 0.0006397
NS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0005848, upper bound: 0.0006397
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0005841, upper bound: 0.0006397
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0005986, upper bound: 0.0006396
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0005965, upper bound: 0.0006197
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0005987, upper bound: 0.0006197
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0005987, upper bound: 0.0006379
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.72
Output dim: 5, lower bound: -0.0005987, upper bound: 0.0006397

## BFS NS instance: NS_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0087743, -0.0052540, -0.0088289, -0.0052438, -0.0024864, 0.0025043
1: -0.0054125, -0.0044199, -0.0054279, -0.0044171, -0.0007010, 0.0007061
2: -0.0013745, 0.0059485, -0.0014881, 0.0059698, -0.0051723, 0.0052094
3: 0.0014454, 0.0024145, 0.0014304, 0.0024173, -0.0006845, 0.0006894
4: 0.0016463, 0.0071191, 0.0016304, 0.0072040, -0.0038932, 0.0038654
5: 0.9959637, 0.9974842, 0.9959592, 0.9975077, -0.0010816, 0.0010739
6: 0.0042198, 0.0056000, 0.0042158, 0.0056214, -0.0009818, 0.0009748
7: -0.0076338, -0.0024833, -0.0076487, -0.0024034, -0.0036639, 0.0036378
8: -0.0072601, -0.0032515, -0.0073223, -0.0032399, -0.0028313, 0.0028516
9: -0.0037292, -0.0033834, -0.0037302, -0.0033780, -0.0002460, 0.0002443

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005740, upper bound: 0.0005602
time: 1.45 seconds

## Relational analysis of NS_A1_A1_B1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005766, upper bound: 0.0005604
time: 1.45 seconds

## BFS NS instance: NS_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0087774, -0.0052526, -0.0088366, -0.0052230, -0.0025222, 0.0025019
1: -0.0054133, -0.0044196, -0.0054300, -0.0044112, -0.0007111, 0.0007054
2: -0.0013809, 0.0059514, -0.0015041, 0.0060130, -0.0052467, 0.0052045
3: 0.0014446, 0.0024149, 0.0014282, 0.0024230, -0.0006943, 0.0006887
4: 0.0016441, 0.0071238, 0.0015981, 0.0072160, -0.0038895, 0.0039211
5: 0.9959630, 0.9974855, 0.9959502, 0.9975110, -0.0010806, 0.0010894
6: 0.0042193, 0.0056012, 0.0042077, 0.0056244, -0.0009809, 0.0009888
7: -0.0076358, -0.0024788, -0.0076791, -0.0023921, -0.0036604, 0.0036902
8: -0.0072636, -0.0032499, -0.0073311, -0.0032162, -0.0028721, 0.0028489
9: -0.0037294, -0.0033831, -0.0037323, -0.0033772, -0.0002458, 0.0002478

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005759, upper bound: 0.0005604
time: 1.52 seconds

## Relational analysis of NS_A1_A1_B1_B2_B2

### Relational analysis result of NS_A1_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005780, upper bound: 0.0005603
time: 1.56 seconds

## BFS NS instance: NS_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0087341, -0.0052679, -0.0091951, -0.0051386, -0.0026536, 0.0029383
1: -0.0054011, -0.0044239, -0.0055311, -0.0043874, -0.0007481, 0.0008284
2: -0.0012908, 0.0059195, -0.0022499, 0.0061885, -0.0055200, 0.0061123
3: 0.0014565, 0.0024107, 0.0013296, 0.0024462, -0.0007305, 0.0008089
4: 0.0016680, 0.0070565, 0.0014670, 0.0077733, -0.0045679, 0.0041253
5: 0.9959697, 0.9974667, 0.9959138, 0.9976659, -0.0012691, 0.0011461
6: 0.0042253, 0.0055842, 0.0041746, 0.0057650, -0.0011520, 0.0010403
7: -0.0076134, -0.0025422, -0.0078026, -0.0018676, -0.0042989, 0.0038823
8: -0.0072143, -0.0032673, -0.0077393, -0.0031201, -0.0030216, 0.0033459
9: -0.0037278, -0.0033873, -0.0037405, -0.0033420, -0.0002887, 0.0002607

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_A1_B1

### Relational analysis result of NS_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006087, upper bound: 0.0005570
time: 1.33 seconds

## Relational analysis of NS_A1_A1_B2_A1_B2

### Relational analysis result of NS_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006144, upper bound: 0.0005570
time: 1.36 seconds

## BFS NS instance: NS_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0087390, -0.0052584, -0.0092002, -0.0051376, -0.0026448, 0.0029564
1: -0.0054025, -0.0044212, -0.0055325, -0.0043871, -0.0007457, 0.0008335
2: -0.0013010, 0.0059393, -0.0022604, 0.0061907, -0.0055017, 0.0061499
3: 0.0014551, 0.0024133, 0.0013282, 0.0024465, -0.0007281, 0.0008138
4: 0.0016532, 0.0070642, 0.0014653, 0.0077811, -0.0045960, 0.0041117
5: 0.9959656, 0.9974689, 0.9959133, 0.9976681, -0.0012769, 0.0011423
6: 0.0042216, 0.0055862, 0.0041742, 0.0057670, -0.0011590, 0.0010369
7: -0.0076273, -0.0025350, -0.0078041, -0.0018602, -0.0043254, 0.0038695
8: -0.0072199, -0.0032565, -0.0077451, -0.0031189, -0.0030117, 0.0033664
9: -0.0037288, -0.0033868, -0.0037406, -0.0033415, -0.0002904, 0.0002598

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005760, upper bound: 0.0005577
time: 2.02 seconds

## Relational analysis of NS_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006144, upper bound: 0.0005577
time: 1.82 seconds

## BFS NS instance: NS_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0088412, -0.0052426, -0.0089244, -0.0052139, -0.0024788, 0.0026414
1: -0.0054313, -0.0044167, -0.0054548, -0.0044086, -0.0006989, 0.0007447
2: -0.0015136, 0.0059722, -0.0016868, 0.0060320, -0.0051563, 0.0054947
3: 0.0014270, 0.0024176, 0.0014041, 0.0024255, -0.0006824, 0.0007271
4: 0.0016286, 0.0072230, 0.0015840, 0.0073525, -0.0041064, 0.0038535
5: 0.9959587, 0.9975130, 0.9959463, 0.9975490, -0.0011409, 0.0010706
6: 0.0042154, 0.0056262, 0.0042041, 0.0056589, -0.0010356, 0.0009718
7: -0.0076504, -0.0023855, -0.0076925, -0.0022636, -0.0038646, 0.0036266
8: -0.0073362, -0.0032385, -0.0074311, -0.0032058, -0.0028226, 0.0030078
9: -0.0037303, -0.0033768, -0.0037332, -0.0033686, -0.0002595, 0.0002435

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B1_A1_B1

### Relational analysis result of NS_A1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005758, upper bound: 0.0005767
time: 1.32 seconds

## Relational analysis of NS_A1_A2_B1_A1_B2

### Relational analysis result of NS_A1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005780, upper bound: 0.0005768
time: 1.43 seconds

## BFS NS instance: NS_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0088543, -0.0052216, -0.0089295, -0.0052124, -0.0024744, 0.0026815
1: -0.0054350, -0.0044108, -0.0054562, -0.0044082, -0.0006976, 0.0007560
2: -0.0015409, 0.0060158, -0.0016973, 0.0060349, -0.0051473, 0.0055780
3: 0.0014234, 0.0024234, 0.0014027, 0.0024259, -0.0006812, 0.0007382
4: 0.0015960, 0.0072434, 0.0015818, 0.0073603, -0.0041686, 0.0038468
5: 0.9959496, 0.9975187, 0.9959457, 0.9975511, -0.0011582, 0.0010688
6: 0.0042072, 0.0056314, 0.0042036, 0.0056608, -0.0010513, 0.0009701
7: -0.0076811, -0.0023663, -0.0076945, -0.0022563, -0.0039231, 0.0036203
8: -0.0073512, -0.0032147, -0.0074368, -0.0032042, -0.0028177, 0.0030534
9: -0.0037324, -0.0033755, -0.0037333, -0.0033681, -0.0002634, 0.0002431

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005760, upper bound: 0.0005781
time: 1.29 seconds

## Relational analysis of NS_A1_A2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005780, upper bound: 0.0005780
time: 1.45 seconds

## BFS NS instance: NS_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0088412, -0.0052426, -0.0092507, -0.0051343, -0.0026583, 0.0030604
1: -0.0054313, -0.0044167, -0.0055468, -0.0043862, -0.0007495, 0.0008628
2: -0.0015136, 0.0059722, -0.0023655, 0.0061974, -0.0055297, 0.0063662
3: 0.0014270, 0.0024176, 0.0013143, 0.0024474, -0.0007318, 0.0008425
4: 0.0016286, 0.0072230, 0.0014603, 0.0078597, -0.0047577, 0.0041326
5: 0.9959587, 0.9975130, 0.9959120, 0.9976899, -0.0013218, 0.0011482
6: 0.0042154, 0.0056262, 0.0041729, 0.0057868, -0.0011998, 0.0010422
7: -0.0076504, -0.0023855, -0.0078088, -0.0017863, -0.0044775, 0.0038892
8: -0.0073362, -0.0032385, -0.0078026, -0.0031153, -0.0030270, 0.0034848
9: -0.0037303, -0.0033768, -0.0037410, -0.0033366, -0.0003007, 0.0002612

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B2_A1_B1

### Relational analysis result of NS_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006087, upper bound: 0.0005699
time: 1.40 seconds

## Relational analysis of NS_A1_A2_B2_A1_B2

### Relational analysis result of NS_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006145, upper bound: 0.0005701
time: 1.41 seconds

## BFS NS instance: NS_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0088543, -0.0052216, -0.0092567, -0.0051332, -0.0026547, 0.0030963
1: -0.0054350, -0.0044108, -0.0055485, -0.0043859, -0.0007485, 0.0008730
2: -0.0015409, 0.0060158, -0.0023780, 0.0061997, -0.0055223, 0.0064410
3: 0.0014234, 0.0024234, 0.0013126, 0.0024477, -0.0007308, 0.0008524
4: 0.0015960, 0.0072434, 0.0014586, 0.0078691, -0.0048136, 0.0041270
5: 0.9959496, 0.9975187, 0.9959114, 0.9976925, -0.0013374, 0.0011466
6: 0.0042072, 0.0056314, 0.0041725, 0.0057891, -0.0012139, 0.0010408
7: -0.0076811, -0.0023663, -0.0078104, -0.0017775, -0.0045301, 0.0038840
8: -0.0073512, -0.0032147, -0.0078095, -0.0031140, -0.0030229, 0.0035258
9: -0.0037324, -0.0033755, -0.0037411, -0.0033360, -0.0003042, 0.0002608

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006087, upper bound: 0.0005722
time: 1.49 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006144, upper bound: 0.0005722
time: 1.47 seconds

## BFS NS instance: NS_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0091951, -0.0051386, -0.0087341, -0.0052679, -0.0029383, 0.0026536
1: -0.0055311, -0.0043874, -0.0054011, -0.0044239, -0.0008284, 0.0007481
2: -0.0022499, 0.0061885, -0.0012908, 0.0059195, -0.0061123, 0.0055200
3: 0.0013296, 0.0024462, 0.0014565, 0.0024107, -0.0008089, 0.0007305
4: 0.0014670, 0.0077733, 0.0016680, 0.0070565, -0.0041253, 0.0045679
5: 0.9959138, 0.9976659, 0.9959697, 0.9974667, -0.0011461, 0.0012691
6: 0.0041746, 0.0057650, 0.0042253, 0.0055842, -0.0010403, 0.0011520
7: -0.0078026, -0.0018676, -0.0076134, -0.0025422, -0.0038823, 0.0042989
8: -0.0077393, -0.0031201, -0.0072143, -0.0032673, -0.0033459, 0.0030216
9: -0.0037405, -0.0033420, -0.0037278, -0.0033873, -0.0002607, 0.0002887

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005570, upper bound: 0.0006088
time: 1.41 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005571, upper bound: 0.0006146
time: 1.43 seconds

## BFS NS instance: NS_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0092002, -0.0051376, -0.0087390, -0.0052584, -0.0029564, 0.0026448
1: -0.0055325, -0.0043871, -0.0054025, -0.0044212, -0.0008335, 0.0007457
2: -0.0022604, 0.0061907, -0.0013010, 0.0059393, -0.0061499, 0.0055017
3: 0.0013282, 0.0024465, 0.0014551, 0.0024133, -0.0008138, 0.0007281
4: 0.0014653, 0.0077811, 0.0016532, 0.0070642, -0.0041117, 0.0045960
5: 0.9959133, 0.9976681, 0.9959656, 0.9974689, -0.0011423, 0.0012769
6: 0.0041742, 0.0057670, 0.0042216, 0.0055862, -0.0010369, 0.0011590
7: -0.0078041, -0.0018602, -0.0076273, -0.0025350, -0.0038695, 0.0043254
8: -0.0077451, -0.0031189, -0.0072199, -0.0032565, -0.0033664, 0.0030117
9: -0.0037406, -0.0033415, -0.0037288, -0.0033868, -0.0002598, 0.0002904

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005576, upper bound: 0.0006088
time: 1.46 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005577, upper bound: 0.0006145
time: 1.46 seconds

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0092507, -0.0051343, -0.0088412, -0.0052426, -0.0030604, 0.0026583
1: -0.0055468, -0.0043862, -0.0054313, -0.0044167, -0.0008628, 0.0007495
2: -0.0023655, 0.0061974, -0.0015136, 0.0059722, -0.0063662, 0.0055297
3: 0.0013143, 0.0024474, 0.0014270, 0.0024176, -0.0008425, 0.0007318
4: 0.0014603, 0.0078597, 0.0016286, 0.0072230, -0.0041326, 0.0047577
5: 0.9959120, 0.9976899, 0.9959587, 0.9975130, -0.0011482, 0.0013218
6: 0.0041729, 0.0057868, 0.0042154, 0.0056262, -0.0010422, 0.0011998
7: -0.0078088, -0.0017863, -0.0076504, -0.0023855, -0.0038892, 0.0044775
8: -0.0078026, -0.0031153, -0.0073362, -0.0032385, -0.0034848, 0.0030270
9: -0.0037410, -0.0033366, -0.0037303, -0.0033768, -0.0002612, 0.0003007

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005701, upper bound: 0.0006088
time: 1.34 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005701, upper bound: 0.0006145
time: 1.55 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0092567, -0.0051332, -0.0088543, -0.0052216, -0.0030963, 0.0026547
1: -0.0055485, -0.0043859, -0.0054350, -0.0044108, -0.0008730, 0.0007485
2: -0.0023780, 0.0061997, -0.0015409, 0.0060158, -0.0064410, 0.0055223
3: 0.0013126, 0.0024477, 0.0014234, 0.0024234, -0.0008524, 0.0007308
4: 0.0014586, 0.0078691, 0.0015960, 0.0072434, -0.0041270, 0.0048136
5: 0.9959114, 0.9976925, 0.9959496, 0.9975187, -0.0011466, 0.0013374
6: 0.0041725, 0.0057891, 0.0042072, 0.0056314, -0.0010408, 0.0012139
7: -0.0078104, -0.0017775, -0.0076811, -0.0023663, -0.0038840, 0.0045301
8: -0.0078095, -0.0031140, -0.0073512, -0.0032147, -0.0035258, 0.0030229
9: -0.0037411, -0.0033360, -0.0037324, -0.0033755, -0.0002608, 0.0003042

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005722, upper bound: 0.0006088
time: 1.43 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005722, upper bound: 0.0006146
time: 1.31 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0090960, -0.0051779, -0.0091522, -0.0051599, -0.0025801, 0.0026008
1: -0.0055031, -0.0043985, -0.0055190, -0.0043934, -0.0007274, 0.0007333
2: -0.0020436, 0.0061068, -0.0021606, 0.0061442, -0.0053672, 0.0054102
3: 0.0013569, 0.0024354, 0.0013414, 0.0024404, -0.0007103, 0.0007159
4: 0.0015280, 0.0076191, 0.0015000, 0.0077065, -0.0040432, 0.0040111
5: 0.9959308, 0.9976231, 0.9959230, 0.9976474, -0.0011233, 0.0011144
6: 0.0041900, 0.0057261, 0.0041830, 0.0057481, -0.0010196, 0.0010115
7: -0.0077451, -0.0020127, -0.0077714, -0.0019304, -0.0038051, 0.0037749
8: -0.0076264, -0.0031649, -0.0076904, -0.0031444, -0.0029380, 0.0029615
9: -0.0037367, -0.0033518, -0.0037385, -0.0033462, -0.0002555, 0.0002535

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005691, upper bound: 0.0005945
time: 1.39 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005704, upper bound: 0.0005944
time: 1.37 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0091003, -0.0051769, -0.0091632, -0.0051431, -0.0026165, 0.0025923
1: -0.0055044, -0.0043982, -0.0055221, -0.0043887, -0.0007377, 0.0007309
2: -0.0020526, 0.0061089, -0.0021834, 0.0061792, -0.0054428, 0.0053924
3: 0.0013557, 0.0024357, 0.0013384, 0.0024450, -0.0007203, 0.0007136
4: 0.0015265, 0.0076259, 0.0014739, 0.0077236, -0.0040300, 0.0040676
5: 0.9959304, 0.9976249, 0.9959157, 0.9976521, -0.0011196, 0.0011301
6: 0.0041896, 0.0057278, 0.0041764, 0.0057524, -0.0010163, 0.0010258
7: -0.0077466, -0.0020064, -0.0077960, -0.0019144, -0.0037926, 0.0038281
8: -0.0076313, -0.0031637, -0.0077029, -0.0031252, -0.0029794, 0.0029518
9: -0.0037368, -0.0033513, -0.0037401, -0.0033452, -0.0002547, 0.0002570

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005716, upper bound: 0.0005944
time: 1.51 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005722, upper bound: 0.0005945
time: 1.59 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0091676, -0.0051593, -0.0092507, -0.0051343, -0.0025717, 0.0027372
1: -0.0055234, -0.0043933, -0.0055468, -0.0043862, -0.0007251, 0.0007717
2: -0.0021927, 0.0061454, -0.0023655, 0.0061974, -0.0053497, 0.0056939
3: 0.0013371, 0.0024405, 0.0013143, 0.0024474, -0.0007079, 0.0007535
4: 0.0014992, 0.0077306, 0.0014603, 0.0078597, -0.0042553, 0.0039980
5: 0.9959227, 0.9976540, 0.9959120, 0.9976899, -0.0011822, 0.0011108
6: 0.0041827, 0.0057542, 0.0041729, 0.0057868, -0.0010731, 0.0010082
7: -0.0077722, -0.0019078, -0.0078088, -0.0017863, -0.0040047, 0.0037626
8: -0.0077080, -0.0031437, -0.0078026, -0.0031153, -0.0029284, 0.0031168
9: -0.0037385, -0.0033447, -0.0037410, -0.0033366, -0.0002689, 0.0002526

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005716, upper bound: 0.0006132
time: 1.54 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005723, upper bound: 0.0006131
time: 1.34 seconds

## BFS NS instance: NS_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0091806, -0.0051422, -0.0092567, -0.0051332, -0.0025649, 0.0027769
1: -0.0055270, -0.0043884, -0.0055485, -0.0043859, -0.0007231, 0.0007829
2: -0.0022197, 0.0061810, -0.0023780, 0.0061997, -0.0053355, 0.0057766
3: 0.0013335, 0.0024453, 0.0013126, 0.0024477, -0.0007061, 0.0007644
4: 0.0014726, 0.0077508, 0.0014586, 0.0078691, -0.0043170, 0.0039874
5: 0.9959154, 0.9976596, 0.9959114, 0.9976925, -0.0011994, 0.0011078
6: 0.0041760, 0.0057593, 0.0041725, 0.0057891, -0.0010887, 0.0010056
7: -0.0077973, -0.0018888, -0.0078104, -0.0017775, -0.0040628, 0.0037526
8: -0.0077228, -0.0031242, -0.0078095, -0.0031140, -0.0029207, 0.0031621
9: -0.0037402, -0.0033435, -0.0037411, -0.0033360, -0.0002728, 0.0002520

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005716, upper bound: 0.0006146
time: 1.52 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005724, upper bound: 0.0006146
time: 1.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.57 seconds
NS_A1_A1_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005740, upper bound: 0.0005602
NS_A1_A1_B1_B1_B2, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005766, upper bound: 0.0005604
NS_A1_A1_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005759, upper bound: 0.0005604
NS_A1_A1_B1_B2_B2, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005780, upper bound: 0.0005603
NS_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0006087, upper bound: 0.0005570
NS_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0006144, upper bound: 0.0005570
NS_A1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005760, upper bound: 0.0005577
NS_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0006144, upper bound: 0.0005577
NS_A1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005758, upper bound: 0.0005767
NS_A1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005780, upper bound: 0.0005768
NS_A1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005760, upper bound: 0.0005781
NS_A1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005780, upper bound: 0.0005780
NS_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0006087, upper bound: 0.0005699
NS_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0006145, upper bound: 0.0005701
NS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0006087, upper bound: 0.0005722
NS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0006144, upper bound: 0.0005722
NS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005570, upper bound: 0.0006088
NS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005571, upper bound: 0.0006146
NS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005576, upper bound: 0.0006088
NS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005577, upper bound: 0.0006145
NS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005701, upper bound: 0.0006088
NS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005701, upper bound: 0.0006145
NS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005722, upper bound: 0.0006088
NS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005722, upper bound: 0.0006146
NS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005691, upper bound: 0.0005945
NS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005704, upper bound: 0.0005944
NS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005716, upper bound: 0.0005944
NS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005722, upper bound: 0.0005945
NS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005716, upper bound: 0.0006132
NS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005723, upper bound: 0.0006131
NS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005716, upper bound: 0.0006146
NS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 5, lower bound: -0.0005724, upper bound: 0.0006146

## BFS NS instance: NS_A1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0087207, -0.0052700, -0.0091306, -0.0051493, -0.0026022, 0.0028346
1: -0.0053974, -0.0044245, -0.0055129, -0.0043904, -0.0007336, 0.0007992
2: -0.0012630, 0.0059151, -0.0021156, 0.0061663, -0.0054130, 0.0058965
3: 0.0014602, 0.0024101, 0.0013473, 0.0024433, -0.0007163, 0.0007803
4: 0.0016713, 0.0070358, 0.0014836, 0.0076730, -0.0044067, 0.0040454
5: 0.9959706, 0.9974610, 0.9959184, 0.9976380, -0.0012243, 0.0011239
6: 0.0042261, 0.0055790, 0.0041788, 0.0057397, -0.0011113, 0.0010202
7: -0.0076103, -0.0025617, -0.0077869, -0.0019620, -0.0041472, 0.0038071
8: -0.0071991, -0.0032698, -0.0076658, -0.0031323, -0.0029631, 0.0032278
9: -0.0037276, -0.0033886, -0.0037395, -0.0033484, -0.0002785, 0.0002556

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005848, upper bound: 0.0005337
time: 1.47 seconds

## Relational analysis of NS_A1_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005930, upper bound: 0.0005389
time: 1.36 seconds

## BFS NS instance: NS_A1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0086712, -0.0052778, -0.0090653, -0.0050543, -0.0026871, 0.0028538
1: -0.0053834, -0.0044267, -0.0054945, -0.0043636, -0.0007576, 0.0008046
2: -0.0011601, 0.0058990, -0.0019798, 0.0063640, -0.0055897, 0.0059365
3: 0.0014738, 0.0024079, 0.0013653, 0.0024695, -0.0007397, 0.0007856
4: 0.0016833, 0.0069589, 0.0013358, 0.0075714, -0.0044365, 0.0041774
5: 0.9959739, 0.9974396, 0.9958774, 0.9976098, -0.0012326, 0.0011606
6: 0.0042292, 0.0055596, 0.0041415, 0.0057141, -0.0011188, 0.0010535
7: -0.0075990, -0.0026341, -0.0079260, -0.0020576, -0.0041753, 0.0039314
8: -0.0071428, -0.0032786, -0.0075915, -0.0030241, -0.0030598, 0.0032496
9: -0.0037269, -0.0033935, -0.0037488, -0.0033548, -0.0002804, 0.0002640

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0005339
time: 1.39 seconds

## Relational analysis of NS_A1_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005388
time: 1.40 seconds

## BFS NS instance: NS_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0086776, -0.0052685, -0.0090692, -0.0050532, -0.0026839, 0.0028735
1: -0.0053852, -0.0044240, -0.0054956, -0.0043633, -0.0007567, 0.0008101
2: -0.0011732, 0.0059184, -0.0019879, 0.0063662, -0.0055831, 0.0059775
3: 0.0014720, 0.0024105, 0.0013642, 0.0024698, -0.0007388, 0.0007910
4: 0.0016688, 0.0069687, 0.0013342, 0.0075775, -0.0044672, 0.0041725
5: 0.9959698, 0.9974424, 0.9958769, 0.9976115, -0.0012411, 0.0011592
6: 0.0042255, 0.0055621, 0.0041411, 0.0057156, -0.0011266, 0.0010522
7: -0.0076126, -0.0026248, -0.0079275, -0.0020519, -0.0042041, 0.0039268
8: -0.0071499, -0.0032680, -0.0075959, -0.0030229, -0.0030562, 0.0032721
9: -0.0037278, -0.0033929, -0.0037489, -0.0033544, -0.0002823, 0.0002637

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0005342
time: 1.44 seconds

## Relational analysis of NS_A1_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005391
time: 1.33 seconds

## BFS NS instance: NS_A1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0088272, -0.0052449, -0.0091861, -0.0051452, -0.0026066, 0.0029570
1: -0.0054274, -0.0044174, -0.0055286, -0.0043893, -0.0007349, 0.0008337
2: -0.0014845, 0.0059675, -0.0022311, 0.0061747, -0.0054222, 0.0061511
3: 0.0014308, 0.0024170, 0.0013321, 0.0024444, -0.0007175, 0.0008140
4: 0.0016322, 0.0072013, 0.0014773, 0.0077592, -0.0045969, 0.0040522
5: 0.9959598, 0.9975070, 0.9959167, 0.9976619, -0.0012772, 0.0011258
6: 0.0042163, 0.0056207, 0.0041772, 0.0057614, -0.0011593, 0.0010219
7: -0.0076471, -0.0024059, -0.0077928, -0.0018809, -0.0043262, 0.0038136
8: -0.0073203, -0.0032411, -0.0077290, -0.0031277, -0.0029681, 0.0033671
9: -0.0037301, -0.0033782, -0.0037399, -0.0033429, -0.0002905, 0.0002561

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_A2_B2_A1_B1_B1

### Relational analysis result of NS_A1_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005848, upper bound: 0.0005487
time: 1.38 seconds

## Relational analysis of NS_A1_A2_B2_A1_B1_B2

### Relational analysis result of NS_A1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005928, upper bound: 0.0005531
time: 1.41 seconds

## BFS NS instance: NS_A1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0087822, -0.0052538, -0.0091217, -0.0050506, -0.0026976, 0.0029741
1: -0.0054147, -0.0044199, -0.0055104, -0.0043626, -0.0007606, 0.0008385
2: -0.0013910, 0.0059488, -0.0020971, 0.0063716, -0.0056116, 0.0061867
3: 0.0014432, 0.0024145, 0.0013498, 0.0024705, -0.0007426, 0.0008187
4: 0.0016461, 0.0071314, 0.0013301, 0.0076591, -0.0046235, 0.0041938
5: 0.9959636, 0.9974876, 0.9958758, 0.9976342, -0.0012846, 0.0011652
6: 0.0042198, 0.0056031, 0.0041401, 0.0057362, -0.0011660, 0.0010576
7: -0.0076340, -0.0024717, -0.0079314, -0.0019750, -0.0043513, 0.0039468
8: -0.0072691, -0.0032513, -0.0076557, -0.0030199, -0.0030718, 0.0033866
9: -0.0037292, -0.0033826, -0.0037492, -0.0033492, -0.0002922, 0.0002650

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_A2_B2_A1_B2_B1

### Relational analysis result of NS_A1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0005486
time: 1.41 seconds

## Relational analysis of NS_A1_A2_B2_A1_B2_B2

### Relational analysis result of NS_A1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005532
time: 1.49 seconds

## BFS NS instance: NS_A1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0088406, -0.0052239, -0.0091923, -0.0051442, -0.0026037, 0.0029937
1: -0.0054312, -0.0044115, -0.0055303, -0.0043890, -0.0007341, 0.0008440
2: -0.0015124, 0.0060110, -0.0022439, 0.0061769, -0.0054161, 0.0062275
3: 0.0014272, 0.0024228, 0.0013303, 0.0024447, -0.0007167, 0.0008241
4: 0.0015996, 0.0072221, 0.0014756, 0.0077688, -0.0046541, 0.0040477
5: 0.9959506, 0.9975128, 0.9959162, 0.9976647, -0.0012930, 0.0011246
6: 0.0042081, 0.0056260, 0.0041768, 0.0057639, -0.0011737, 0.0010208
7: -0.0076777, -0.0023863, -0.0077944, -0.0018718, -0.0043800, 0.0038093
8: -0.0073356, -0.0032173, -0.0077361, -0.0031265, -0.0029648, 0.0034090
9: -0.0037322, -0.0033769, -0.0037400, -0.0033423, -0.0002941, 0.0002558

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_A2_B2_A2_B1_B1

### Relational analysis result of NS_A1_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005848, upper bound: 0.0005512
time: 1.48 seconds

## Relational analysis of NS_A1_A2_B2_A2_B1_B2

### Relational analysis result of NS_A1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005930, upper bound: 0.0005553
time: 1.53 seconds

## BFS NS instance: NS_A1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0087940, -0.0052330, -0.0091256, -0.0050495, -0.0026976, 0.0030117
1: -0.0054180, -0.0044140, -0.0055115, -0.0043623, -0.0007605, 0.0008491
2: -0.0014154, 0.0059922, -0.0021053, 0.0063739, -0.0056115, 0.0062649
3: 0.0014400, 0.0024203, 0.0013487, 0.0024708, -0.0007426, 0.0008291
4: 0.0016136, 0.0071497, 0.0013284, 0.0076652, -0.0046820, 0.0041937
5: 0.9959545, 0.9974926, 0.9958752, 0.9976359, -0.0013008, 0.0011651
6: 0.0042116, 0.0056077, 0.0041397, 0.0057377, -0.0011807, 0.0010576
7: -0.0076645, -0.0024545, -0.0079329, -0.0019693, -0.0044063, 0.0039467
8: -0.0072825, -0.0032276, -0.0076602, -0.0030187, -0.0030717, 0.0034294
9: -0.0037313, -0.0033814, -0.0037493, -0.0033489, -0.0002959, 0.0002650

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005943, upper bound: 0.0005486
time: 1.50 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005553
time: 1.86 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0091306, -0.0051493, -0.0087207, -0.0052700, -0.0028346, 0.0026022
1: -0.0055129, -0.0043904, -0.0053974, -0.0044245, -0.0007992, 0.0007336
2: -0.0021156, 0.0061663, -0.0012630, 0.0059151, -0.0058965, 0.0054130
3: 0.0013473, 0.0024433, 0.0014602, 0.0024101, -0.0007803, 0.0007163
4: 0.0014836, 0.0076730, 0.0016713, 0.0070358, -0.0040454, 0.0044067
5: 0.9959184, 0.9976380, 0.9959706, 0.9974610, -0.0011239, 0.0012243
6: 0.0041788, 0.0057397, 0.0042261, 0.0055790, -0.0010202, 0.0011113
7: -0.0077869, -0.0019620, -0.0076103, -0.0025617, -0.0038071, 0.0041472
8: -0.0076658, -0.0031323, -0.0071991, -0.0032698, -0.0032278, 0.0029631
9: -0.0037395, -0.0033484, -0.0037276, -0.0033886, -0.0002556, 0.0002785

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005339, upper bound: 0.0005849
time: 1.44 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005389, upper bound: 0.0005930
time: 1.44 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0090653, -0.0050543, -0.0086712, -0.0052778, -0.0028538, 0.0026871
1: -0.0054945, -0.0043636, -0.0053834, -0.0044267, -0.0008046, 0.0007576
2: -0.0019798, 0.0063640, -0.0011601, 0.0058990, -0.0059365, 0.0055897
3: 0.0013653, 0.0024695, 0.0014738, 0.0024079, -0.0007856, 0.0007397
4: 0.0013358, 0.0075714, 0.0016833, 0.0069589, -0.0041774, 0.0044365
5: 0.9958774, 0.9976098, 0.9959739, 0.9974396, -0.0011606, 0.0012326
6: 0.0041415, 0.0057141, 0.0042292, 0.0055596, -0.0010535, 0.0011188
7: -0.0079260, -0.0020576, -0.0075990, -0.0026341, -0.0039314, 0.0041753
8: -0.0075915, -0.0030241, -0.0071428, -0.0032786, -0.0032496, 0.0030598
9: -0.0037488, -0.0033548, -0.0037269, -0.0033935, -0.0002640, 0.0002804

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005339, upper bound: 0.0005897
time: 1.54 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005389, upper bound: 0.0005984
time: 1.56 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0091363, -0.0051482, -0.0087255, -0.0052605, -0.0028532, 0.0025934
1: -0.0055145, -0.0043901, -0.0053987, -0.0044218, -0.0008044, 0.0007312
2: -0.0021275, 0.0061685, -0.0012729, 0.0059349, -0.0059352, 0.0053948
3: 0.0013458, 0.0024436, 0.0014588, 0.0024127, -0.0007854, 0.0007139
4: 0.0014819, 0.0076818, 0.0016565, 0.0070432, -0.0040318, 0.0044356
5: 0.9959179, 0.9976404, 0.9959665, 0.9974630, -0.0011201, 0.0012323
6: 0.0041784, 0.0057419, 0.0042224, 0.0055809, -0.0010167, 0.0011186
7: -0.0077885, -0.0019537, -0.0076242, -0.0025547, -0.0037943, 0.0041744
8: -0.0076723, -0.0031311, -0.0072045, -0.0032589, -0.0032489, 0.0029531
9: -0.0037396, -0.0033478, -0.0037286, -0.0033882, -0.0002548, 0.0002803

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005343, upper bound: 0.0005849
time: 1.63 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005390, upper bound: 0.0005930
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0090692, -0.0050532, -0.0086776, -0.0052685, -0.0028735, 0.0026839
1: -0.0054956, -0.0043633, -0.0053852, -0.0044240, -0.0008101, 0.0007567
2: -0.0019879, 0.0063662, -0.0011732, 0.0059184, -0.0059775, 0.0055831
3: 0.0013642, 0.0024698, 0.0014720, 0.0024105, -0.0007910, 0.0007388
4: 0.0013342, 0.0075775, 0.0016688, 0.0069687, -0.0041725, 0.0044672
5: 0.9958769, 0.9976115, 0.9959698, 0.9974424, -0.0011592, 0.0012411
6: 0.0041411, 0.0057156, 0.0042255, 0.0055621, -0.0010522, 0.0011266
7: -0.0079275, -0.0020519, -0.0076126, -0.0026248, -0.0039268, 0.0042041
8: -0.0075959, -0.0030229, -0.0071499, -0.0032680, -0.0032721, 0.0030562
9: -0.0037489, -0.0033544, -0.0037278, -0.0033929, -0.0002637, 0.0002823

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005341, upper bound: 0.0005897
time: 1.66 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005390, upper bound: 0.0005983
time: 1.67 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0091861, -0.0051452, -0.0088272, -0.0052449, -0.0029570, 0.0026066
1: -0.0055286, -0.0043893, -0.0054274, -0.0044174, -0.0008337, 0.0007349
2: -0.0022311, 0.0061747, -0.0014845, 0.0059675, -0.0061511, 0.0054222
3: 0.0013321, 0.0024444, 0.0014308, 0.0024170, -0.0008140, 0.0007175
4: 0.0014773, 0.0077592, 0.0016322, 0.0072013, -0.0040522, 0.0045969
5: 0.9959167, 0.9976619, 0.9959598, 0.9975070, -0.0011258, 0.0012772
6: 0.0041772, 0.0057614, 0.0042163, 0.0056207, -0.0010219, 0.0011593
7: -0.0077928, -0.0018809, -0.0076471, -0.0024059, -0.0038136, 0.0043262
8: -0.0077290, -0.0031277, -0.0073203, -0.0032411, -0.0033671, 0.0029681
9: -0.0037399, -0.0033429, -0.0037301, -0.0033782, -0.0002561, 0.0002905

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005485, upper bound: 0.0005849
time: 1.84 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005532, upper bound: 0.0005930
time: 1.66 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0091217, -0.0050506, -0.0087822, -0.0052538, -0.0029741, 0.0026976
1: -0.0055104, -0.0043626, -0.0054147, -0.0044199, -0.0008385, 0.0007606
2: -0.0020971, 0.0063716, -0.0013910, 0.0059488, -0.0061867, 0.0056116
3: 0.0013498, 0.0024705, 0.0014432, 0.0024145, -0.0008187, 0.0007426
4: 0.0013301, 0.0076591, 0.0016461, 0.0071314, -0.0041938, 0.0046235
5: 0.9958758, 0.9976342, 0.9959636, 0.9974876, -0.0011652, 0.0012846
6: 0.0041401, 0.0057362, 0.0042198, 0.0056031, -0.0010576, 0.0011660
7: -0.0079314, -0.0019750, -0.0076340, -0.0024717, -0.0039468, 0.0043513
8: -0.0076557, -0.0030199, -0.0072691, -0.0032513, -0.0033866, 0.0030718
9: -0.0037492, -0.0033492, -0.0037292, -0.0033826, -0.0002650, 0.0002922

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005485, upper bound: 0.0005896
time: 1.97 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005530, upper bound: 0.0005983
time: 1.61 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0091923, -0.0051442, -0.0088406, -0.0052239, -0.0029937, 0.0026037
1: -0.0055303, -0.0043890, -0.0054312, -0.0044115, -0.0008440, 0.0007341
2: -0.0022439, 0.0061769, -0.0015124, 0.0060110, -0.0062275, 0.0054161
3: 0.0013303, 0.0024447, 0.0014272, 0.0024228, -0.0008241, 0.0007167
4: 0.0014756, 0.0077688, 0.0015996, 0.0072221, -0.0040477, 0.0046541
5: 0.9959162, 0.9976647, 0.9959506, 0.9975128, -0.0011246, 0.0012930
6: 0.0041768, 0.0057639, 0.0042081, 0.0056260, -0.0010208, 0.0011737
7: -0.0077944, -0.0018718, -0.0076777, -0.0023863, -0.0038093, 0.0043800
8: -0.0077361, -0.0031265, -0.0073356, -0.0032173, -0.0034090, 0.0029648
9: -0.0037400, -0.0033423, -0.0037322, -0.0033769, -0.0002558, 0.0002941

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_B2_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005510, upper bound: 0.0005849
time: 1.62 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005552, upper bound: 0.0005930
time: 1.75 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0091256, -0.0050495, -0.0087940, -0.0052330, -0.0030117, 0.0026976
1: -0.0055115, -0.0043623, -0.0054180, -0.0044140, -0.0008491, 0.0007605
2: -0.0021053, 0.0063739, -0.0014154, 0.0059922, -0.0062649, 0.0056115
3: 0.0013487, 0.0024708, 0.0014400, 0.0024203, -0.0008291, 0.0007426
4: 0.0013284, 0.0076652, 0.0016136, 0.0071497, -0.0041937, 0.0046820
5: 0.9958752, 0.9976359, 0.9959545, 0.9974926, -0.0011651, 0.0013008
6: 0.0041397, 0.0057377, 0.0042116, 0.0056077, -0.0010576, 0.0011807
7: -0.0079329, -0.0019693, -0.0076645, -0.0024545, -0.0039467, 0.0044063
8: -0.0076602, -0.0030187, -0.0072825, -0.0032276, -0.0034294, 0.0030717
9: -0.0037493, -0.0033489, -0.0037313, -0.0033814, -0.0002650, 0.0002959

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A2_B1_B2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005485, upper bound: 0.0005944
time: 1.48 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005552, upper bound: 0.0005984
time: 1.48 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0090819, -0.0051800, -0.0090876, -0.0051704, -0.0025226, 0.0025066
1: -0.0054992, -0.0043991, -0.0055008, -0.0043964, -0.0007112, 0.0007067
2: -0.0020144, 0.0061023, -0.0020263, 0.0061224, -0.0052475, 0.0052143
3: 0.0013607, 0.0024348, 0.0013592, 0.0024375, -0.0006944, 0.0006900
4: 0.0015314, 0.0075973, 0.0015163, 0.0076062, -0.0038969, 0.0039217
5: 0.9959316, 0.9976169, 0.9959276, 0.9976195, -0.0010827, 0.0010896
6: 0.0041909, 0.0057206, 0.0041871, 0.0057228, -0.0009827, 0.0009890
7: -0.0077419, -0.0020333, -0.0077561, -0.0020249, -0.0036674, 0.0036907
8: -0.0076104, -0.0031673, -0.0076169, -0.0031563, -0.0028725, 0.0028543
9: -0.0037365, -0.0033531, -0.0037374, -0.0033526, -0.0002463, 0.0002478

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A2_B2_A1_B1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005465, upper bound: 0.0005713
time: 1.44 seconds

## Relational analysis of NS_A2_B2_A1_B1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005519, upper bound: 0.0005767
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0090377, -0.0051897, -0.0090230, -0.0050731, -0.0026032, 0.0025463
1: -0.0054867, -0.0044018, -0.0054826, -0.0043690, -0.0007339, 0.0007179
2: -0.0019224, 0.0060822, -0.0018918, 0.0063248, -0.0054152, 0.0052969
3: 0.0013729, 0.0024322, 0.0013769, 0.0024643, -0.0007166, 0.0007010
4: 0.0015464, 0.0075286, 0.0013651, 0.0075057, -0.0039586, 0.0040470
5: 0.9959359, 0.9975979, 0.9958855, 0.9975916, -0.0010998, 0.0011244
6: 0.0041946, 0.0057033, 0.0041489, 0.0056975, -0.0009983, 0.0010206
7: -0.0077278, -0.0020979, -0.0078984, -0.0021194, -0.0037255, 0.0038087
8: -0.0075601, -0.0031783, -0.0075433, -0.0030455, -0.0029643, 0.0028995
9: -0.0037355, -0.0033575, -0.0037470, -0.0033589, -0.0002502, 0.0002557

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A2_B2_A1_B1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005468, upper bound: 0.0005714
time: 1.48 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005532, upper bound: 0.0005767
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0090864, -0.0051790, -0.0090995, -0.0051537, -0.0025585, 0.0024982
1: -0.0055005, -0.0043988, -0.0055041, -0.0043917, -0.0007213, 0.0007043
2: -0.0020237, 0.0061044, -0.0020510, 0.0061572, -0.0053223, 0.0051968
3: 0.0013595, 0.0024351, 0.0013559, 0.0024421, -0.0007043, 0.0006877
4: 0.0015298, 0.0076042, 0.0014904, 0.0076246, -0.0038838, 0.0039775
5: 0.9959313, 0.9976190, 0.9959202, 0.9976246, -0.0010790, 0.0011051
6: 0.0041905, 0.0057223, 0.0041805, 0.0057275, -0.0009794, 0.0010031
7: -0.0077434, -0.0020267, -0.0077805, -0.0020075, -0.0036550, 0.0037433
8: -0.0076155, -0.0031662, -0.0076304, -0.0031373, -0.0029134, 0.0028447
9: -0.0037366, -0.0033527, -0.0037391, -0.0033514, -0.0002454, 0.0002514

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A2_B2_A1_B2_B1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005489, upper bound: 0.0005714
time: 1.37 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_B2

### Relational analysis result of NS_A2_B2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005547, upper bound: 0.0005767
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0090415, -0.0051887, -0.0090309, -0.0050584, -0.0026351, 0.0025377
1: -0.0054878, -0.0044016, -0.0054848, -0.0043648, -0.0007429, 0.0007155
2: -0.0019304, 0.0060842, -0.0019082, 0.0063553, -0.0054815, 0.0052789
3: 0.0013718, 0.0024324, 0.0013748, 0.0024683, -0.0007254, 0.0006986
4: 0.0015449, 0.0075345, 0.0013423, 0.0075179, -0.0039451, 0.0040965
5: 0.9959354, 0.9975995, 0.9958792, 0.9975950, -0.0010961, 0.0011381
6: 0.0041943, 0.0057048, 0.0041432, 0.0057006, -0.0009949, 0.0010331
7: -0.0077292, -0.0020923, -0.0079199, -0.0021079, -0.0037128, 0.0038553
8: -0.0075644, -0.0031772, -0.0075523, -0.0030288, -0.0030006, 0.0028897
9: -0.0037356, -0.0033571, -0.0037484, -0.0033582, -0.0002493, 0.0002589

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A2_B2_A1_B2_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005712
time: 1.41 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005555, upper bound: 0.0005766
time: 1.41 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0091536, -0.0051616, -0.0091861, -0.0051452, -0.0025128, 0.0026444
1: -0.0055194, -0.0043939, -0.0055286, -0.0043893, -0.0007085, 0.0007456
2: -0.0021635, 0.0061406, -0.0022311, 0.0061747, -0.0052271, 0.0055009
3: 0.0013410, 0.0024399, 0.0013321, 0.0024444, -0.0006917, 0.0007280
4: 0.0015027, 0.0077087, 0.0014773, 0.0077592, -0.0041111, 0.0039064
5: 0.9959237, 0.9976479, 0.9959167, 0.9976619, -0.0011422, 0.0010853
6: 0.0041836, 0.0057487, 0.0041772, 0.0057614, -0.0010367, 0.0009851
7: -0.0077689, -0.0019284, -0.0077928, -0.0018809, -0.0038690, 0.0036764
8: -0.0076920, -0.0031463, -0.0077290, -0.0031277, -0.0028613, 0.0030112
9: -0.0037383, -0.0033461, -0.0037399, -0.0033429, -0.0002598, 0.0002469

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A2_B2_A2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005491, upper bound: 0.0005934
time: 1.45 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005547, upper bound: 0.0005975
time: 1.43 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0091098, -0.0051721, -0.0091217, -0.0050506, -0.0025938, 0.0026839
1: -0.0055071, -0.0043969, -0.0055104, -0.0043626, -0.0007313, 0.0007567
2: -0.0020725, 0.0061189, -0.0020971, 0.0063716, -0.0053955, 0.0055830
3: 0.0013530, 0.0024370, 0.0013498, 0.0024705, -0.0007140, 0.0007388
4: 0.0015190, 0.0076407, 0.0013301, 0.0076591, -0.0041724, 0.0040323
5: 0.9959282, 0.9976290, 0.9958758, 0.9976342, -0.0011592, 0.0011203
6: 0.0041877, 0.0057315, 0.0041401, 0.0057362, -0.0010522, 0.0010169
7: -0.0077536, -0.0019924, -0.0079314, -0.0019750, -0.0039267, 0.0037948
8: -0.0076422, -0.0031582, -0.0076557, -0.0030199, -0.0029535, 0.0030562
9: -0.0037373, -0.0033504, -0.0037492, -0.0033492, -0.0002637, 0.0002548

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A2_B2_A2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005933
time: 1.49 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005554, upper bound: 0.0005974
time: 1.34 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0091666, -0.0051445, -0.0091923, -0.0051442, -0.0025066, 0.0026849
1: -0.0055231, -0.0043891, -0.0055303, -0.0043890, -0.0007067, 0.0007570
2: -0.0021906, 0.0061763, -0.0022439, 0.0061769, -0.0052142, 0.0055851
3: 0.0013374, 0.0024446, 0.0013303, 0.0024447, -0.0006900, 0.0007391
4: 0.0014761, 0.0077290, 0.0014756, 0.0077688, -0.0041739, 0.0038968
5: 0.9959163, 0.9976536, 0.9959162, 0.9976647, -0.0011596, 0.0010826
6: 0.0041769, 0.0057538, 0.0041768, 0.0057639, -0.0010526, 0.0009827
7: -0.0077940, -0.0019093, -0.0077944, -0.0018718, -0.0039281, 0.0036673
8: -0.0077068, -0.0031268, -0.0077361, -0.0031265, -0.0028543, 0.0030573
9: -0.0037400, -0.0033448, -0.0037400, -0.0033423, -0.0002638, 0.0002463

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A2_B2_A2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005489, upper bound: 0.0005944
time: 1.44 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005547, upper bound: 0.0005984
time: 1.50 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0091224, -0.0051548, -0.0091256, -0.0050495, -0.0025867, 0.0027259
1: -0.0055106, -0.0043920, -0.0055115, -0.0043623, -0.0007293, 0.0007685
2: -0.0020986, 0.0061547, -0.0021053, 0.0063739, -0.0053808, 0.0056704
3: 0.0013496, 0.0024418, 0.0013487, 0.0024708, -0.0007121, 0.0007504
4: 0.0014922, 0.0076602, 0.0013284, 0.0076652, -0.0042377, 0.0040213
5: 0.9959208, 0.9976345, 0.9958752, 0.9976359, -0.0011774, 0.0011172
6: 0.0041810, 0.0057365, 0.0041397, 0.0057377, -0.0010687, 0.0010141
7: -0.0077788, -0.0019740, -0.0079329, -0.0019693, -0.0039881, 0.0037845
8: -0.0076565, -0.0031386, -0.0076602, -0.0030187, -0.0029455, 0.0031040
9: -0.0037390, -0.0033492, -0.0037493, -0.0033489, -0.0002678, 0.0002541

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A2_B2_A2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005944
time: 1.41 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005554, upper bound: 0.0005984
time: 1.49 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.69 seconds
NS_A1_A1_B2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005848, upper bound: 0.0005337
NS_A1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005930, upper bound: 0.0005389
NS_A1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0005339
NS_A1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005388
NS_A1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0005342
NS_A1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005391
NS_A1_A2_B2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005848, upper bound: 0.0005487
NS_A1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005928, upper bound: 0.0005531
NS_A1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005896, upper bound: 0.0005486
NS_A1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005532
NS_A1_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005848, upper bound: 0.0005512
NS_A1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005930, upper bound: 0.0005553
NS_A1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005943, upper bound: 0.0005486
NS_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005982, upper bound: 0.0005553
NS_A2_B1_B1_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005339, upper bound: 0.0005849
NS_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005389, upper bound: 0.0005930
NS_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005339, upper bound: 0.0005897
NS_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005389, upper bound: 0.0005984
NS_A2_B1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005343, upper bound: 0.0005849
NS_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005390, upper bound: 0.0005930
NS_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005341, upper bound: 0.0005897
NS_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005390, upper bound: 0.0005983
NS_A2_B1_B2_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005485, upper bound: 0.0005849
NS_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005532, upper bound: 0.0005930
NS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005485, upper bound: 0.0005896
NS_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005530, upper bound: 0.0005983
NS_A2_B1_B2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005510, upper bound: 0.0005849
NS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005552, upper bound: 0.0005930
NS_A2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005485, upper bound: 0.0005944
NS_A2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005552, upper bound: 0.0005984
NS_A2_B2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005465, upper bound: 0.0005713
NS_A2_B2_A1_B1_B1_B2, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005519, upper bound: 0.0005767
NS_A2_B2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005468, upper bound: 0.0005714
NS_A2_B2_A1_B1_B2_B2, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005532, upper bound: 0.0005767
NS_A2_B2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005489, upper bound: 0.0005714
NS_A2_B2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005547, upper bound: 0.0005767
NS_A2_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005712
NS_A2_B2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005555, upper bound: 0.0005766
NS_A2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005491, upper bound: 0.0005934
NS_A2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005547, upper bound: 0.0005975
NS_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005933
NS_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005554, upper bound: 0.0005974
NS_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005489, upper bound: 0.0005944
NS_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005547, upper bound: 0.0005984
NS_A2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005490, upper bound: 0.0005944
NS_A2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 5, lower bound: -0.0005554, upper bound: 0.0005984

## BFS NS instance: NS_A1_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0087153, -0.0052702, -0.0090845, -0.0051506, -0.0025977, 0.0026875
1: -0.0053958, -0.0044245, -0.0054999, -0.0043908, -0.0007324, 0.0007577
2: -0.0012518, 0.0059147, -0.0020198, 0.0061635, -0.0054036, 0.0055905
3: 0.0014616, 0.0024100, 0.0013600, 0.0024429, -0.0007151, 0.0007398
4: 0.0016716, 0.0070273, 0.0014857, 0.0076013, -0.0041780, 0.0040383
5: 0.9959707, 0.9974586, 0.9959189, 0.9976181, -0.0011608, 0.0011220
6: 0.0042262, 0.0055769, 0.0041793, 0.0057216, -0.0010536, 0.0010184
7: -0.0076100, -0.0025696, -0.0077849, -0.0020295, -0.0039319, 0.0038005
8: -0.0071929, -0.0032700, -0.0076133, -0.0031338, -0.0029580, 0.0030602
9: -0.0037276, -0.0033892, -0.0037394, -0.0033529, -0.0002640, 0.0002552

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A1_B2_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_B2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005745, upper bound: 0.0005278
time: 1.51 seconds

## Relational analysis of NS_A1_A1_B2_A1_B1_B2_B2

### Relational analysis result of NS_A1_A1_B2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005763, upper bound: 0.0005208
time: 1.56 seconds

## BFS NS instance: NS_A1_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0086112, -0.0052804, -0.0089113, -0.0049567, -0.0026618, 0.0026699
1: -0.0053665, -0.0044274, -0.0054511, -0.0043361, -0.0007505, 0.0007527
2: -0.0010352, 0.0058936, -0.0016594, 0.0065669, -0.0055370, 0.0055539
3: 0.0014903, 0.0024072, 0.0014077, 0.0024963, -0.0007327, 0.0007350
4: 0.0016874, 0.0068655, 0.0011842, 0.0073320, -0.0041506, 0.0041380
5: 0.9959750, 0.9974137, 0.9958352, 0.9975433, -0.0011532, 0.0011497
6: 0.0042302, 0.0055361, 0.0041033, 0.0056537, -0.0010467, 0.0010436
7: -0.0075951, -0.0027219, -0.0080687, -0.0022829, -0.0039062, 0.0038943
8: -0.0070744, -0.0032816, -0.0074161, -0.0029130, -0.0030310, 0.0030402
9: -0.0037266, -0.0033994, -0.0037584, -0.0033699, -0.0002623, 0.0002615

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A1_B2_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005696, upper bound: 0.0005226
time: 1.55 seconds

## Relational analysis of NS_A1_A1_B2_A1_B2_B1_B2

### Relational analysis result of NS_A1_A1_B2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005708, upper bound: 0.0005134
time: 1.64 seconds

## BFS NS instance: NS_A1_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0086658, -0.0052779, -0.0090198, -0.0050556, -0.0026825, 0.0027102
1: -0.0053819, -0.0044267, -0.0054817, -0.0043640, -0.0007563, 0.0007641
2: -0.0011488, 0.0058986, -0.0018853, 0.0063612, -0.0055801, 0.0056378
3: 0.0014753, 0.0024079, 0.0013778, 0.0024691, -0.0007384, 0.0007461
4: 0.0016836, 0.0069504, 0.0013379, 0.0075008, -0.0042134, 0.0041702
5: 0.9959740, 0.9974372, 0.9958779, 0.9975901, -0.0011706, 0.0011586
6: 0.0042292, 0.0055575, 0.0041421, 0.0056963, -0.0010625, 0.0010517
7: -0.0075987, -0.0026421, -0.0079240, -0.0021241, -0.0039652, 0.0039246
8: -0.0071366, -0.0032788, -0.0075397, -0.0030256, -0.0030545, 0.0030862
9: -0.0037269, -0.0033940, -0.0037487, -0.0033592, -0.0002663, 0.0002635

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A1_B2_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005798, upper bound: 0.0005278
time: 1.43 seconds

## Relational analysis of NS_A1_A1_B2_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_B2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005817, upper bound: 0.0005208
time: 1.43 seconds

## BFS NS instance: NS_A1_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0086179, -0.0052711, -0.0089157, -0.0049556, -0.0026555, 0.0026907
1: -0.0053684, -0.0044248, -0.0054523, -0.0043358, -0.0007487, 0.0007586
2: -0.0010492, 0.0059128, -0.0016686, 0.0065691, -0.0055240, 0.0055972
3: 0.0014884, 0.0024098, 0.0014065, 0.0024966, -0.0007310, 0.0007407
4: 0.0016730, 0.0068760, 0.0011825, 0.0073388, -0.0041830, 0.0041283
5: 0.9959711, 0.9974166, 0.9958348, 0.9975452, -0.0011622, 0.0011470
6: 0.0042266, 0.0055387, 0.0041029, 0.0056554, -0.0010549, 0.0010411
7: -0.0076087, -0.0027121, -0.0080703, -0.0022765, -0.0039367, 0.0038852
8: -0.0070820, -0.0032710, -0.0074211, -0.0029118, -0.0030239, 0.0030639
9: -0.0037275, -0.0033987, -0.0037585, -0.0033695, -0.0002643, 0.0002609

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A1_B2_A2_B2_B1_B1

### Relational analysis result of NS_A1_A1_B2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005696, upper bound: 0.0005232
time: 1.61 seconds

## Relational analysis of NS_A1_A1_B2_A2_B2_B1_B2

### Relational analysis result of NS_A1_A1_B2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005708, upper bound: 0.0005136
time: 1.64 seconds

## BFS NS instance: NS_A1_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0086720, -0.0052686, -0.0090234, -0.0050545, -0.0026793, 0.0027432
1: -0.0053836, -0.0044241, -0.0054827, -0.0043637, -0.0007554, 0.0007734
2: -0.0011617, 0.0059180, -0.0018927, 0.0063635, -0.0055735, 0.0057063
3: 0.0014736, 0.0024104, 0.0013768, 0.0024694, -0.0007376, 0.0007551
4: 0.0016691, 0.0069600, 0.0013362, 0.0075064, -0.0042645, 0.0041653
5: 0.9959700, 0.9974400, 0.9958774, 0.9975917, -0.0011848, 0.0011572
6: 0.0042256, 0.0055599, 0.0041416, 0.0056977, -0.0010755, 0.0010504
7: -0.0076123, -0.0026330, -0.0079256, -0.0021188, -0.0040134, 0.0039200
8: -0.0071436, -0.0032682, -0.0075438, -0.0030243, -0.0030509, 0.0031236
9: -0.0037278, -0.0033934, -0.0037488, -0.0033589, -0.0002695, 0.0002632

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A1_B2_A2_B2_B2_B1

### Relational analysis result of NS_A1_A1_B2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005798, upper bound: 0.0005281
time: 1.82 seconds

## Relational analysis of NS_A1_A1_B2_A2_B2_B2_B2

### Relational analysis result of NS_A1_A1_B2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005818, upper bound: 0.0005209
time: 1.51 seconds

## BFS NS instance: NS_A1_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0088219, -0.0052450, -0.0091405, -0.0051466, -0.0026022, 0.0028110
1: -0.0054259, -0.0044174, -0.0055157, -0.0043897, -0.0007337, 0.0007925
2: -0.0014736, 0.0059671, -0.0021362, 0.0061719, -0.0054131, 0.0058474
3: 0.0014323, 0.0024169, 0.0013446, 0.0024441, -0.0007163, 0.0007738
4: 0.0016324, 0.0071931, 0.0014793, 0.0076883, -0.0043700, 0.0040454
5: 0.9959598, 0.9975047, 0.9959171, 0.9976423, -0.0012141, 0.0011239
6: 0.0042163, 0.0056187, 0.0041777, 0.0057435, -0.0011020, 0.0010202
7: -0.0076469, -0.0024136, -0.0077909, -0.0019476, -0.0041126, 0.0038072
8: -0.0073144, -0.0032413, -0.0076771, -0.0031292, -0.0029631, 0.0032009
9: -0.0037301, -0.0033787, -0.0037398, -0.0033474, -0.0002762, 0.0002556

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_B2_A1_B1_B2_B1

### Relational analysis result of NS_A1_A2_B2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005745, upper bound: 0.0005410
time: 1.67 seconds

## Relational analysis of NS_A1_A2_B2_A1_B1_B2_B2

### Relational analysis result of NS_A1_A2_B2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005763, upper bound: 0.0005359
time: 1.44 seconds

## BFS NS instance: NS_A1_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0087202, -0.0052563, -0.0089665, -0.0049528, -0.0026765, 0.0027853
1: -0.0053972, -0.0044206, -0.0054667, -0.0043350, -0.0007546, 0.0007853
2: -0.0012619, 0.0059437, -0.0017743, 0.0065750, -0.0055678, 0.0057939
3: 0.0014603, 0.0024139, 0.0013925, 0.0024974, -0.0007368, 0.0007667
4: 0.0016499, 0.0070349, 0.0011781, 0.0074179, -0.0043300, 0.0041610
5: 0.9959647, 0.9974607, 0.9958335, 0.9975671, -0.0012030, 0.0011560
6: 0.0042208, 0.0055788, 0.0041018, 0.0056753, -0.0010920, 0.0010493
7: -0.0076304, -0.0025625, -0.0080744, -0.0022021, -0.0040750, 0.0039160
8: -0.0071985, -0.0032541, -0.0074790, -0.0029086, -0.0030478, 0.0031716
9: -0.0037290, -0.0033887, -0.0037588, -0.0033645, -0.0002736, 0.0002629

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_B2_A1_B2_B1_B1

### Relational analysis result of NS_A1_A2_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005696, upper bound: 0.0005365
time: 1.70 seconds

## Relational analysis of NS_A1_A2_B2_A1_B2_B1_B2

### Relational analysis result of NS_A1_A2_B2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005707, upper bound: 0.0005299
time: 1.94 seconds

## BFS NS instance: NS_A1_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0087770, -0.0052540, -0.0090763, -0.0050518, -0.0026930, 0.0028284
1: -0.0054132, -0.0044200, -0.0054976, -0.0043630, -0.0007593, 0.0007974
2: -0.0013801, 0.0059485, -0.0020026, 0.0063690, -0.0056020, 0.0058836
3: 0.0014447, 0.0024145, 0.0013623, 0.0024701, -0.0007413, 0.0007786
4: 0.0016463, 0.0071232, 0.0013321, 0.0075885, -0.0043970, 0.0041866
5: 0.9959636, 0.9974853, 0.9958763, 0.9976145, -0.0012216, 0.0011632
6: 0.0042198, 0.0056010, 0.0041406, 0.0057184, -0.0011089, 0.0010558
7: -0.0076338, -0.0024794, -0.0079295, -0.0020415, -0.0041381, 0.0039401
8: -0.0072632, -0.0032515, -0.0076039, -0.0030213, -0.0030666, 0.0032207
9: -0.0037292, -0.0033831, -0.0037491, -0.0033537, -0.0002779, 0.0002646

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_B2_A1_B2_B2_B1

### Relational analysis result of NS_A1_A2_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005798, upper bound: 0.0005411
time: 1.60 seconds

## Relational analysis of NS_A1_A2_B2_A1_B2_B2_B2

### Relational analysis result of NS_A1_A2_B2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005816, upper bound: 0.0005359
time: 1.83 seconds

## BFS NS instance: NS_A1_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0088353, -0.0052241, -0.0091460, -0.0051455, -0.0025992, 0.0028607
1: -0.0054297, -0.0044115, -0.0055173, -0.0043894, -0.0007328, 0.0008065
2: -0.0015014, 0.0060107, -0.0021477, 0.0061742, -0.0054070, 0.0059508
3: 0.0014286, 0.0024227, 0.0013431, 0.0024444, -0.0007155, 0.0007875
4: 0.0015998, 0.0072139, 0.0014776, 0.0076969, -0.0044472, 0.0040408
5: 0.9959507, 0.9975104, 0.9959168, 0.9976447, -0.0012356, 0.0011227
6: 0.0042081, 0.0056239, 0.0041773, 0.0057457, -0.0011215, 0.0010190
7: -0.0076775, -0.0023940, -0.0077925, -0.0019395, -0.0041853, 0.0038029
8: -0.0073296, -0.0032175, -0.0076834, -0.0031279, -0.0029598, 0.0032575
9: -0.0037321, -0.0033774, -0.0037399, -0.0033469, -0.0002810, 0.0002554

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_B2_A2_B1_B2_B1

### Relational analysis result of NS_A1_A2_B2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005745, upper bound: 0.0005433
time: 1.52 seconds

## Relational analysis of NS_A1_A2_B2_A2_B1_B2_B2

### Relational analysis result of NS_A1_A2_B2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005763, upper bound: 0.0005382
time: 1.62 seconds

## BFS NS instance: NS_A1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0086411, -0.0051332, -0.0090624, -0.0050516, -0.0025160, 0.0029846
1: -0.0053749, -0.0043859, -0.0054937, -0.0043629, -0.0007094, 0.0008415
2: -0.0010974, 0.0061998, -0.0019738, 0.0063696, -0.0052338, 0.0062085
3: 0.0014821, 0.0024477, 0.0013661, 0.0024702, -0.0006926, 0.0008216
4: 0.0014585, 0.0069120, 0.0013317, 0.0075670, -0.0046399, 0.0039114
5: 0.9959115, 0.9974266, 0.9958762, 0.9976085, -0.0012891, 0.0010867
6: 0.0041725, 0.0055478, 0.0041405, 0.0057129, -0.0011701, 0.0009864
7: -0.0078105, -0.0026782, -0.0079299, -0.0020618, -0.0043666, 0.0036811
8: -0.0071084, -0.0031140, -0.0075882, -0.0030210, -0.0028650, 0.0033986
9: -0.0037411, -0.0033965, -0.0037491, -0.0033551, -0.0002932, 0.0002472

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005755, upper bound: 0.0005366
time: 1.67 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005767, upper bound: 0.0005294
time: 1.52 seconds

## BFS NS instance: NS_A1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0087477, -0.0052343, -0.0091204, -0.0050496, -0.0025740, 0.0030071
1: -0.0054050, -0.0044144, -0.0055100, -0.0043623, -0.0007257, 0.0008478
2: -0.0013191, 0.0059893, -0.0020944, 0.0063736, -0.0053544, 0.0062554
3: 0.0014527, 0.0024199, 0.0013501, 0.0024707, -0.0007086, 0.0008278
4: 0.0016158, 0.0070777, 0.0013287, 0.0076571, -0.0046749, 0.0040015
5: 0.9959551, 0.9974726, 0.9958754, 0.9976336, -0.0012988, 0.0011117
6: 0.0042122, 0.0055896, 0.0041397, 0.0057357, -0.0011789, 0.0010091
7: -0.0076625, -0.0025222, -0.0079327, -0.0019770, -0.0043996, 0.0037659
8: -0.0072298, -0.0032291, -0.0076542, -0.0030188, -0.0029310, 0.0034242
9: -0.0037311, -0.0033860, -0.0037493, -0.0033494, -0.0002954, 0.0002529

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005798, upper bound: 0.0005433
time: 1.81 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005818, upper bound: 0.0005383
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0090845, -0.0051506, -0.0087153, -0.0052702, -0.0026875, 0.0025977
1: -0.0054999, -0.0043908, -0.0053958, -0.0044245, -0.0007577, 0.0007324
2: -0.0020198, 0.0061635, -0.0012518, 0.0059147, -0.0055905, 0.0054036
3: 0.0013600, 0.0024429, 0.0014616, 0.0024100, -0.0007398, 0.0007151
4: 0.0014857, 0.0076013, 0.0016716, 0.0070273, -0.0040383, 0.0041780
5: 0.9959189, 0.9976181, 0.9959707, 0.9974586, -0.0011220, 0.0011608
6: 0.0041793, 0.0057216, 0.0042262, 0.0055769, -0.0010184, 0.0010536
7: -0.0077849, -0.0020295, -0.0076100, -0.0025696, -0.0038005, 0.0039319
8: -0.0076133, -0.0031338, -0.0071929, -0.0032700, -0.0030602, 0.0029580
9: -0.0037394, -0.0033529, -0.0037276, -0.0033892, -0.0002552, 0.0002640

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B1_B1_A1_A2_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005277, upper bound: 0.0005745
time: 1.64 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_A2_A2

### Relational analysis result of NS_A2_B1_B1_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005208, upper bound: 0.0005763
time: 1.54 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0089113, -0.0049567, -0.0086112, -0.0052804, -0.0026699, 0.0026618
1: -0.0054511, -0.0043361, -0.0053665, -0.0044274, -0.0007527, 0.0007505
2: -0.0016594, 0.0065669, -0.0010352, 0.0058936, -0.0055539, 0.0055370
3: 0.0014077, 0.0024963, 0.0014903, 0.0024072, -0.0007350, 0.0007327
4: 0.0011842, 0.0073320, 0.0016874, 0.0068655, -0.0041380, 0.0041506
5: 0.9958352, 0.9975433, 0.9959750, 0.9974137, -0.0011497, 0.0011532
6: 0.0041033, 0.0056537, 0.0042302, 0.0055361, -0.0010436, 0.0010467
7: -0.0080687, -0.0022829, -0.0075951, -0.0027219, -0.0038943, 0.0039062
8: -0.0074161, -0.0029130, -0.0070744, -0.0032816, -0.0030402, 0.0030310
9: -0.0037584, -0.0033699, -0.0037266, -0.0033994, -0.0002615, 0.0002623

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B1_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005225, upper bound: 0.0005697
time: 1.62 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_A1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005133, upper bound: 0.0005709
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0090198, -0.0050556, -0.0086658, -0.0052779, -0.0027102, 0.0026825
1: -0.0054817, -0.0043640, -0.0053819, -0.0044267, -0.0007641, 0.0007563
2: -0.0018853, 0.0063612, -0.0011488, 0.0058986, -0.0056378, 0.0055801
3: 0.0013778, 0.0024691, 0.0014753, 0.0024079, -0.0007461, 0.0007384
4: 0.0013379, 0.0075008, 0.0016836, 0.0069504, -0.0041702, 0.0042134
5: 0.9958779, 0.9975901, 0.9959740, 0.9974372, -0.0011586, 0.0011706
6: 0.0041421, 0.0056963, 0.0042292, 0.0055575, -0.0010517, 0.0010625
7: -0.0079240, -0.0021241, -0.0075987, -0.0026421, -0.0039246, 0.0039652
8: -0.0075397, -0.0030256, -0.0071366, -0.0032788, -0.0030862, 0.0030545
9: -0.0037487, -0.0033592, -0.0037269, -0.0033940, -0.0002635, 0.0002663

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B1_B1_A2_A2_A1

### Relational analysis result of NS_A2_B1_B1_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005278, upper bound: 0.0005798
time: 1.50 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_A2_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005209, upper bound: 0.0005818
time: 1.72 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0090900, -0.0051496, -0.0087199, -0.0052607, -0.0027179, 0.0025889
1: -0.0055015, -0.0043905, -0.0053971, -0.0044218, -0.0007663, 0.0007299
2: -0.0020312, 0.0061656, -0.0012613, 0.0059345, -0.0056537, 0.0053854
3: 0.0013585, 0.0024432, 0.0014604, 0.0024126, -0.0007482, 0.0007127
4: 0.0014841, 0.0076098, 0.0016568, 0.0070344, -0.0040247, 0.0042253
5: 0.9959186, 0.9976205, 0.9959666, 0.9974607, -0.0011182, 0.0011739
6: 0.0041789, 0.0057238, 0.0042225, 0.0055787, -0.0010150, 0.0010655
7: -0.0077865, -0.0020214, -0.0076239, -0.0025629, -0.0037877, 0.0039764
8: -0.0076196, -0.0031326, -0.0071981, -0.0032591, -0.0030949, 0.0029480
9: -0.0037395, -0.0033524, -0.0037286, -0.0033887, -0.0002543, 0.0002670

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B1_B2_A1_A2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005281, upper bound: 0.0005745
time: 1.65 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_A2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005209, upper bound: 0.0005763
time: 1.60 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0089157, -0.0049556, -0.0086179, -0.0052711, -0.0026907, 0.0026555
1: -0.0054523, -0.0043358, -0.0053684, -0.0044248, -0.0007586, 0.0007487
2: -0.0016686, 0.0065691, -0.0010492, 0.0059128, -0.0055972, 0.0055240
3: 0.0014065, 0.0024966, 0.0014884, 0.0024098, -0.0007407, 0.0007310
4: 0.0011825, 0.0073388, 0.0016730, 0.0068760, -0.0041283, 0.0041830
5: 0.9958348, 0.9975452, 0.9959711, 0.9974166, -0.0011470, 0.0011622
6: 0.0041029, 0.0056554, 0.0042266, 0.0055387, -0.0010411, 0.0010549
7: -0.0080703, -0.0022765, -0.0076087, -0.0027121, -0.0038852, 0.0039367
8: -0.0074211, -0.0029118, -0.0070820, -0.0032710, -0.0030639, 0.0030239
9: -0.0037585, -0.0033695, -0.0037275, -0.0033987, -0.0002609, 0.0002643

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B1_B2_A2_A1_A1

### Relational analysis result of NS_A2_B1_B1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005226, upper bound: 0.0005697
time: 2.09 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_A1_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005136, upper bound: 0.0005709
time: 1.53 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0090234, -0.0050545, -0.0086720, -0.0052686, -0.0027432, 0.0026793
1: -0.0054827, -0.0043637, -0.0053836, -0.0044241, -0.0007734, 0.0007554
2: -0.0018927, 0.0063635, -0.0011617, 0.0059180, -0.0057063, 0.0055735
3: 0.0013768, 0.0024694, 0.0014736, 0.0024104, -0.0007551, 0.0007376
4: 0.0013362, 0.0075064, 0.0016691, 0.0069600, -0.0041653, 0.0042645
5: 0.9958774, 0.9975917, 0.9959700, 0.9974400, -0.0011572, 0.0011848
6: 0.0041416, 0.0056977, 0.0042256, 0.0055599, -0.0010504, 0.0010755
7: -0.0079256, -0.0021188, -0.0076123, -0.0026330, -0.0039200, 0.0040134
8: -0.0075438, -0.0030243, -0.0071436, -0.0032682, -0.0031236, 0.0030509
9: -0.0037488, -0.0033589, -0.0037278, -0.0033934, -0.0002632, 0.0002695

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B1_B2_A2_A2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005281, upper bound: 0.0005798
time: 1.55 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_A2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005209, upper bound: 0.0005818
time: 1.97 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0091405, -0.0051466, -0.0088219, -0.0052450, -0.0028110, 0.0026022
1: -0.0055157, -0.0043897, -0.0054259, -0.0044174, -0.0007925, 0.0007337
2: -0.0021362, 0.0061719, -0.0014736, 0.0059671, -0.0058474, 0.0054131
3: 0.0013446, 0.0024441, 0.0014323, 0.0024169, -0.0007738, 0.0007163
4: 0.0014793, 0.0076883, 0.0016324, 0.0071931, -0.0040454, 0.0043700
5: 0.9959171, 0.9976423, 0.9959598, 0.9975047, -0.0011239, 0.0012141
6: 0.0041777, 0.0057435, 0.0042163, 0.0056187, -0.0010202, 0.0011020
7: -0.0077909, -0.0019476, -0.0076469, -0.0024136, -0.0038072, 0.0041126
8: -0.0076771, -0.0031292, -0.0073144, -0.0032413, -0.0032009, 0.0029631
9: -0.0037398, -0.0033474, -0.0037301, -0.0033787, -0.0002556, 0.0002762

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B2_B1_A1_A2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005411, upper bound: 0.0005745
time: 1.46 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_A2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005359, upper bound: 0.0005763
time: 1.66 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0089665, -0.0049528, -0.0087202, -0.0052563, -0.0027853, 0.0026765
1: -0.0054667, -0.0043350, -0.0053972, -0.0044206, -0.0007853, 0.0007546
2: -0.0017743, 0.0065750, -0.0012619, 0.0059437, -0.0057939, 0.0055678
3: 0.0013925, 0.0024974, 0.0014603, 0.0024139, -0.0007667, 0.0007368
4: 0.0011781, 0.0074179, 0.0016499, 0.0070349, -0.0041610, 0.0043300
5: 0.9958335, 0.9975671, 0.9959647, 0.9974607, -0.0011560, 0.0012030
6: 0.0041018, 0.0056753, 0.0042208, 0.0055788, -0.0010493, 0.0010920
7: -0.0080744, -0.0022021, -0.0076304, -0.0025625, -0.0039160, 0.0040750
8: -0.0074790, -0.0029086, -0.0071985, -0.0032541, -0.0031716, 0.0030478
9: -0.0037588, -0.0033645, -0.0037290, -0.0033887, -0.0002629, 0.0002736

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B2_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005365, upper bound: 0.0005698
time: 1.59 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_A1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005299, upper bound: 0.0005709
time: 1.67 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0090763, -0.0050518, -0.0087770, -0.0052540, -0.0028284, 0.0026930
1: -0.0054976, -0.0043630, -0.0054132, -0.0044200, -0.0007974, 0.0007593
2: -0.0020026, 0.0063690, -0.0013801, 0.0059485, -0.0058836, 0.0056020
3: 0.0013623, 0.0024701, 0.0014447, 0.0024145, -0.0007786, 0.0007413
4: 0.0013321, 0.0075885, 0.0016463, 0.0071232, -0.0041866, 0.0043970
5: 0.9958763, 0.9976145, 0.9959636, 0.9974853, -0.0011632, 0.0012216
6: 0.0041406, 0.0057184, 0.0042198, 0.0056010, -0.0010558, 0.0011089
7: -0.0079295, -0.0020415, -0.0076338, -0.0024794, -0.0039401, 0.0041381
8: -0.0076039, -0.0030213, -0.0072632, -0.0032515, -0.0032207, 0.0030666
9: -0.0037491, -0.0033537, -0.0037292, -0.0033831, -0.0002646, 0.0002779

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005410, upper bound: 0.0005798
time: 1.70 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005359, upper bound: 0.0005817
time: 1.76 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0091460, -0.0051455, -0.0088353, -0.0052241, -0.0028607, 0.0025992
1: -0.0055173, -0.0043894, -0.0054297, -0.0044115, -0.0008065, 0.0007328
2: -0.0021477, 0.0061742, -0.0015014, 0.0060107, -0.0059508, 0.0054070
3: 0.0013431, 0.0024444, 0.0014286, 0.0024227, -0.0007875, 0.0007155
4: 0.0014776, 0.0076969, 0.0015998, 0.0072139, -0.0040408, 0.0044472
5: 0.9959168, 0.9976447, 0.9959507, 0.9975104, -0.0011227, 0.0012356
6: 0.0041773, 0.0057457, 0.0042081, 0.0056239, -0.0010190, 0.0011215
7: -0.0077925, -0.0019395, -0.0076775, -0.0023940, -0.0038029, 0.0041853
8: -0.0076834, -0.0031279, -0.0073296, -0.0032175, -0.0032575, 0.0029598
9: -0.0037399, -0.0033469, -0.0037321, -0.0033774, -0.0002554, 0.0002810

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B2_B2_A1_A2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005433, upper bound: 0.0005745
time: 1.52 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_A2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005383, upper bound: 0.0005763
time: 1.65 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0090624, -0.0050516, -0.0086411, -0.0051332, -0.0029846, 0.0025160
1: -0.0054937, -0.0043629, -0.0053749, -0.0043859, -0.0008415, 0.0007094
2: -0.0019738, 0.0063696, -0.0010974, 0.0061998, -0.0062085, 0.0052338
3: 0.0013661, 0.0024702, 0.0014821, 0.0024477, -0.0008216, 0.0006926
4: 0.0013317, 0.0075670, 0.0014585, 0.0069120, -0.0039114, 0.0046399
5: 0.9958762, 0.9976085, 0.9959115, 0.9974266, -0.0010867, 0.0012891
6: 0.0041405, 0.0057129, 0.0041725, 0.0055478, -0.0009864, 0.0011701
7: -0.0079299, -0.0020618, -0.0078105, -0.0026782, -0.0036811, 0.0043666
8: -0.0075882, -0.0030210, -0.0071084, -0.0031140, -0.0033986, 0.0028650
9: -0.0037491, -0.0033551, -0.0037411, -0.0033965, -0.0002472, 0.0002932

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005364, upper bound: 0.0005756
time: 1.53 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005294, upper bound: 0.0005768
time: 1.55 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0091204, -0.0050496, -0.0087477, -0.0052343, -0.0030071, 0.0025740
1: -0.0055100, -0.0043623, -0.0054050, -0.0044144, -0.0008478, 0.0007257
2: -0.0020944, 0.0063736, -0.0013191, 0.0059893, -0.0062554, 0.0053544
3: 0.0013501, 0.0024707, 0.0014527, 0.0024199, -0.0008278, 0.0007086
4: 0.0013287, 0.0076571, 0.0016158, 0.0070777, -0.0040015, 0.0046749
5: 0.9958754, 0.9976336, 0.9959551, 0.9974726, -0.0011117, 0.0012988
6: 0.0041397, 0.0057357, 0.0042122, 0.0055896, -0.0010091, 0.0011789
7: -0.0079327, -0.0019770, -0.0076625, -0.0025222, -0.0037659, 0.0043996
8: -0.0076542, -0.0030188, -0.0072298, -0.0032291, -0.0034242, 0.0029310
9: -0.0037493, -0.0033494, -0.0037311, -0.0033860, -0.0002529, 0.0002954

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005433, upper bound: 0.0005798
time: 1.50 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005382, upper bound: 0.0005818
time: 1.57 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0090904, -0.0051641, -0.0090325, -0.0050491, -0.0024806, 0.0024508
1: -0.0055016, -0.0043946, -0.0054853, -0.0043622, -0.0006994, 0.0006910
2: -0.0020320, 0.0061356, -0.0019117, 0.0063747, -0.0051601, 0.0050982
3: 0.0013584, 0.0024392, 0.0013743, 0.0024709, -0.0006829, 0.0006747
4: 0.0015065, 0.0076104, 0.0013278, 0.0075205, -0.0038101, 0.0038563
5: 0.9959248, 0.9976206, 0.9958751, 0.9975957, -0.0010585, 0.0010714
6: 0.0041846, 0.0057239, 0.0041395, 0.0057012, -0.0009608, 0.0009725
7: -0.0077653, -0.0020209, -0.0079335, -0.0021055, -0.0035857, 0.0036293
8: -0.0076200, -0.0031491, -0.0075542, -0.0030182, -0.0028247, 0.0027907
9: -0.0037380, -0.0033523, -0.0037493, -0.0033580, -0.0002408, 0.0002437

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005297, upper bound: 0.0005811
time: 1.44 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005309, upper bound: 0.0005755
time: 1.41 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0091484, -0.0051618, -0.0091405, -0.0051466, -0.0025086, 0.0024952
1: -0.0055179, -0.0043940, -0.0055157, -0.0043897, -0.0007073, 0.0007035
2: -0.0021527, 0.0061403, -0.0021362, 0.0061719, -0.0052184, 0.0051906
3: 0.0013424, 0.0024399, 0.0013446, 0.0024441, -0.0006906, 0.0006869
4: 0.0015030, 0.0077007, 0.0014793, 0.0076883, -0.0038791, 0.0038999
5: 0.9959238, 0.9976457, 0.9959171, 0.9976423, -0.0010777, 0.0010835
6: 0.0041837, 0.0057467, 0.0041777, 0.0057435, -0.0009783, 0.0009835
7: -0.0077687, -0.0019360, -0.0077909, -0.0019476, -0.0036507, 0.0036702
8: -0.0076861, -0.0031465, -0.0076771, -0.0031292, -0.0028565, 0.0028413
9: -0.0037383, -0.0033466, -0.0037398, -0.0033474, -0.0002451, 0.0002464

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_A1_B1_B2_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005355, upper bound: 0.0005853
time: 1.53 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_B2_B2

### Relational analysis result of NS_A2_B2_A2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005376, upper bound: 0.0005806
time: 1.59 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0090461, -0.0051743, -0.0089665, -0.0049528, -0.0025699, 0.0024878
1: -0.0054891, -0.0043975, -0.0054667, -0.0043350, -0.0007246, 0.0007014
2: -0.0019398, 0.0061142, -0.0017743, 0.0065750, -0.0053459, 0.0051751
3: 0.0013706, 0.0024364, 0.0013925, 0.0024974, -0.0007075, 0.0006848
4: 0.0015225, 0.0075416, 0.0011781, 0.0074179, -0.0038675, 0.0039952
5: 0.9959292, 0.9976016, 0.9958335, 0.9975671, -0.0010745, 0.0011100
6: 0.0041886, 0.0057065, 0.0041018, 0.0056753, -0.0009753, 0.0010075
7: -0.0077503, -0.0020857, -0.0080744, -0.0022021, -0.0036398, 0.0037600
8: -0.0075696, -0.0031608, -0.0074790, -0.0029086, -0.0029264, 0.0028329
9: -0.0037370, -0.0033567, -0.0037588, -0.0033645, -0.0002444, 0.0002525

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005296, upper bound: 0.0005812
time: 1.59 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005306, upper bound: 0.0005755
time: 2.07 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0091047, -0.0051722, -0.0090763, -0.0050518, -0.0025891, 0.0025354
1: -0.0055056, -0.0043969, -0.0054976, -0.0043630, -0.0007300, 0.0007148
2: -0.0020617, 0.0061186, -0.0020026, 0.0063690, -0.0053859, 0.0052741
3: 0.0013545, 0.0024370, 0.0013623, 0.0024701, -0.0007127, 0.0006979
4: 0.0015192, 0.0076327, 0.0013321, 0.0075885, -0.0039415, 0.0040251
5: 0.9959283, 0.9976268, 0.9958763, 0.9976145, -0.0010951, 0.0011183
6: 0.0041878, 0.0057295, 0.0041406, 0.0057184, -0.0009940, 0.0010151
7: -0.0077534, -0.0020000, -0.0079295, -0.0020415, -0.0037094, 0.0037881
8: -0.0076363, -0.0031584, -0.0076039, -0.0030213, -0.0029483, 0.0028870
9: -0.0037372, -0.0033509, -0.0037491, -0.0033537, -0.0002491, 0.0002544

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_B1

### Relational analysis result of NS_A2_B2_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005351, upper bound: 0.0005853
time: 2.19 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005385, upper bound: 0.0005806
time: 1.66 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0091043, -0.0051470, -0.0090391, -0.0050480, -0.0024604, 0.0024958
1: -0.0055055, -0.0043898, -0.0054871, -0.0043619, -0.0006937, 0.0007037
2: -0.0020609, 0.0061711, -0.0019253, 0.0063770, -0.0051181, 0.0051919
3: 0.0013546, 0.0024439, 0.0013725, 0.0024712, -0.0006773, 0.0006871
4: 0.0014800, 0.0076320, 0.0013261, 0.0075307, -0.0038801, 0.0038250
5: 0.9959174, 0.9976267, 0.9958748, 0.9975985, -0.0010780, 0.0010627
6: 0.0041779, 0.0057294, 0.0041391, 0.0057038, -0.0009785, 0.0009646
7: -0.0077903, -0.0020006, -0.0079351, -0.0020959, -0.0036516, 0.0035997
8: -0.0076358, -0.0031297, -0.0075616, -0.0030170, -0.0028017, 0.0028420
9: -0.0037397, -0.0033510, -0.0037494, -0.0033574, -0.0002452, 0.0002417

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_A2_B1_B1_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005296, upper bound: 0.0005822
time: 1.67 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_B1_B2

### Relational analysis result of NS_A2_B2_A2_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005309, upper bound: 0.0005768
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0091614, -0.0051446, -0.0091460, -0.0051455, -0.0025024, 0.0025509
1: -0.0055216, -0.0043891, -0.0055173, -0.0043894, -0.0007055, 0.0007192
2: -0.0021798, 0.0061759, -0.0021477, 0.0061742, -0.0052055, 0.0053064
3: 0.0013388, 0.0024446, 0.0013431, 0.0024444, -0.0006889, 0.0007022
4: 0.0014764, 0.0077209, 0.0014776, 0.0076969, -0.0039656, 0.0038902
5: 0.9959164, 0.9976513, 0.9959168, 0.9976447, -0.0011018, 0.0010808
6: 0.0041770, 0.0057518, 0.0041773, 0.0057457, -0.0010001, 0.0009811
7: -0.0077937, -0.0019169, -0.0077925, -0.0019395, -0.0037321, 0.0036611
8: -0.0077009, -0.0031270, -0.0076834, -0.0031279, -0.0028495, 0.0029047
9: -0.0037400, -0.0033453, -0.0037399, -0.0033469, -0.0002506, 0.0002458

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_A2_B1_B2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005355, upper bound: 0.0005863
time: 1.58 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_B2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005376, upper bound: 0.0005818
time: 1.55 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0090595, -0.0051572, -0.0089717, -0.0049518, -0.0025531, 0.0025341
1: -0.0054929, -0.0043927, -0.0054681, -0.0043347, -0.0007198, 0.0007145
2: -0.0019677, 0.0061499, -0.0017851, 0.0065772, -0.0053111, 0.0052715
3: 0.0013669, 0.0024411, 0.0013911, 0.0024977, -0.0007028, 0.0006976
4: 0.0014958, 0.0075624, 0.0011765, 0.0074260, -0.0039396, 0.0039692
5: 0.9959219, 0.9976072, 0.9958330, 0.9975694, -0.0010945, 0.0011027
6: 0.0041819, 0.0057118, 0.0041014, 0.0056774, -0.0009935, 0.0010010
7: -0.0077754, -0.0020661, -0.0080759, -0.0021945, -0.0037076, 0.0037354
8: -0.0075849, -0.0031413, -0.0074849, -0.0029074, -0.0029073, 0.0028856
9: -0.0037387, -0.0033554, -0.0037589, -0.0033640, -0.0002490, 0.0002508

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005296, upper bound: 0.0005822
time: 1.68 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005306, upper bound: 0.0005766
time: 1.69 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0091172, -0.0051550, -0.0090798, -0.0050507, -0.0025821, 0.0025920
1: -0.0055091, -0.0043920, -0.0054986, -0.0043627, -0.0007280, 0.0007308
2: -0.0020878, 0.0061544, -0.0020100, 0.0063713, -0.0053713, 0.0053919
3: 0.0013510, 0.0024417, 0.0013613, 0.0024704, -0.0007108, 0.0007135
4: 0.0014924, 0.0076521, 0.0013304, 0.0075940, -0.0040296, 0.0040142
5: 0.9959208, 0.9976322, 0.9958758, 0.9976161, -0.0011195, 0.0011153
6: 0.0041810, 0.0057344, 0.0041402, 0.0057198, -0.0010162, 0.0010123
7: -0.0077786, -0.0019816, -0.0079311, -0.0020363, -0.0037923, 0.0037778
8: -0.0076506, -0.0031388, -0.0076080, -0.0030201, -0.0029403, 0.0029515
9: -0.0037389, -0.0033497, -0.0037492, -0.0033534, -0.0002546, 0.0002537

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005362, upper bound: 0.0005864
time: 1.68 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005386, upper bound: 0.0005818
time: 1.75 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.29 seconds
NS_A1_A1_B2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005745, upper bound: 0.0005278
NS_A1_A1_B2_A1_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005763, upper bound: 0.0005208
NS_A1_A1_B2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005696, upper bound: 0.0005226
NS_A1_A1_B2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005708, upper bound: 0.0005134
NS_A1_A1_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005798, upper bound: 0.0005278
NS_A1_A1_B2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005817, upper bound: 0.0005208
NS_A1_A1_B2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005696, upper bound: 0.0005232
NS_A1_A1_B2_A2_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005708, upper bound: 0.0005136
NS_A1_A1_B2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005798, upper bound: 0.0005281
NS_A1_A1_B2_A2_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005818, upper bound: 0.0005209
NS_A1_A2_B2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005745, upper bound: 0.0005410
NS_A1_A2_B2_A1_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005763, upper bound: 0.0005359
NS_A1_A2_B2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005696, upper bound: 0.0005365
NS_A1_A2_B2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005707, upper bound: 0.0005299
NS_A1_A2_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005798, upper bound: 0.0005411
NS_A1_A2_B2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005816, upper bound: 0.0005359
NS_A1_A2_B2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005745, upper bound: 0.0005433
NS_A1_A2_B2_A2_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005763, upper bound: 0.0005382
NS_A1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005755, upper bound: 0.0005366
NS_A1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005767, upper bound: 0.0005294
NS_A1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005798, upper bound: 0.0005433
NS_A1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005818, upper bound: 0.0005383
NS_A2_B1_B1_B1_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005277, upper bound: 0.0005745
NS_A2_B1_B1_B1_A1_A2_A2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005208, upper bound: 0.0005763
NS_A2_B1_B1_B1_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005225, upper bound: 0.0005697
NS_A2_B1_B1_B1_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005133, upper bound: 0.0005709
NS_A2_B1_B1_B1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005278, upper bound: 0.0005798
NS_A2_B1_B1_B1_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005209, upper bound: 0.0005818
NS_A2_B1_B1_B2_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005281, upper bound: 0.0005745
NS_A2_B1_B1_B2_A1_A2_A2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005209, upper bound: 0.0005763
NS_A2_B1_B1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005226, upper bound: 0.0005697
NS_A2_B1_B1_B2_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005136, upper bound: 0.0005709
NS_A2_B1_B1_B2_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005281, upper bound: 0.0005798
NS_A2_B1_B1_B2_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005209, upper bound: 0.0005818
NS_A2_B1_B2_B1_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005411, upper bound: 0.0005745
NS_A2_B1_B2_B1_A1_A2_A2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005359, upper bound: 0.0005763
NS_A2_B1_B2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005365, upper bound: 0.0005698
NS_A2_B1_B2_B1_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005299, upper bound: 0.0005709
NS_A2_B1_B2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005410, upper bound: 0.0005798
NS_A2_B1_B2_B1_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005359, upper bound: 0.0005817
NS_A2_B1_B2_B2_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005433, upper bound: 0.0005745
NS_A2_B1_B2_B2_A1_A2_A2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005383, upper bound: 0.0005763
NS_A2_B1_B2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005364, upper bound: 0.0005756
NS_A2_B1_B2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005294, upper bound: 0.0005768
NS_A2_B1_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005433, upper bound: 0.0005798
NS_A2_B1_B2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005382, upper bound: 0.0005818
NS_A2_B2_A2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005297, upper bound: 0.0005811
NS_A2_B2_A2_A1_B1_B1_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005309, upper bound: 0.0005755
NS_A2_B2_A2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005355, upper bound: 0.0005853
NS_A2_B2_A2_A1_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005376, upper bound: 0.0005806
NS_A2_B2_A2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005296, upper bound: 0.0005812
NS_A2_B2_A2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005306, upper bound: 0.0005755
NS_A2_B2_A2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005351, upper bound: 0.0005853
NS_A2_B2_A2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005385, upper bound: 0.0005806
NS_A2_B2_A2_A2_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005296, upper bound: 0.0005822
NS_A2_B2_A2_A2_B1_B1_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005309, upper bound: 0.0005768
NS_A2_B2_A2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005355, upper bound: 0.0005863
NS_A2_B2_A2_A2_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005376, upper bound: 0.0005818
NS_A2_B2_A2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005296, upper bound: 0.0005822
NS_A2_B2_A2_A2_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005306, upper bound: 0.0005766
NS_A2_B2_A2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005362, upper bound: 0.0005864
NS_A2_B2_A2_A2_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 5, lower bound: -0.0005386, upper bound: 0.0005818

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 3.86 + 404.88 = 408.74 seconds
