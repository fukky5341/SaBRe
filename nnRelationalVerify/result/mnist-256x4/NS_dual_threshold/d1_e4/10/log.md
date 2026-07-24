## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0059612499999999995


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974)
1: (-0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539)
2: (0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646)
3: (-0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001)
4: (-0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409)
5: (0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188)
6: (-0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569)
7: (0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987)
8: (-0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740)
9: (-0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.67 + 2.96 = 4.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0062749, upper bound: 0.0062749

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 186

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062569, upper bound: 0.0062173
time: 2.21 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062586, upper bound: 0.0062585
time: 1.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.81 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.81
Output dim: 7, lower bound: -0.0062569, upper bound: 0.0062173
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.81
Output dim: 7, lower bound: -0.0062586, upper bound: 0.0062585

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0037215, 0.0009741, -0.0038642, 0.0010259, -0.0047474, 0.0048383
1: -0.0057615, -0.0021505, -0.0059433, -0.0020884, -0.0036730, 0.0037928
2: 0.0306388, 0.0366429, 0.0305560, 0.0369765, -0.0063378, 0.0060868
3: -0.0031186, 0.0013226, -0.0031375, 0.0014773, -0.0045959, 0.0044601
4: -0.0047534, 0.0007608, -0.0049384, 0.0009091, -0.0056626, 0.0056992
5: 0.0093087, 0.0140828, 0.0091901, 0.0142278, -0.0049191, 0.0048927
6: -0.0069319, 0.0014377, -0.0073793, 0.0017050, -0.0086369, 0.0088170
7: 0.9718048, 0.9790652, 0.9714591, 0.9792523, -0.0074475, 0.0076061
8: -0.0161998, -0.0017576, -0.0164650, -0.0014268, -0.0147730, 0.0147074
9: -0.0029856, 0.0053936, -0.0031771, 0.0055877, -0.0085733, 0.0085707

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062473, upper bound: 0.0062095
time: 2.33 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062486, upper bound: 0.0062095
time: 1.45 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0039181, 0.0010454, -0.0039948, 0.0010733, -0.0049913, 0.0050402
1: -0.0060239, -0.0020650, -0.0060650, -0.0020317, -0.0039922, 0.0040000
2: 0.0305248, 0.0371025, 0.0304803, 0.0372821, -0.0067572, 0.0066221
3: -0.0031447, 0.0015357, -0.0031549, 0.0016190, -0.0047637, 0.0046906
4: -0.0050203, 0.0009651, -0.0050622, 0.0010449, -0.0060653, 0.0060273
5: 0.0091453, 0.0142825, 0.0090815, 0.0143605, -0.0052152, 0.0052010
6: -0.0075481, 0.0018235, -0.0077889, 0.0018840, -0.0094321, 0.0096124
7: 0.9713286, 0.9793352, 0.9711424, 0.9793775, -0.0080489, 0.0081928
8: -0.0165651, -0.0013019, -0.0167079, -0.0011238, -0.0154413, 0.0154060
9: -0.0032494, 0.0056609, -0.0033525, 0.0057654, -0.0090148, 0.0090134

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 186

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062172, upper bound: 0.0062569
time: 1.97 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062172, upper bound: 0.0062585
time: 2.07 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.70 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.70
Output dim: 7, lower bound: -0.0062473, upper bound: 0.0062095
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.70
Output dim: 7, lower bound: -0.0062486, upper bound: 0.0062095
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.70
Output dim: 7, lower bound: -0.0062172, upper bound: 0.0062569
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.70
Output dim: 7, lower bound: -0.0062172, upper bound: 0.0062585

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0036853, 0.0009609, -0.0036277, 0.0009400, -0.0046253, 0.0045886
1: -0.0057395, -0.0021662, -0.0058877, -0.0021912, -0.0035483, 0.0037215
2: 0.0306598, 0.0365581, 0.0306932, 0.0364234, -0.0057636, 0.0058649
3: -0.0031138, 0.0012833, -0.0031061, 0.0012208, -0.0043346, 0.0043894
4: -0.0047311, 0.0007232, -0.0048818, 0.0006633, -0.0053944, 0.0056050
5: 0.0093388, 0.0140460, 0.0093867, 0.0139875, -0.0046487, 0.0046593
6: -0.0068183, 0.0014054, -0.0066377, 0.0016232, -0.0084415, 0.0080431
7: 0.9718927, 0.9790427, 0.9720322, 0.9791951, -0.0073024, 0.0070105
8: -0.0161325, -0.0018417, -0.0160254, -0.0019753, -0.0141572, 0.0141837
9: -0.0029369, 0.0053443, -0.0028596, 0.0052659, -0.0082029, 0.0082039

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 186

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062173, upper bound: 0.0062095
time: 1.93 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062173, upper bound: 0.0062095
time: 2.14 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0036956, 0.0009647, -0.0037946, 0.0010006, -0.0046962, 0.0047592
1: -0.0057424, -0.0021617, -0.0058960, -0.0021187, -0.0036237, 0.0037342
2: 0.0306538, 0.0365822, 0.0305964, 0.0368137, -0.0061599, 0.0059858
3: -0.0031151, 0.0012945, -0.0031283, 0.0014018, -0.0045169, 0.0044227
4: -0.0047340, 0.0007339, -0.0048902, 0.0008368, -0.0055708, 0.0056241
5: 0.0093303, 0.0140565, 0.0092480, 0.0141570, -0.0048268, 0.0048085
6: -0.0068506, 0.0014097, -0.0071610, 0.0016354, -0.0084860, 0.0085706
7: 0.9718676, 0.9790456, 0.9716278, 0.9792036, -0.0073360, 0.0074179
8: -0.0161516, -0.0018178, -0.0163356, -0.0015883, -0.0145633, 0.0145178
9: -0.0029508, 0.0053583, -0.0030836, 0.0054930, -0.0084437, 0.0084419

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 186

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062194, upper bound: 0.0062095
time: 2.13 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062194, upper bound: 0.0062095
time: 1.90 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0039181, 0.0010454, -0.0037215, 0.0009741, -0.0048921, 0.0047670
1: -0.0060239, -0.0020650, -0.0057615, -0.0021505, -0.0038735, 0.0036964
2: 0.0305248, 0.0371025, 0.0306388, 0.0366429, -0.0061180, 0.0064637
3: -0.0031447, 0.0015357, -0.0031186, 0.0013226, -0.0044673, 0.0046543
4: -0.0050203, 0.0009651, -0.0047534, 0.0007608, -0.0057812, 0.0057185
5: 0.0091453, 0.0142825, 0.0093087, 0.0140828, -0.0049375, 0.0049738
6: -0.0075481, 0.0018235, -0.0069319, 0.0014377, -0.0089858, 0.0087554
7: 0.9713286, 0.9793352, 0.9718048, 0.9790652, -0.0077366, 0.0075305
8: -0.0165651, -0.0013019, -0.0161998, -0.0017576, -0.0148075, 0.0148979
9: -0.0032494, 0.0056609, -0.0029856, 0.0053936, -0.0086430, 0.0086465

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062473
time: 2.08 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062486
time: 2.20 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0039181, 0.0010454, -0.0039181, 0.0010454, -0.0049635, 0.0049635
1: -0.0060239, -0.0020650, -0.0060239, -0.0020650, -0.0039589, 0.0039589
2: 0.0305248, 0.0371025, 0.0305248, 0.0371025, -0.0065776, 0.0065776
3: -0.0031447, 0.0015357, -0.0031447, 0.0015357, -0.0046804, 0.0046804
4: -0.0050203, 0.0009651, -0.0050203, 0.0009651, -0.0059855, 0.0059855
5: 0.0091453, 0.0142825, 0.0091453, 0.0142825, -0.0051371, 0.0051371
6: -0.0075481, 0.0018235, -0.0075481, 0.0018235, -0.0093716, 0.0093716
7: 0.9713286, 0.9793352, 0.9713286, 0.9793352, -0.0080066, 0.0080066
8: -0.0165651, -0.0013019, -0.0165651, -0.0013019, -0.0152632, 0.0152632
9: -0.0032494, 0.0056609, -0.0032494, 0.0056609, -0.0089103, 0.0089103

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062486
time: 2.08 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062500
time: 1.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.79 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.79
Output dim: 7, lower bound: -0.0062173, upper bound: 0.0062095
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.79
Output dim: 7, lower bound: -0.0062173, upper bound: 0.0062095
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 5.79
Output dim: 7, lower bound: -0.0062194, upper bound: 0.0062095
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 5.79
Output dim: 7, lower bound: -0.0062194, upper bound: 0.0062095
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.79
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062473
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.79
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062486
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.79
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062486
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.79
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062500

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0036853, 0.0009609, -0.0034787, 0.0008860, -0.0045713, 0.0044396
1: -0.0057395, -0.0021662, -0.0057049, -0.0022560, -0.0034835, 0.0035387
2: 0.0306598, 0.0365581, 0.0307795, 0.0360750, -0.0054152, 0.0057786
3: -0.0031138, 0.0012833, -0.0030863, 0.0010593, -0.0041730, 0.0043696
4: -0.0047311, 0.0007232, -0.0046959, 0.0005084, -0.0052395, 0.0054190
5: 0.0093388, 0.0140460, 0.0095106, 0.0138361, -0.0044973, 0.0045354
6: -0.0068183, 0.0014054, -0.0061705, 0.0013545, -0.0081728, 0.0075759
7: 0.9718927, 0.9790427, 0.9723933, 0.9790071, -0.0071144, 0.0066494
8: -0.0161325, -0.0018417, -0.0157484, -0.0023208, -0.0138116, 0.0139067
9: -0.0029369, 0.0053443, -0.0026596, 0.0050632, -0.0080002, 0.0080039

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_B1_B1

### Relational analysis result of NS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061904, upper bound: 0.0061905
time: 2.15 seconds

## Relational analysis of NS_A1_B1_B1_B2

### Relational analysis result of NS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062014, upper bound: 0.0061938
time: 1.58 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0036853, 0.0009609, -0.0036824, 0.0009599, -0.0046452, 0.0046433
1: -0.0057395, -0.0021662, -0.0059691, -0.0021675, -0.0035721, 0.0038029
2: 0.0306598, 0.0365581, 0.0306614, 0.0365513, -0.0058915, 0.0058967
3: -0.0031138, 0.0012833, -0.0031134, 0.0012802, -0.0043939, 0.0043967
4: -0.0047311, 0.0007232, -0.0049646, 0.0007202, -0.0054513, 0.0056878
5: 0.0093388, 0.0140460, 0.0093412, 0.0140431, -0.0047042, 0.0047048
6: -0.0068183, 0.0014054, -0.0068092, 0.0017429, -0.0085612, 0.0082146
7: 0.9718927, 0.9790427, 0.9718997, 0.9792788, -0.0073861, 0.0071430
8: -0.0161325, -0.0018417, -0.0161271, -0.0018484, -0.0142841, 0.0142854
9: -0.0029369, 0.0053443, -0.0029330, 0.0053403, -0.0082773, 0.0082774

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061979, upper bound: 0.0061838
time: 1.94 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062014, upper bound: 0.0061938
time: 1.82 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0036956, 0.0009647, -0.0036580, 0.0009510, -0.0046466, 0.0046226
1: -0.0057424, -0.0021617, -0.0057145, -0.0021781, -0.0035643, 0.0035528
2: 0.0306538, 0.0365822, 0.0306756, 0.0364942, -0.0058404, 0.0059066
3: -0.0031151, 0.0012945, -0.0031101, 0.0012537, -0.0043688, 0.0044046
4: -0.0047340, 0.0007339, -0.0047057, 0.0006948, -0.0054288, 0.0054395
5: 0.0093303, 0.0140565, 0.0093615, 0.0140182, -0.0046880, 0.0046949
6: -0.0068506, 0.0014097, -0.0067326, 0.0013686, -0.0082192, 0.0081423
7: 0.9718676, 0.9790456, 0.9719589, 0.9790169, -0.0071493, 0.0070868
8: -0.0161516, -0.0018178, -0.0160816, -0.0019051, -0.0142466, 0.0142638
9: -0.0029508, 0.0053583, -0.0029002, 0.0053071, -0.0082579, 0.0082586

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061905
time: 2.05 seconds

## Relational analysis of NS_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062033, upper bound: 0.0061938
time: 1.76 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0036956, 0.0009647, -0.0038471, 0.0010196, -0.0047152, 0.0048118
1: -0.0057424, -0.0021617, -0.0059746, -0.0020959, -0.0036465, 0.0038129
2: 0.0306538, 0.0365822, 0.0305660, 0.0369365, -0.0062827, 0.0060162
3: -0.0031151, 0.0012945, -0.0031352, 0.0014588, -0.0045739, 0.0044297
4: -0.0047340, 0.0007339, -0.0049702, 0.0008913, -0.0056254, 0.0057041
5: 0.0093303, 0.0140565, 0.0092043, 0.0142104, -0.0048801, 0.0048521
6: -0.0068506, 0.0014097, -0.0073256, 0.0017511, -0.0086017, 0.0087353
7: 0.9718676, 0.9790456, 0.9715006, 0.9792845, -0.0074169, 0.0075451
8: -0.0161516, -0.0018178, -0.0164332, -0.0014665, -0.0146851, 0.0146154
9: -0.0029508, 0.0053583, -0.0031541, 0.0055644, -0.0085152, 0.0085125

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061994, upper bound: 0.0061838
time: 1.28 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062033, upper bound: 0.0061937
time: 1.69 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0036824, 0.0009599, -0.0036853, 0.0009609, -0.0046433, 0.0046452
1: -0.0059691, -0.0021675, -0.0057395, -0.0021662, -0.0038029, 0.0035721
2: 0.0306614, 0.0365513, 0.0306598, 0.0365581, -0.0058967, 0.0058915
3: -0.0031134, 0.0012802, -0.0031138, 0.0012833, -0.0043967, 0.0043939
4: -0.0049646, 0.0007202, -0.0047311, 0.0007232, -0.0056878, 0.0054513
5: 0.0093412, 0.0140431, 0.0093388, 0.0140460, -0.0047048, 0.0047042
6: -0.0068092, 0.0017429, -0.0068183, 0.0014054, -0.0082146, 0.0085612
7: 0.9718997, 0.9792788, 0.9718927, 0.9790427, -0.0071430, 0.0073861
8: -0.0161271, -0.0018484, -0.0161325, -0.0018417, -0.0142854, 0.0142841
9: -0.0029330, 0.0053403, -0.0029369, 0.0053443, -0.0082774, 0.0082773

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061838, upper bound: 0.0062289
time: 2.02 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061937, upper bound: 0.0062314
time: 1.91 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0038471, 0.0010196, -0.0036956, 0.0009647, -0.0048118, 0.0047152
1: -0.0059746, -0.0020959, -0.0057424, -0.0021617, -0.0038129, 0.0036465
2: 0.0305660, 0.0369365, 0.0306538, 0.0365822, -0.0060162, 0.0062827
3: -0.0031352, 0.0014588, -0.0031151, 0.0012945, -0.0044297, 0.0045739
4: -0.0049702, 0.0008913, -0.0047340, 0.0007339, -0.0057041, 0.0056254
5: 0.0092043, 0.0142104, 0.0093303, 0.0140565, -0.0048521, 0.0048801
6: -0.0073256, 0.0017511, -0.0068506, 0.0014097, -0.0087353, 0.0086017
7: 0.9715006, 0.9792845, 0.9718676, 0.9790456, -0.0075451, 0.0074169
8: -0.0164332, -0.0014665, -0.0161516, -0.0018178, -0.0146154, 0.0146851
9: -0.0031541, 0.0055644, -0.0029508, 0.0053583, -0.0085125, 0.0085152

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061838, upper bound: 0.0062299
time: 1.76 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061937, upper bound: 0.0062327
time: 1.97 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0036824, 0.0009599, -0.0038784, 0.0010310, -0.0047134, 0.0048383
1: -0.0059691, -0.0021675, -0.0060027, -0.0020823, -0.0038868, 0.0038353
2: 0.0306614, 0.0365513, 0.0305478, 0.0370097, -0.0063483, 0.0060035
3: -0.0031134, 0.0012802, -0.0031394, 0.0014927, -0.0046061, 0.0044196
4: -0.0049646, 0.0007202, -0.0049988, 0.0009239, -0.0058885, 0.0057190
5: 0.0093412, 0.0140431, 0.0091783, 0.0142422, -0.0049009, 0.0048648
6: -0.0068092, 0.0017429, -0.0074237, 0.0017924, -0.0086016, 0.0091666
7: 0.9718997, 0.9792788, 0.9714248, 0.9793135, -0.0074138, 0.0078540
8: -0.0161271, -0.0018484, -0.0164914, -0.0013939, -0.0147332, 0.0146430
9: -0.0029330, 0.0053403, -0.0031961, 0.0056070, -0.0085400, 0.0085365

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061936, upper bound: 0.0062196
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061976, upper bound: 0.0062327
time: 1.76 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0038471, 0.0010196, -0.0038893, 0.0010350, -0.0048821, 0.0049089
1: -0.0059746, -0.0020959, -0.0060039, -0.0020775, -0.0038971, 0.0039080
2: 0.0305660, 0.0369365, 0.0305415, 0.0370353, -0.0064693, 0.0063950
3: -0.0031352, 0.0014588, -0.0031409, 0.0015046, -0.0046398, 0.0045996
4: -0.0049702, 0.0008913, -0.0050000, 0.0009352, -0.0059055, 0.0058913
5: 0.0092043, 0.0142104, 0.0091692, 0.0142533, -0.0050490, 0.0050412
6: -0.0073256, 0.0017511, -0.0074580, 0.0017941, -0.0091197, 0.0092091
7: 0.9715006, 0.9792845, 0.9713982, 0.9793146, -0.0078140, 0.0078864
8: -0.0164332, -0.0014665, -0.0165117, -0.0013685, -0.0150647, 0.0150452
9: -0.0031541, 0.0055644, -0.0032108, 0.0056219, -0.0087760, 0.0087752

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061936, upper bound: 0.0062198
time: 2.12 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061977, upper bound: 0.0062340
time: 1.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.21 seconds
NS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0061904, upper bound: 0.0061905
NS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0062014, upper bound: 0.0061938
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0061979, upper bound: 0.0061838
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0062014, upper bound: 0.0061938
NS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061905
NS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0062033, upper bound: 0.0061938
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0061994, upper bound: 0.0061838
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0062033, upper bound: 0.0061937
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0061838, upper bound: 0.0062289
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0061937, upper bound: 0.0062314
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0061838, upper bound: 0.0062299
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0061937, upper bound: 0.0062327
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0061936, upper bound: 0.0062196
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0061976, upper bound: 0.0062327
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0061936, upper bound: 0.0062198
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 7, lower bound: -0.0061977, upper bound: 0.0062340

## BFS NS instance: NS_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0033662, 0.0008451, -0.0026877, 0.0007237, -0.0040899, 0.0035328
1: -0.0057376, -0.0023049, -0.0057992, -0.0024742, -0.0030115, 0.0034942
2: 0.0308448, 0.0358118, 0.0312381, 0.0347775, -0.0039327, 0.0045737
3: -0.0030714, 0.0009372, -0.0029812, 0.0008706, -0.0039420, 0.0039184
4: -0.0047292, 0.0003914, -0.0047917, -0.0000998, -0.0046293, 0.0051832
5: 0.0096041, 0.0137218, 0.0101683, 0.0133703, -0.0037662, 0.0035535
6: -0.0058177, 0.0014026, -0.0047067, 0.0014931, -0.0073107, 0.0061093
7: 0.9726660, 0.9790407, 0.9738801, 0.9791039, -0.0064379, 0.0051606
8: -0.0155392, -0.0025818, -0.0144739, -0.0041554, -0.0113838, 0.0118921
9: -0.0025085, 0.0049102, -0.0015976, 0.0042505, -0.0067590, 0.0065078

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061754, upper bound: 0.0061834
time: 1.93 seconds

## Relational analysis of NS_A1_B1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061739, upper bound: 0.0061833
time: 2.19 seconds

## BFS NS instance: NS_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0036550, 0.0009499, -0.0031220, 0.0007565, -0.0044115, 0.0040719
1: -0.0057395, -0.0021794, -0.0057040, -0.0024111, -0.0033284, 0.0035246
2: 0.0306773, 0.0364872, 0.0309863, 0.0352406, -0.0045633, 0.0055009
3: -0.0031097, 0.0012504, -0.0030389, 0.0007604, -0.0038701, 0.0042893
4: -0.0047310, 0.0006917, -0.0046950, 0.0001376, -0.0048686, 0.0053866
5: 0.0093640, 0.0140152, 0.0098072, 0.0134737, -0.0041096, 0.0042080
6: -0.0067232, 0.0014053, -0.0050519, 0.0013532, -0.0080764, 0.0064572
7: 0.9719661, 0.9790426, 0.9732578, 0.9790061, -0.0070400, 0.0057847
8: -0.0160761, -0.0019120, -0.0150852, -0.0031481, -0.0129280, 0.0131733
9: -0.0028963, 0.0053031, -0.0021807, 0.0045779, -0.0074742, 0.0074838

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_B1_B2_B1

### Relational analysis result of NS_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061894, upper bound: 0.0061919
time: 2.29 seconds

## Relational analysis of NS_A1_B1_B1_B2_B2

### Relational analysis result of NS_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061894, upper bound: 0.0061911
time: 2.21 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0029020, 0.0007333, -0.0033599, 0.0008429, -0.0037449, 0.0040933
1: -0.0058337, -0.0024496, -0.0059674, -0.0023076, -0.0035260, 0.0031347
2: 0.0311139, 0.0349772, 0.0308484, 0.0357972, -0.0046833, 0.0041288
3: -0.0030097, 0.0009106, -0.0030705, 0.0010655, -0.0040752, 0.0039811
4: -0.0048268, 0.0000061, -0.0049629, 0.0003849, -0.0052118, 0.0049690
5: 0.0099900, 0.0134036, 0.0096093, 0.0137154, -0.0037254, 0.0037943
6: -0.0048239, 0.0015438, -0.0057981, 0.0017404, -0.0065644, 0.0073419
7: 0.9735956, 0.9791394, 0.9726812, 0.9792771, -0.0056815, 0.0064582
8: -0.0147654, -0.0036583, -0.0155276, -0.0025963, -0.0121691, 0.0118693
9: -0.0018854, 0.0043983, -0.0025001, 0.0049016, -0.0067871, 0.0068984

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061992, upper bound: 0.0061628
time: 1.69 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062070, upper bound: 0.0061566
time: 1.74 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0033245, 0.0008300, -0.0036529, 0.0009492, -0.0042737, 0.0044829
1: -0.0057387, -0.0023230, -0.0059690, -0.0021803, -0.0035584, 0.0036460
2: 0.0308689, 0.0357144, 0.0306786, 0.0364824, -0.0056134, 0.0050358
3: -0.0030658, 0.0008920, -0.0031094, 0.0012482, -0.0043140, 0.0040015
4: -0.0047302, 0.0003481, -0.0049645, 0.0006895, -0.0054197, 0.0053127
5: 0.0096388, 0.0136795, 0.0093658, 0.0140131, -0.0043743, 0.0043137
6: -0.0056870, 0.0014041, -0.0067168, 0.0017428, -0.0074299, 0.0081209
7: 0.9727671, 0.9790418, 0.9719711, 0.9792788, -0.0065117, 0.0070707
8: -0.0154618, -0.0026784, -0.0160723, -0.0019168, -0.0135450, 0.0133938
9: -0.0024526, 0.0048535, -0.0028934, 0.0053002, -0.0077528, 0.0077469

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062214, upper bound: 0.0061821
time: 2.09 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062189, upper bound: 0.0061821
time: 1.88 seconds

## BFS NS instance: NS_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0033766, 0.0008489, -0.0028785, 0.0007323, -0.0041089, 0.0037275
1: -0.0057405, -0.0023004, -0.0058096, -0.0024523, -0.0030455, 0.0035092
2: 0.0308387, 0.0358362, 0.0311275, 0.0349553, -0.0041166, 0.0047087
3: -0.0030727, 0.0009486, -0.0030066, 0.0008826, -0.0039554, 0.0039551
4: -0.0047321, 0.0004023, -0.0048023, -0.0000055, -0.0047266, 0.0052046
5: 0.0095955, 0.0137324, 0.0100096, 0.0134000, -0.0038045, 0.0037228
6: -0.0058504, 0.0014068, -0.0048111, 0.0015083, -0.0073588, 0.0062179
7: 0.9726407, 0.9790437, 0.9736267, 0.9791147, -0.0064740, 0.0054170
8: -0.0155587, -0.0025576, -0.0147334, -0.0037128, -0.0118459, 0.0121758
9: -0.0025226, 0.0049244, -0.0018539, 0.0043821, -0.0069047, 0.0067782

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_B1_B1_A1

### Relational analysis result of NS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061761, upper bound: 0.0061834
time: 2.08 seconds

## Relational analysis of NS_A1_B2_B1_B1_A2

### Relational analysis result of NS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061745, upper bound: 0.0061833
time: 2.07 seconds

## BFS NS instance: NS_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0036653, 0.0009537, -0.0032971, 0.0008200, -0.0044853, 0.0042507
1: -0.0057423, -0.0021749, -0.0057136, -0.0023350, -0.0034074, 0.0035387
2: 0.0306714, 0.0365112, 0.0308849, 0.0356500, -0.0049787, 0.0056263
3: -0.0031111, 0.0012616, -0.0030622, 0.0008622, -0.0039733, 0.0043237
4: -0.0047339, 0.0007023, -0.0047047, 0.0003195, -0.0050535, 0.0054071
5: 0.0093555, 0.0140256, 0.0096616, 0.0136515, -0.0042960, 0.0043640
6: -0.0067555, 0.0014095, -0.0056008, 0.0013673, -0.0081228, 0.0070104
7: 0.9719412, 0.9790455, 0.9728337, 0.9790161, -0.0070748, 0.0062118
8: -0.0160952, -0.0018882, -0.0154107, -0.0027422, -0.0133530, 0.0135225
9: -0.0029100, 0.0053170, -0.0024157, 0.0048161, -0.0077261, 0.0077327

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_B1_B2_B1

### Relational analysis result of NS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061919
time: 1.44 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2

### Relational analysis result of NS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061911
time: 1.66 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0029139, 0.0007339, -0.0035353, 0.0009065, -0.0038204, 0.0042692
1: -0.0058370, -0.0024482, -0.0059729, -0.0022314, -0.0036056, 0.0031911
2: 0.0311070, 0.0349882, 0.0307467, 0.0362073, -0.0051003, 0.0042415
3: -0.0030113, 0.0009145, -0.0030938, 0.0011206, -0.0041319, 0.0040083
4: -0.0048303, 0.0000120, -0.0049684, 0.0005672, -0.0053975, 0.0049804
5: 0.0099802, 0.0134055, 0.0094635, 0.0138936, -0.0039134, 0.0039420
6: -0.0048304, 0.0015487, -0.0063480, 0.0017485, -0.0065789, 0.0078967
7: 0.9735798, 0.9791430, 0.9722562, 0.9792826, -0.0057028, 0.0068868
8: -0.0147815, -0.0036308, -0.0158536, -0.0021896, -0.0125919, 0.0122228
9: -0.0019013, 0.0044065, -0.0027356, 0.0051402, -0.0070415, 0.0071421

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061996, upper bound: 0.0061628
time: 1.92 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062080, upper bound: 0.0061566
time: 1.96 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0033347, 0.0008337, -0.0038170, 0.0010087, -0.0043434, 0.0046507
1: -0.0057415, -0.0023186, -0.0059746, -0.0021090, -0.0036326, 0.0036560
2: 0.0308630, 0.0357382, 0.0305834, 0.0368662, -0.0060032, 0.0051548
3: -0.0030672, 0.0009031, -0.0031313, 0.0014261, -0.0044933, 0.0040343
4: -0.0047331, 0.0003587, -0.0049702, 0.0008601, -0.0055932, 0.0053289
5: 0.0096303, 0.0136898, 0.0092293, 0.0141798, -0.0045495, 0.0044605
6: -0.0057190, 0.0014083, -0.0072313, 0.0017510, -0.0074699, 0.0086396
7: 0.9727424, 0.9790447, 0.9715734, 0.9792844, -0.0065420, 0.0074713
8: -0.0154807, -0.0026548, -0.0163773, -0.0015362, -0.0139445, 0.0137225
9: -0.0024663, 0.0048673, -0.0031137, 0.0055235, -0.0079898, 0.0079811

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062228, upper bound: 0.0061822
time: 1.36 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062199, upper bound: 0.0061822
time: 1.93 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0033599, 0.0008429, -0.0029020, 0.0007333, -0.0040933, 0.0037449
1: -0.0059674, -0.0023076, -0.0058337, -0.0024496, -0.0031347, 0.0035260
2: 0.0308484, 0.0357972, 0.0311139, 0.0349772, -0.0041288, 0.0046833
3: -0.0030705, 0.0010655, -0.0030097, 0.0009106, -0.0039811, 0.0040752
4: -0.0049629, 0.0003849, -0.0048268, 0.0000061, -0.0049690, 0.0052118
5: 0.0096093, 0.0137154, 0.0099900, 0.0134036, -0.0037943, 0.0037254
6: -0.0057981, 0.0017404, -0.0048239, 0.0015438, -0.0073419, 0.0065644
7: 0.9726812, 0.9792771, 0.9735956, 0.9791394, -0.0064582, 0.0056815
8: -0.0155276, -0.0025963, -0.0147654, -0.0036583, -0.0118693, 0.0121691
9: -0.0025001, 0.0049016, -0.0018854, 0.0043983, -0.0068984, 0.0067871

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061628, upper bound: 0.0061992
time: 2.02 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061567, upper bound: 0.0062070
time: 1.93 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0036529, 0.0009492, -0.0033245, 0.0008300, -0.0044829, 0.0042737
1: -0.0059690, -0.0021803, -0.0057387, -0.0023230, -0.0036460, 0.0035584
2: 0.0306786, 0.0364824, 0.0308689, 0.0357144, -0.0050358, 0.0056134
3: -0.0031094, 0.0012482, -0.0030658, 0.0008920, -0.0040015, 0.0043140
4: -0.0049645, 0.0006895, -0.0047302, 0.0003481, -0.0053127, 0.0054197
5: 0.0093658, 0.0140131, 0.0096388, 0.0136795, -0.0043137, 0.0043743
6: -0.0067168, 0.0017428, -0.0056870, 0.0014041, -0.0081209, 0.0074299
7: 0.9719711, 0.9792788, 0.9727671, 0.9790418, -0.0070707, 0.0065117
8: -0.0160723, -0.0019168, -0.0154618, -0.0026784, -0.0133938, 0.0135450
9: -0.0028934, 0.0053002, -0.0024526, 0.0048535, -0.0077469, 0.0077528

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061821, upper bound: 0.0062215
time: 2.40 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061822, upper bound: 0.0062189
time: 2.04 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0035353, 0.0009065, -0.0029139, 0.0007339, -0.0042692, 0.0038204
1: -0.0059729, -0.0022314, -0.0058370, -0.0024482, -0.0031911, 0.0036056
2: 0.0307467, 0.0362073, 0.0311070, 0.0349882, -0.0042415, 0.0051003
3: -0.0030938, 0.0011206, -0.0030113, 0.0009145, -0.0040083, 0.0041319
4: -0.0049684, 0.0005672, -0.0048303, 0.0000120, -0.0049804, 0.0053975
5: 0.0094635, 0.0138936, 0.0099802, 0.0134055, -0.0039420, 0.0039134
6: -0.0063480, 0.0017485, -0.0048304, 0.0015487, -0.0078967, 0.0065789
7: 0.9722562, 0.9792826, 0.9735798, 0.9791430, -0.0068868, 0.0057028
8: -0.0158536, -0.0021896, -0.0147815, -0.0036308, -0.0122228, 0.0125919
9: -0.0027356, 0.0051402, -0.0019013, 0.0044065, -0.0071421, 0.0070415

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061628, upper bound: 0.0061996
time: 1.76 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061567, upper bound: 0.0062079
time: 2.15 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0038170, 0.0010087, -0.0033347, 0.0008337, -0.0046507, 0.0043434
1: -0.0059746, -0.0021090, -0.0057415, -0.0023186, -0.0036560, 0.0036326
2: 0.0305834, 0.0368662, 0.0308630, 0.0357382, -0.0051548, 0.0060032
3: -0.0031313, 0.0014261, -0.0030672, 0.0009031, -0.0040343, 0.0044933
4: -0.0049702, 0.0008601, -0.0047331, 0.0003587, -0.0053289, 0.0055932
5: 0.0092293, 0.0141798, 0.0096303, 0.0136898, -0.0044605, 0.0045495
6: -0.0072313, 0.0017510, -0.0057190, 0.0014083, -0.0086396, 0.0074699
7: 0.9715734, 0.9792844, 0.9727424, 0.9790447, -0.0074713, 0.0065420
8: -0.0163773, -0.0015362, -0.0154807, -0.0026548, -0.0137225, 0.0139445
9: -0.0031137, 0.0055235, -0.0024663, 0.0048673, -0.0079811, 0.0079898

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061821, upper bound: 0.0062228
time: 2.07 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061821, upper bound: 0.0062199
time: 1.94 seconds

## BFS NS instance: NS_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0029214, 0.0007342, -0.0035664, 0.0009178, -0.0038392, 0.0043006
1: -0.0060629, -0.0024474, -0.0060010, -0.0022179, -0.0038450, 0.0031627
2: 0.0311026, 0.0349953, 0.0307287, 0.0362800, -0.0051773, 0.0042665
3: -0.0030123, 0.0011761, -0.0030980, 0.0011543, -0.0041666, 0.0042741
4: -0.0050600, 0.0000157, -0.0049970, 0.0005995, -0.0056595, 0.0050128
5: 0.0099739, 0.0134067, 0.0094377, 0.0139252, -0.0039512, 0.0039690
6: -0.0048346, 0.0018808, -0.0064454, 0.0017898, -0.0066243, 0.0083262
7: 0.9735698, 0.9793753, 0.9721808, 0.9793116, -0.0057418, 0.0071945
8: -0.0147918, -0.0036133, -0.0159114, -0.0021175, -0.0126742, 0.0122981
9: -0.0019115, 0.0044117, -0.0027773, 0.0051825, -0.0070940, 0.0071890

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061810, upper bound: 0.0062087
time: 2.02 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061810, upper bound: 0.0062054
time: 1.89 seconds

## BFS NS instance: NS_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0033317, 0.0008326, -0.0038483, 0.0010201, -0.0043518, 0.0046809
1: -0.0059683, -0.0023199, -0.0060027, -0.0020954, -0.0038729, 0.0036828
2: 0.0308647, 0.0357312, 0.0305653, 0.0369393, -0.0060746, 0.0051659
3: -0.0030668, 0.0010665, -0.0031354, 0.0014601, -0.0045269, 0.0042019
4: -0.0049638, 0.0003556, -0.0049988, 0.0008926, -0.0058564, 0.0053544
5: 0.0096328, 0.0136868, 0.0092033, 0.0142116, -0.0045788, 0.0044835
6: -0.0057097, 0.0017417, -0.0073294, 0.0017923, -0.0075019, 0.0090712
7: 0.9727495, 0.9792778, 0.9714977, 0.9793134, -0.0065639, 0.0077802
8: -0.0154752, -0.0026617, -0.0164355, -0.0014636, -0.0140115, 0.0137738
9: -0.0024623, 0.0048633, -0.0031558, 0.0055661, -0.0080283, 0.0080190

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A2_B2_A1_A2_A1

### Relational analysis result of NS_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061863, upper bound: 0.0062210
time: 1.97 seconds

## Relational analysis of NS_A2_B2_A1_A2_A2

### Relational analysis result of NS_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061869, upper bound: 0.0062212
time: 2.38 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0031174, 0.0007548, -0.0035774, 0.0009218, -0.0040391, 0.0043322
1: -0.0060706, -0.0024131, -0.0060021, -0.0022131, -0.0038575, 0.0035891
2: 0.0309890, 0.0352298, 0.0307223, 0.0363057, -0.0053167, 0.0045075
3: -0.0030383, 0.0011851, -0.0030994, 0.0011663, -0.0042046, 0.0042845
4: -0.0050678, 0.0001328, -0.0049982, 0.0006110, -0.0056788, 0.0051310
5: 0.0098110, 0.0134690, 0.0094286, 0.0139364, -0.0041253, 0.0040404
6: -0.0050375, 0.0018921, -0.0064799, 0.0017915, -0.0068290, 0.0083720
7: 0.9732691, 0.9793832, 0.9721541, 0.9793129, -0.0060438, 0.0072291
8: -0.0150767, -0.0031588, -0.0159318, -0.0020920, -0.0129847, 0.0127730
9: -0.0021745, 0.0045717, -0.0027921, 0.0051975, -0.0073720, 0.0073637

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061810, upper bound: 0.0062088
time: 2.01 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061810, upper bound: 0.0062054
time: 2.08 seconds

## BFS NS instance: NS_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0034918, 0.0008907, -0.0038592, 0.0010241, -0.0045159, 0.0047499
1: -0.0059738, -0.0022503, -0.0060038, -0.0020906, -0.0038832, 0.0037535
2: 0.0307720, 0.0361055, 0.0305589, 0.0369649, -0.0061930, 0.0055466
3: -0.0030880, 0.0010734, -0.0031369, 0.0014719, -0.0045600, 0.0042103
4: -0.0049694, 0.0005220, -0.0049999, 0.0009040, -0.0058734, 0.0055219
5: 0.0094997, 0.0138494, 0.0091942, 0.0142227, -0.0047230, 0.0046552
6: -0.0062115, 0.0017498, -0.0073637, 0.0017940, -0.0080055, 0.0091135
7: 0.9723616, 0.9792837, 0.9714712, 0.9793146, -0.0069531, 0.0078125
8: -0.0157727, -0.0022905, -0.0164558, -0.0014383, -0.0143344, 0.0141653
9: -0.0026772, 0.0050810, -0.0031704, 0.0055809, -0.0082581, 0.0082515

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A2_B2_A2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061863, upper bound: 0.0062221
time: 1.81 seconds

## Relational analysis of NS_A2_B2_A2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061870, upper bound: 0.0062222
time: 1.99 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.40 seconds
NS_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061754, upper bound: 0.0061834
NS_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061739, upper bound: 0.0061833
NS_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061894, upper bound: 0.0061919
NS_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061894, upper bound: 0.0061911
NS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061992, upper bound: 0.0061628
NS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0062070, upper bound: 0.0061566
NS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0062214, upper bound: 0.0061821
NS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0062189, upper bound: 0.0061821
NS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061761, upper bound: 0.0061834
NS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061745, upper bound: 0.0061833
NS_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061919
NS_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061911
NS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061996, upper bound: 0.0061628
NS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0062080, upper bound: 0.0061566
NS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0062228, upper bound: 0.0061822
NS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0062199, upper bound: 0.0061822
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061628, upper bound: 0.0061992
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061567, upper bound: 0.0062070
NS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061821, upper bound: 0.0062215
NS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061822, upper bound: 0.0062189
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061628, upper bound: 0.0061996
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061567, upper bound: 0.0062079
NS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061821, upper bound: 0.0062228
NS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061821, upper bound: 0.0062199
NS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061810, upper bound: 0.0062087
NS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061810, upper bound: 0.0062054
NS_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061863, upper bound: 0.0062210
NS_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061869, upper bound: 0.0062212
NS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061810, upper bound: 0.0062088
NS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061810, upper bound: 0.0062054
NS_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061863, upper bound: 0.0062221
NS_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 7, lower bound: -0.0061870, upper bound: 0.0062222

## BFS NS instance: NS_A1_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0030153, 0.0007385, -0.0025658, 0.0007182, -0.0037334, 0.0033043
1: -0.0057371, -0.0024366, -0.0057990, -0.0024882, -0.0028732, 0.0028348
2: 0.0310482, 0.0350827, 0.0313088, 0.0346640, -0.0036158, 0.0037739
3: -0.0030247, 0.0007987, -0.0029650, 0.0008704, -0.0038951, 0.0037638
4: -0.0047286, 0.0000621, -0.0047916, -0.0001601, -0.0045685, 0.0048537
5: 0.0098959, 0.0134213, 0.0102696, 0.0133513, -0.0034555, 0.0031517
6: -0.0048859, 0.0014018, -0.0046401, 0.0014928, -0.0063787, 0.0060419
7: 0.9734451, 0.9790401, 0.9740420, 0.9791037, -0.0056586, 0.0049981
8: -0.0149194, -0.0033956, -0.0143082, -0.0044380, -0.0104814, 0.0109126
9: -0.0020375, 0.0044764, -0.0014341, 0.0041664, -0.0062039, 0.0059105

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061303, upper bound: 0.0061604
time: 1.82 seconds

## Relational analysis of NS_A1_B1_B1_B1_A1_B2

### Relational analysis result of NS_A1_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061313, upper bound: 0.0061452
time: 2.00 seconds

## BFS NS instance: NS_A1_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0028938, 0.0007330, -0.0024777, 0.0007142, -0.0036080, 0.0032106
1: -0.0057670, -0.0024505, -0.0057988, -0.0024983, -0.0028612, 0.0028054
2: 0.0311186, 0.0349695, 0.0313599, 0.0345818, -0.0034632, 0.0036096
3: -0.0030086, 0.0008334, -0.0029533, 0.0008702, -0.0038788, 0.0037867
4: -0.0047590, 0.0000021, -0.0047914, -0.0002037, -0.0045553, 0.0047934
5: 0.0099969, 0.0134024, 0.0103429, 0.0133376, -0.0033408, 0.0030595
6: -0.0048195, 0.0014458, -0.0045919, 0.0014925, -0.0063120, 0.0060377
7: 0.9736064, 0.9790709, 0.9741590, 0.9791035, -0.0054971, 0.0049119
8: -0.0147542, -0.0036773, -0.0141883, -0.0046425, -0.0101117, 0.0105110
9: -0.0018744, 0.0043927, -0.0013157, 0.0041056, -0.0059800, 0.0057084

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061279, upper bound: 0.0061602
time: 2.01 seconds

## Relational analysis of NS_A1_B1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061279, upper bound: 0.0061434
time: 1.91 seconds

## BFS NS instance: NS_A1_B1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0035362, 0.0009068, -0.0027736, 0.0007275, -0.0042638, 0.0036804
1: -0.0057393, -0.0022310, -0.0057035, -0.0024643, -0.0031229, 0.0034725
2: 0.0307462, 0.0362095, 0.0311883, 0.0348575, -0.0041113, 0.0050211
3: -0.0030939, 0.0011216, -0.0029926, 0.0007598, -0.0038538, 0.0041143
4: -0.0047309, 0.0005682, -0.0046945, -0.0000574, -0.0046735, 0.0052627
5: 0.0094628, 0.0138945, 0.0100968, 0.0133837, -0.0039209, 0.0037977
6: -0.0063509, 0.0014051, -0.0047537, 0.0013525, -0.0077033, 0.0061587
7: 0.9722539, 0.9790424, 0.9737661, 0.9790057, -0.0067518, 0.0052763
8: -0.0158553, -0.0021874, -0.0145907, -0.0039562, -0.0118991, 0.0124033
9: -0.0027368, 0.0051415, -0.0017130, 0.0043097, -0.0070465, 0.0068545

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_B1_B2_B1_A1

### Relational analysis result of NS_A1_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061735, upper bound: 0.0061678
time: 2.31 seconds

## Relational analysis of NS_A1_B1_B1_B2_B1_A2

### Relational analysis result of NS_A1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061735, upper bound: 0.0061764
time: 1.48 seconds

## BFS NS instance: NS_A1_B1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0034393, 0.0008717, -0.0026485, 0.0007219, -0.0041612, 0.0035201
1: -0.0057391, -0.0022731, -0.0057338, -0.0024787, -0.0030802, 0.0034607
2: 0.0308024, 0.0359827, 0.0312609, 0.0347410, -0.0039386, 0.0047218
3: -0.0030811, 0.0010165, -0.0029760, 0.0007950, -0.0038760, 0.0039925
4: -0.0047307, 0.0004674, -0.0047253, -0.0001192, -0.0046114, 0.0051927
5: 0.0095434, 0.0137960, 0.0102008, 0.0133642, -0.0038208, 0.0035952
6: -0.0060468, 0.0014048, -0.0046853, 0.0013970, -0.0074438, 0.0060901
7: 0.9724889, 0.9790422, 0.9739322, 0.9790369, -0.0065479, 0.0051100
8: -0.0156751, -0.0024123, -0.0144206, -0.0042463, -0.0114288, 0.0120083
9: -0.0026066, 0.0050096, -0.0015450, 0.0042235, -0.0068301, 0.0065546

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_B1_B2_B2_B1

### Relational analysis result of NS_A1_B1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061666, upper bound: 0.0061754
time: 2.11 seconds

## Relational analysis of NS_A1_B1_B1_B2_B2_B2

### Relational analysis result of NS_A1_B1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061735, upper bound: 0.0061754
time: 2.06 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0028817, 0.0007324, -0.0032932, 0.0008186, -0.0037003, 0.0040256
1: -0.0058189, -0.0024519, -0.0059164, -0.0023366, -0.0034823, 0.0030794
2: 0.0311257, 0.0349582, 0.0308871, 0.0356411, -0.0045154, 0.0040711
3: -0.0030070, 0.0008935, -0.0030617, 0.0010064, -0.0040134, 0.0039552
4: -0.0048119, -0.0000039, -0.0049110, 0.0003156, -0.0051274, 0.0049071
5: 0.0100070, 0.0134005, 0.0096648, 0.0136476, -0.0036407, 0.0037357
6: -0.0048128, 0.0015221, -0.0055888, 0.0016654, -0.0064783, 0.0071110
7: 0.9736226, 0.9791244, 0.9728429, 0.9792246, -0.0056020, 0.0062815
8: -0.0147377, -0.0037054, -0.0154035, -0.0027511, -0.0119867, 0.0116981
9: -0.0018581, 0.0043843, -0.0024105, 0.0048109, -0.0066690, 0.0067948

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061836, upper bound: 0.0061438
time: 2.14 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061820, upper bound: 0.0061428
time: 1.88 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0028536, 0.0007312, -0.0039327, 0.0010507, -0.0039043, 0.0046639
1: -0.0058073, -0.0024551, -0.0059090, -0.0020587, -0.0037486, 0.0032837
2: 0.0311419, 0.0349321, 0.0305163, 0.0371367, -0.0059948, 0.0044157
3: -0.0030033, 0.0008801, -0.0031466, 0.0015516, -0.0045549, 0.0040267
4: -0.0048000, -0.0000178, -0.0049034, 0.0009803, -0.0057804, 0.0048856
5: 0.0100303, 0.0133961, 0.0091331, 0.0142974, -0.0042671, 0.0042630
6: -0.0047975, 0.0015050, -0.0075941, 0.0016545, -0.0064520, 0.0090991
7: 0.9736597, 0.9791124, 0.9712931, 0.9792171, -0.0055574, 0.0078193
8: -0.0146996, -0.0037705, -0.0165924, -0.0012679, -0.0134317, 0.0128219
9: -0.0018205, 0.0043649, -0.0032691, 0.0056809, -0.0075013, 0.0076340

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061878, upper bound: 0.0061303
time: 2.05 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061878, upper bound: 0.0061286
time: 2.06 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0029800, 0.0007369, -0.0035330, 0.0009057, -0.0038856, 0.0042699
1: -0.0057381, -0.0024407, -0.0059689, -0.0022324, -0.0035057, 0.0032453
2: 0.0310687, 0.0350498, 0.0307481, 0.0362019, -0.0051333, 0.0043017
3: -0.0030200, 0.0007999, -0.0030935, 0.0011181, -0.0041382, 0.0038934
4: -0.0047297, 0.0000447, -0.0049644, 0.0005648, -0.0052945, 0.0050090
5: 0.0099253, 0.0134158, 0.0094655, 0.0138913, -0.0039660, 0.0039503
6: -0.0048666, 0.0014034, -0.0063408, 0.0017426, -0.0066091, 0.0077441
7: 0.9734921, 0.9790412, 0.9722618, 0.9792786, -0.0057865, 0.0067794
8: -0.0148713, -0.0034775, -0.0158493, -0.0021949, -0.0126764, 0.0123718
9: -0.0019900, 0.0044521, -0.0027325, 0.0051371, -0.0071272, 0.0071845

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061968, upper bound: 0.0061674
time: 2.07 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062079, upper bound: 0.0061674
time: 2.34 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0028483, 0.0007309, -0.0034139, 0.0008624, -0.0037108, 0.0041448
1: -0.0057680, -0.0024558, -0.0059687, -0.0022842, -0.0034838, 0.0032040
2: 0.0311450, 0.0349272, 0.0308171, 0.0359233, -0.0047783, 0.0041100
3: -0.0030026, 0.0008345, -0.0030777, 0.0010670, -0.0040696, 0.0039122
4: -0.0047601, -0.0000204, -0.0049642, 0.0004410, -0.0052011, 0.0049438
5: 0.0100347, 0.0133953, 0.0095645, 0.0137702, -0.0037356, 0.0038308
6: -0.0047946, 0.0014473, -0.0059672, 0.0017423, -0.0065369, 0.0074145
7: 0.9736668, 0.9790720, 0.9725504, 0.9792784, -0.0056116, 0.0065216
8: -0.0146924, -0.0037827, -0.0156279, -0.0024712, -0.0122212, 0.0118451
9: -0.0018134, 0.0043613, -0.0025725, 0.0049750, -0.0067884, 0.0069338

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061954, upper bound: 0.0061674
time: 2.24 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062049, upper bound: 0.0061674
time: 1.87 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0030256, 0.0007389, -0.0027568, 0.0007268, -0.0037524, 0.0034957
1: -0.0057400, -0.0024354, -0.0058094, -0.0024663, -0.0029074, 0.0028931
2: 0.0310422, 0.0350923, 0.0311981, 0.0348419, -0.0037997, 0.0038942
3: -0.0030261, 0.0008021, -0.0029904, 0.0008824, -0.0039085, 0.0037924
4: -0.0047316, 0.0000672, -0.0048021, -0.0000657, -0.0046659, 0.0048694
5: 0.0098873, 0.0134229, 0.0101108, 0.0133811, -0.0034938, 0.0033121
6: -0.0048915, 0.0014061, -0.0047445, 0.0015081, -0.0063996, 0.0061506
7: 0.9734314, 0.9790432, 0.9737883, 0.9791146, -0.0056832, 0.0052549
8: -0.0149335, -0.0033716, -0.0145679, -0.0039951, -0.0109383, 0.0111962
9: -0.0020513, 0.0044836, -0.0016904, 0.0042981, -0.0063495, 0.0061740

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_B1_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061302, upper bound: 0.0061604
time: 1.75 seconds

## Relational analysis of NS_A1_B2_B1_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061313, upper bound: 0.0061452
time: 1.76 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0029048, 0.0007335, -0.0026672, 0.0007227, -0.0036275, 0.0034007
1: -0.0057692, -0.0024493, -0.0058092, -0.0024765, -0.0028957, 0.0028612
2: 0.0311123, 0.0349798, 0.0312500, 0.0347585, -0.0036462, 0.0037298
3: -0.0030101, 0.0008359, -0.0029785, 0.0008822, -0.0038923, 0.0038143
4: -0.0047612, 0.0000075, -0.0048019, -0.0001100, -0.0046513, 0.0048094
5: 0.0099878, 0.0134041, 0.0101853, 0.0133671, -0.0033794, 0.0032188
6: -0.0048254, 0.0014489, -0.0046955, 0.0015078, -0.0063332, 0.0061445
7: 0.9735919, 0.9790731, 0.9739073, 0.9791142, -0.0055223, 0.0051658
8: -0.0147691, -0.0036519, -0.0144461, -0.0042028, -0.0105663, 0.0107942
9: -0.0018891, 0.0044002, -0.0015702, 0.0042364, -0.0061255, 0.0059704

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_B1_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061278, upper bound: 0.0061602
time: 2.06 seconds

## Relational analysis of NS_A1_B2_B1_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061290, upper bound: 0.0061434
time: 1.92 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0035464, 0.0009105, -0.0029522, 0.0007356, -0.0042820, 0.0038627
1: -0.0057422, -0.0022266, -0.0057131, -0.0024438, -0.0031570, 0.0034865
2: 0.0307403, 0.0362333, 0.0310848, 0.0350239, -0.0042836, 0.0051485
3: -0.0030953, 0.0011327, -0.0030163, 0.0007709, -0.0038662, 0.0041490
4: -0.0047338, 0.0005788, -0.0047042, 0.0000309, -0.0047647, 0.0052830
5: 0.0094543, 0.0139049, 0.0099483, 0.0134115, -0.0039571, 0.0039565
6: -0.0063828, 0.0014093, -0.0048514, 0.0013665, -0.0077493, 0.0062607
7: 0.9722292, 0.9790453, 0.9735289, 0.9790155, -0.0067862, 0.0055164
8: -0.0158742, -0.0021638, -0.0148336, -0.0035419, -0.0123323, 0.0126698
9: -0.0027505, 0.0051553, -0.0019528, 0.0044329, -0.0071834, 0.0071081

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_B1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061754, upper bound: 0.0061678
time: 1.73 seconds

## Relational analysis of NS_A1_B2_B1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061754, upper bound: 0.0061764
time: 2.39 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0034496, 0.0008754, -0.0028204, 0.0007297, -0.0041793, 0.0036958
1: -0.0057420, -0.0022687, -0.0057420, -0.0024590, -0.0031151, 0.0034733
2: 0.0307964, 0.0360069, 0.0311612, 0.0349012, -0.0041048, 0.0048457
3: -0.0030824, 0.0010277, -0.0029988, 0.0008044, -0.0038869, 0.0040265
4: -0.0047336, 0.0004781, -0.0047336, -0.0000342, -0.0046994, 0.0052118
5: 0.0095348, 0.0138065, 0.0100579, 0.0133909, -0.0038562, 0.0037486
6: -0.0060792, 0.0014090, -0.0047793, 0.0014090, -0.0074883, 0.0061883
7: 0.9724639, 0.9790452, 0.9737040, 0.9790452, -0.0065813, 0.0053412
8: -0.0156943, -0.0023883, -0.0146544, -0.0038476, -0.0118467, 0.0122661
9: -0.0026205, 0.0050237, -0.0017758, 0.0043420, -0.0069625, 0.0067995

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_B1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061754, upper bound: 0.0061675
time: 2.19 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061753, upper bound: 0.0061754
time: 1.60 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0028934, 0.0007330, -0.0034691, 0.0008825, -0.0037758, 0.0042021
1: -0.0058221, -0.0024506, -0.0059218, -0.0022602, -0.0035619, 0.0031368
2: 0.0311189, 0.0349691, 0.0307851, 0.0360525, -0.0049336, 0.0041840
3: -0.0030085, 0.0008972, -0.0030850, 0.0010488, -0.0040574, 0.0039822
4: -0.0048151, 0.0000018, -0.0049164, 0.0004984, -0.0053135, 0.0049183
5: 0.0099973, 0.0134023, 0.0095186, 0.0138263, -0.0038291, 0.0038837
6: -0.0048192, 0.0015268, -0.0061404, 0.0016733, -0.0064925, 0.0076671
7: 0.9736070, 0.9791276, 0.9724166, 0.9792301, -0.0056231, 0.0067110
8: -0.0147536, -0.0036784, -0.0157305, -0.0023431, -0.0124105, 0.0120521
9: -0.0018738, 0.0043923, -0.0026467, 0.0050502, -0.0069239, 0.0070390

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061824, upper bound: 0.0061434
time: 1.85 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061823, upper bound: 0.0061428
time: 1.77 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0028650, 0.0007317, -0.0040838, 0.0011055, -0.0039705, 0.0048155
1: -0.0058110, -0.0024538, -0.0059154, -0.0019930, -0.0038180, 0.0033433
2: 0.0311353, 0.0349427, 0.0304287, 0.0374902, -0.0063548, 0.0045139
3: -0.0030048, 0.0008843, -0.0031667, 0.0017155, -0.0047203, 0.0040510
4: -0.0048038, -0.0000122, -0.0049100, 0.0011374, -0.0059412, 0.0048978
5: 0.0100209, 0.0133979, 0.0090075, 0.0144509, -0.0044300, 0.0043904
6: -0.0048037, 0.0015104, -0.0080679, 0.0016640, -0.0064677, 0.0095783
7: 0.9736447, 0.9791162, 0.9709268, 0.9792237, -0.0055790, 0.0081894
8: -0.0147150, -0.0037442, -0.0168733, -0.0009175, -0.0137976, 0.0131291
9: -0.0018357, 0.0043728, -0.0034719, 0.0058865, -0.0077221, 0.0078447

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061888, upper bound: 0.0061303
time: 2.06 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061888, upper bound: 0.0061286
time: 1.90 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0029899, 0.0007373, -0.0036989, 0.0009658, -0.0039558, 0.0044362
1: -0.0057410, -0.0024395, -0.0059744, -0.0021603, -0.0035807, 0.0032985
2: 0.0310629, 0.0350590, 0.0306519, 0.0365898, -0.0055269, 0.0044071
3: -0.0030214, 0.0008033, -0.0031156, 0.0012980, -0.0043193, 0.0039188
4: -0.0047326, 0.0000496, -0.0049700, 0.0007372, -0.0054698, 0.0050196
5: 0.0099170, 0.0134173, 0.0093276, 0.0140598, -0.0041428, 0.0040898
6: -0.0048720, 0.0014076, -0.0068608, 0.0017507, -0.0066227, 0.0082684
7: 0.9734789, 0.9790441, 0.9718598, 0.9792843, -0.0058054, 0.0071844
8: -0.0148849, -0.0034544, -0.0161576, -0.0018103, -0.0130746, 0.0127032
9: -0.0020034, 0.0044589, -0.0029551, 0.0053627, -0.0073661, 0.0074140

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061975, upper bound: 0.0061674
time: 1.56 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062093, upper bound: 0.0061674
time: 2.00 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0028592, 0.0007314, -0.0035776, 0.0009218, -0.0037810, 0.0043090
1: -0.0057702, -0.0024545, -0.0059742, -0.0022130, -0.0035572, 0.0032555
2: 0.0311387, 0.0349372, 0.0307223, 0.0363061, -0.0051674, 0.0042150
3: -0.0030040, 0.0008371, -0.0030994, 0.0011664, -0.0041704, 0.0039365
4: -0.0047623, -0.0000151, -0.0049698, 0.0006111, -0.0053734, 0.0049547
5: 0.0100257, 0.0133970, 0.0094284, 0.0139365, -0.0039108, 0.0039686
6: -0.0048005, 0.0014505, -0.0064805, 0.0017504, -0.0065509, 0.0079310
7: 0.9736524, 0.9790742, 0.9721538, 0.9792840, -0.0056316, 0.0069205
8: -0.0147071, -0.0037577, -0.0159322, -0.0020916, -0.0126155, 0.0121745
9: -0.0018279, 0.0043687, -0.0027923, 0.0051977, -0.0070256, 0.0071610

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061959, upper bound: 0.0061674
time: 1.99 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062060, upper bound: 0.0061674
time: 1.82 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0032932, 0.0008186, -0.0028817, 0.0007324, -0.0040256, 0.0037003
1: -0.0059164, -0.0023366, -0.0058189, -0.0024519, -0.0030794, 0.0034823
2: 0.0308871, 0.0356411, 0.0311257, 0.0349582, -0.0040711, 0.0045154
3: -0.0030617, 0.0010064, -0.0030070, 0.0008935, -0.0039552, 0.0040134
4: -0.0049110, 0.0003156, -0.0048119, -0.0000039, -0.0049071, 0.0051274
5: 0.0096648, 0.0136476, 0.0100070, 0.0134005, -0.0037357, 0.0036407
6: -0.0055888, 0.0016654, -0.0048128, 0.0015221, -0.0071110, 0.0064783
7: 0.9728429, 0.9792246, 0.9736226, 0.9791244, -0.0062815, 0.0056020
8: -0.0154035, -0.0027511, -0.0147377, -0.0037054, -0.0116981, 0.0119867
9: -0.0024105, 0.0048109, -0.0018581, 0.0043843, -0.0067948, 0.0066690

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061438, upper bound: 0.0061836
time: 2.08 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061428, upper bound: 0.0061820
time: 1.95 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0039327, 0.0010507, -0.0028536, 0.0007312, -0.0046639, 0.0039043
1: -0.0059090, -0.0020587, -0.0058073, -0.0024551, -0.0032837, 0.0037486
2: 0.0305163, 0.0371367, 0.0311419, 0.0349321, -0.0044157, 0.0059948
3: -0.0031466, 0.0015516, -0.0030033, 0.0008801, -0.0040267, 0.0045549
4: -0.0049034, 0.0009803, -0.0048000, -0.0000178, -0.0048856, 0.0057804
5: 0.0091331, 0.0142974, 0.0100303, 0.0133961, -0.0042630, 0.0042671
6: -0.0075941, 0.0016545, -0.0047975, 0.0015050, -0.0090991, 0.0064520
7: 0.9712931, 0.9792171, 0.9736597, 0.9791124, -0.0078193, 0.0055574
8: -0.0165924, -0.0012679, -0.0146996, -0.0037705, -0.0128219, 0.0134317
9: -0.0032691, 0.0056809, -0.0018205, 0.0043649, -0.0076340, 0.0075013

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A2_B1_A1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061303, upper bound: 0.0061878
time: 1.94 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061286, upper bound: 0.0061878
time: 1.82 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0035330, 0.0009057, -0.0029800, 0.0007369, -0.0042699, 0.0038856
1: -0.0059689, -0.0022324, -0.0057381, -0.0024407, -0.0032453, 0.0035057
2: 0.0307481, 0.0362019, 0.0310687, 0.0350498, -0.0043017, 0.0051333
3: -0.0030935, 0.0011181, -0.0030200, 0.0007999, -0.0038934, 0.0041382
4: -0.0049644, 0.0005648, -0.0047297, 0.0000447, -0.0050090, 0.0052945
5: 0.0094655, 0.0138913, 0.0099253, 0.0134158, -0.0039503, 0.0039660
6: -0.0063408, 0.0017426, -0.0048666, 0.0014034, -0.0077441, 0.0066091
7: 0.9722618, 0.9792786, 0.9734921, 0.9790412, -0.0067794, 0.0057865
8: -0.0158493, -0.0021949, -0.0148713, -0.0034775, -0.0123718, 0.0126764
9: -0.0027325, 0.0051371, -0.0019900, 0.0044521, -0.0071845, 0.0071272

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061674, upper bound: 0.0061968
time: 2.25 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061674, upper bound: 0.0062079
time: 2.02 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0034139, 0.0008624, -0.0028483, 0.0007309, -0.0041448, 0.0037108
1: -0.0059687, -0.0022842, -0.0057680, -0.0024558, -0.0032040, 0.0034838
2: 0.0308171, 0.0359233, 0.0311450, 0.0349272, -0.0041100, 0.0047783
3: -0.0030777, 0.0010670, -0.0030026, 0.0008345, -0.0039122, 0.0040696
4: -0.0049642, 0.0004410, -0.0047601, -0.0000204, -0.0049438, 0.0052011
5: 0.0095645, 0.0137702, 0.0100347, 0.0133953, -0.0038308, 0.0037356
6: -0.0059672, 0.0017423, -0.0047946, 0.0014473, -0.0074145, 0.0065369
7: 0.9725504, 0.9792784, 0.9736668, 0.9790720, -0.0065216, 0.0056116
8: -0.0156279, -0.0024712, -0.0146924, -0.0037827, -0.0118451, 0.0122212
9: -0.0025725, 0.0049750, -0.0018134, 0.0043613, -0.0069338, 0.0067884

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061674, upper bound: 0.0061954
time: 1.97 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061673, upper bound: 0.0062049
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0034691, 0.0008825, -0.0028934, 0.0007330, -0.0042021, 0.0037758
1: -0.0059218, -0.0022602, -0.0058221, -0.0024506, -0.0031368, 0.0035619
2: 0.0307851, 0.0360525, 0.0311189, 0.0349691, -0.0041840, 0.0049336
3: -0.0030850, 0.0010488, -0.0030085, 0.0008972, -0.0039822, 0.0040574
4: -0.0049164, 0.0004984, -0.0048151, 0.0000018, -0.0049183, 0.0053135
5: 0.0095186, 0.0138263, 0.0099973, 0.0134023, -0.0038837, 0.0038291
6: -0.0061404, 0.0016733, -0.0048192, 0.0015268, -0.0076671, 0.0064925
7: 0.9724166, 0.9792301, 0.9736070, 0.9791276, -0.0067110, 0.0056231
8: -0.0157305, -0.0023431, -0.0147536, -0.0036784, -0.0120521, 0.0124105
9: -0.0026467, 0.0050502, -0.0018738, 0.0043923, -0.0070390, 0.0069239

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061434, upper bound: 0.0061824
time: 1.70 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061428, upper bound: 0.0061823
time: 1.59 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0040838, 0.0011055, -0.0028650, 0.0007317, -0.0048155, 0.0039705
1: -0.0059154, -0.0019930, -0.0058110, -0.0024538, -0.0033433, 0.0038180
2: 0.0304287, 0.0374902, 0.0311353, 0.0349427, -0.0045139, 0.0063548
3: -0.0031667, 0.0017155, -0.0030048, 0.0008843, -0.0040510, 0.0047203
4: -0.0049100, 0.0011374, -0.0048038, -0.0000122, -0.0048978, 0.0059412
5: 0.0090075, 0.0144509, 0.0100209, 0.0133979, -0.0043904, 0.0044300
6: -0.0080679, 0.0016640, -0.0048037, 0.0015104, -0.0095783, 0.0064677
7: 0.9709268, 0.9792237, 0.9736447, 0.9791162, -0.0081894, 0.0055790
8: -0.0168733, -0.0009175, -0.0147150, -0.0037442, -0.0131291, 0.0137976
9: -0.0034719, 0.0058865, -0.0018357, 0.0043728, -0.0078447, 0.0077221

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A2_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061303, upper bound: 0.0061888
time: 2.07 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061286, upper bound: 0.0061888
time: 2.15 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0036989, 0.0009658, -0.0029899, 0.0007373, -0.0044362, 0.0039558
1: -0.0059744, -0.0021603, -0.0057410, -0.0024395, -0.0032985, 0.0035807
2: 0.0306519, 0.0365898, 0.0310629, 0.0350590, -0.0044071, 0.0055269
3: -0.0031156, 0.0012980, -0.0030214, 0.0008033, -0.0039188, 0.0043193
4: -0.0049700, 0.0007372, -0.0047326, 0.0000496, -0.0050196, 0.0054698
5: 0.0093276, 0.0140598, 0.0099170, 0.0134173, -0.0040898, 0.0041428
6: -0.0068608, 0.0017507, -0.0048720, 0.0014076, -0.0082684, 0.0066227
7: 0.9718598, 0.9792843, 0.9734789, 0.9790441, -0.0071844, 0.0058054
8: -0.0161576, -0.0018103, -0.0148849, -0.0034544, -0.0127032, 0.0130746
9: -0.0029551, 0.0053627, -0.0020034, 0.0044589, -0.0074140, 0.0073661

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061674, upper bound: 0.0061975
time: 2.12 seconds

## Relational analysis of NS_A2_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061674, upper bound: 0.0062093
time: 1.56 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0035776, 0.0009218, -0.0028592, 0.0007314, -0.0043090, 0.0037810
1: -0.0059742, -0.0022130, -0.0057702, -0.0024545, -0.0032555, 0.0035572
2: 0.0307223, 0.0363061, 0.0311387, 0.0349372, -0.0042150, 0.0051674
3: -0.0030994, 0.0011664, -0.0030040, 0.0008371, -0.0039365, 0.0041704
4: -0.0049698, 0.0006111, -0.0047623, -0.0000151, -0.0049547, 0.0053734
5: 0.0094284, 0.0139365, 0.0100257, 0.0133970, -0.0039686, 0.0039108
6: -0.0064805, 0.0017504, -0.0048005, 0.0014505, -0.0079310, 0.0065509
7: 0.9721538, 0.9792840, 0.9736524, 0.9790742, -0.0069205, 0.0056316
8: -0.0159322, -0.0020916, -0.0147071, -0.0037577, -0.0121745, 0.0126155
9: -0.0027923, 0.0051977, -0.0018279, 0.0043687, -0.0071610, 0.0070256

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061674, upper bound: 0.0061959
time: 1.45 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061674, upper bound: 0.0062061
time: 1.86 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0027960, 0.0007286, -0.0032125, 0.0007893, -0.0035853, 0.0039410
1: -0.0060627, -0.0024618, -0.0060004, -0.0023717, -0.0036910, 0.0030247
2: 0.0311753, 0.0348784, 0.0309339, 0.0354522, -0.0042769, 0.0039445
3: -0.0029956, 0.0011759, -0.0030509, 0.0011038, -0.0040994, 0.0042269
4: -0.0050598, -0.0000463, -0.0049965, 0.0002316, -0.0052914, 0.0049502
5: 0.0100782, 0.0133872, 0.0097319, 0.0135656, -0.0034874, 0.0036552
6: -0.0047660, 0.0018805, -0.0053356, 0.0017890, -0.0065550, 0.0072162
7: 0.9737363, 0.9793751, 0.9730386, 0.9793111, -0.0055748, 0.0063366
8: -0.0146212, -0.0039042, -0.0152534, -0.0029383, -0.0116829, 0.0113492
9: -0.0017431, 0.0043252, -0.0023022, 0.0047010, -0.0064441, 0.0066273

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061594, upper bound: 0.0061747
time: 1.77 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061536, upper bound: 0.0061790
time: 1.38 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0026901, 0.0007238, -0.0030460, 0.0007399, -0.0034299, 0.0037698
1: -0.0060625, -0.0024739, -0.0060201, -0.0024331, -0.0029670, 0.0029983
2: 0.0312368, 0.0347797, 0.0310304, 0.0351113, -0.0038745, 0.0037494
3: -0.0029815, 0.0011757, -0.0030288, 0.0011265, -0.0041080, 0.0042045
4: -0.0050596, -0.0000987, -0.0050164, 0.0000773, -0.0051370, 0.0049178
5: 0.0101663, 0.0133707, 0.0098703, 0.0134261, -0.0032598, 0.0035003
6: -0.0047080, 0.0018803, -0.0049027, 0.0018179, -0.0065259, 0.0067829
7: 0.9738769, 0.9793749, 0.9734043, 0.9793312, -0.0054543, 0.0059705
8: -0.0144772, -0.0041498, -0.0149611, -0.0033244, -0.0111528, 0.0108113
9: -0.0016009, 0.0042521, -0.0020787, 0.0044976, -0.0060985, 0.0063308

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061595, upper bound: 0.0061717
time: 2.40 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061529, upper bound: 0.0061745
time: 1.96 seconds

## BFS NS instance: NS_A2_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0029792, 0.0007368, -0.0037300, 0.0009771, -0.0039564, 0.0044668
1: -0.0059678, -0.0024407, -0.0060025, -0.0021468, -0.0038210, 0.0032638
2: 0.0310691, 0.0350491, 0.0306338, 0.0366627, -0.0055935, 0.0044152
3: -0.0030199, 0.0010659, -0.0031197, 0.0013318, -0.0043517, 0.0041856
4: -0.0049632, 0.0000443, -0.0049986, 0.0007696, -0.0057329, 0.0050429
5: 0.0099259, 0.0134157, 0.0093017, 0.0140914, -0.0041656, 0.0041140
6: -0.0048661, 0.0017410, -0.0069585, 0.0017920, -0.0066582, 0.0086994
7: 0.9734930, 0.9792774, 0.9717843, 0.9793132, -0.0058202, 0.0074931
8: -0.0148703, -0.0034792, -0.0162156, -0.0017380, -0.0131323, 0.0127363
9: -0.0019890, 0.0044516, -0.0029970, 0.0054051, -0.0073941, 0.0074485

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_A2_A1_B1

### Relational analysis result of NS_A2_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061645, upper bound: 0.0062064
time: 2.23 seconds

## Relational analysis of NS_A2_B2_A1_A2_A1_B2

### Relational analysis result of NS_A2_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061714, upper bound: 0.0062065
time: 2.14 seconds

## BFS NS instance: NS_A2_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0027966, 0.0007286, -0.0036088, 0.0009332, -0.0037298, 0.0043374
1: -0.0059876, -0.0024617, -0.0060023, -0.0021995, -0.0037881, 0.0032124
2: 0.0311750, 0.0348790, 0.0307041, 0.0363792, -0.0052042, 0.0041749
3: -0.0029957, 0.0010889, -0.0031036, 0.0012003, -0.0041960, 0.0041925
4: -0.0049834, -0.0000460, -0.0049984, 0.0006436, -0.0056270, 0.0049524
5: 0.0100777, 0.0133873, 0.0094024, 0.0139683, -0.0038906, 0.0039848
6: -0.0047663, 0.0017701, -0.0065784, 0.0017918, -0.0065581, 0.0083485
7: 0.9737354, 0.9792978, 0.9720780, 0.9793130, -0.0055776, 0.0072198
8: -0.0146221, -0.0039027, -0.0159902, -0.0020191, -0.0126030, 0.0120876
9: -0.0017439, 0.0043256, -0.0028342, 0.0052402, -0.0069842, 0.0071599

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_A2_A2_A1

### Relational analysis result of NS_A2_B2_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061719, upper bound: 0.0061975
time: 2.00 seconds

## Relational analysis of NS_A2_B2_A1_A2_A2_A2

### Relational analysis result of NS_A2_B2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061720, upper bound: 0.0062066
time: 2.45 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0029940, 0.0007375, -0.0032235, 0.0007934, -0.0037873, 0.0039610
1: -0.0060704, -0.0024391, -0.0060016, -0.0023669, -0.0037035, 0.0030632
2: 0.0310606, 0.0350628, 0.0309275, 0.0354781, -0.0044175, 0.0041353
3: -0.0030219, 0.0011848, -0.0030524, 0.0011051, -0.0041270, 0.0042372
4: -0.0050676, 0.0000516, -0.0049977, 0.0002431, -0.0053108, 0.0050492
5: 0.0099136, 0.0134180, 0.0097228, 0.0135768, -0.0036632, 0.0036952
6: -0.0048742, 0.0018918, -0.0053703, 0.0017907, -0.0066649, 0.0072622
7: 0.9734735, 0.9793831, 0.9730118, 0.9793123, -0.0058388, 0.0063713
8: -0.0148903, -0.0034451, -0.0152740, -0.0029126, -0.0119777, 0.0118289
9: -0.0020088, 0.0044617, -0.0023170, 0.0047161, -0.0067249, 0.0067787

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061594, upper bound: 0.0061740
time: 1.67 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061536, upper bound: 0.0061790
time: 1.97 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0028843, 0.0007326, -0.0030570, 0.0007404, -0.0036247, 0.0037896
1: -0.0060702, -0.0024516, -0.0060210, -0.0024318, -0.0030216, 0.0030359
2: 0.0311241, 0.0349607, 0.0310240, 0.0351215, -0.0039974, 0.0039367
3: -0.0030073, 0.0011846, -0.0030303, 0.0011276, -0.0041350, 0.0042149
4: -0.0050674, -0.0000026, -0.0050174, 0.0000828, -0.0051502, 0.0050148
5: 0.0100047, 0.0134009, 0.0098612, 0.0134278, -0.0034230, 0.0035397
6: -0.0048143, 0.0018916, -0.0049087, 0.0018193, -0.0066335, 0.0068003
7: 0.9736190, 0.9793829, 0.9733897, 0.9793322, -0.0057132, 0.0059932
8: -0.0147413, -0.0036993, -0.0149761, -0.0032988, -0.0114425, 0.0112768
9: -0.0018617, 0.0043861, -0.0020935, 0.0045052, -0.0063669, 0.0064796

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061594, upper bound: 0.0061708
time: 2.05 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061529, upper bound: 0.0061742
time: 1.89 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0031436, 0.0007644, -0.0037409, 0.0009811, -0.0041247, 0.0045053
1: -0.0059733, -0.0024017, -0.0060036, -0.0021420, -0.0038312, 0.0036020
2: 0.0309738, 0.0352912, 0.0306275, 0.0366881, -0.0057143, 0.0046636
3: -0.0030418, 0.0010723, -0.0031211, 0.0013436, -0.0043854, 0.0041935
4: -0.0049688, 0.0001600, -0.0049997, 0.0007810, -0.0057498, 0.0051598
5: 0.0097892, 0.0134956, 0.0092926, 0.0141025, -0.0043133, 0.0042030
6: -0.0051197, 0.0017491, -0.0069927, 0.0017937, -0.0069134, 0.0087417
7: 0.9732056, 0.9792832, 0.9717579, 0.9793144, -0.0061089, 0.0075253
8: -0.0151254, -0.0030980, -0.0162358, -0.0017127, -0.0134127, 0.0131378
9: -0.0022097, 0.0046073, -0.0030116, 0.0054199, -0.0076296, 0.0076189

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061645, upper bound: 0.0062078
time: 1.62 seconds

## Relational analysis of NS_A2_B2_A2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061713, upper bound: 0.0062078
time: 1.90 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0029609, 0.0007360, -0.0036197, 0.0009371, -0.0038980, 0.0043557
1: -0.0059925, -0.0024428, -0.0060035, -0.0021947, -0.0037978, 0.0032517
2: 0.0310797, 0.0350320, 0.0306978, 0.0364047, -0.0053249, 0.0043342
3: -0.0030175, 0.0010947, -0.0031050, 0.0012121, -0.0042296, 0.0041997
4: -0.0049884, 0.0000353, -0.0049996, 0.0006550, -0.0056434, 0.0050348
5: 0.0099411, 0.0134128, 0.0093934, 0.0139793, -0.0040383, 0.0040194
6: -0.0048561, 0.0017774, -0.0066126, 0.0017934, -0.0066496, 0.0083899
7: 0.9735173, 0.9793029, 0.9720517, 0.9793143, -0.0057970, 0.0072513
8: -0.0148454, -0.0035217, -0.0160105, -0.0019939, -0.0128516, 0.0124888
9: -0.0019645, 0.0044389, -0.0028488, 0.0052550, -0.0072195, 0.0072878

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061658, upper bound: 0.0062079
time: 2.26 seconds

## Relational analysis of NS_A2_B2_A2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061719, upper bound: 0.0062079
time: 2.31 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.27 seconds
NS_A1_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061303, upper bound: 0.0061604
NS_A1_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061313, upper bound: 0.0061452
NS_A1_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061279, upper bound: 0.0061602
NS_A1_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061279, upper bound: 0.0061434
NS_A1_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061735, upper bound: 0.0061678
NS_A1_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061735, upper bound: 0.0061764
NS_A1_B1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061666, upper bound: 0.0061754
NS_A1_B1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061735, upper bound: 0.0061754
NS_A1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061836, upper bound: 0.0061438
NS_A1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061820, upper bound: 0.0061428
NS_A1_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061878, upper bound: 0.0061303
NS_A1_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061878, upper bound: 0.0061286
NS_A1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061968, upper bound: 0.0061674
NS_A1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0062079, upper bound: 0.0061674
NS_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061954, upper bound: 0.0061674
NS_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0062049, upper bound: 0.0061674
NS_A1_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061302, upper bound: 0.0061604
NS_A1_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061313, upper bound: 0.0061452
NS_A1_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061278, upper bound: 0.0061602
NS_A1_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061290, upper bound: 0.0061434
NS_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061754, upper bound: 0.0061678
NS_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061754, upper bound: 0.0061764
NS_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061754, upper bound: 0.0061675
NS_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061753, upper bound: 0.0061754
NS_A1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061824, upper bound: 0.0061434
NS_A1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061823, upper bound: 0.0061428
NS_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061888, upper bound: 0.0061303
NS_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061888, upper bound: 0.0061286
NS_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061975, upper bound: 0.0061674
NS_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0062093, upper bound: 0.0061674
NS_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061959, upper bound: 0.0061674
NS_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0062060, upper bound: 0.0061674
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061438, upper bound: 0.0061836
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061428, upper bound: 0.0061820
NS_A2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061303, upper bound: 0.0061878
NS_A2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061286, upper bound: 0.0061878
NS_A2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061674, upper bound: 0.0061968
NS_A2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061674, upper bound: 0.0062079
NS_A2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061674, upper bound: 0.0061954
NS_A2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061673, upper bound: 0.0062049
NS_A2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061434, upper bound: 0.0061824
NS_A2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061428, upper bound: 0.0061823
NS_A2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061303, upper bound: 0.0061888
NS_A2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061286, upper bound: 0.0061888
NS_A2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061674, upper bound: 0.0061975
NS_A2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061674, upper bound: 0.0062093
NS_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061674, upper bound: 0.0061959
NS_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061674, upper bound: 0.0062061
NS_A2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061594, upper bound: 0.0061747
NS_A2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061536, upper bound: 0.0061790
NS_A2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061595, upper bound: 0.0061717
NS_A2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061529, upper bound: 0.0061745
NS_A2_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061645, upper bound: 0.0062064
NS_A2_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061714, upper bound: 0.0062065
NS_A2_B2_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061719, upper bound: 0.0061975
NS_A2_B2_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061720, upper bound: 0.0062066
NS_A2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061594, upper bound: 0.0061740
NS_A2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061536, upper bound: 0.0061790
NS_A2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061594, upper bound: 0.0061708
NS_A2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061529, upper bound: 0.0061742
NS_A2_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061645, upper bound: 0.0062078
NS_A2_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061713, upper bound: 0.0062078
NS_A2_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061658, upper bound: 0.0062079
NS_A2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 7, lower bound: -0.0061719, upper bound: 0.0062079

## BFS NS instance: NS_A1_B1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0029947, 0.0007375, -0.0024975, 0.0007151, -0.0037097, 0.0032350
1: -0.0057221, -0.0024390, -0.0057492, -0.0024960, -0.0028512, 0.0027791
2: 0.0310602, 0.0350635, 0.0313484, 0.0346003, -0.0035402, 0.0037151
3: -0.0030220, 0.0007813, -0.0029559, 0.0008127, -0.0038347, 0.0037373
4: -0.0047133, 0.0000519, -0.0047409, -0.0001939, -0.0045195, 0.0047928
5: 0.0099130, 0.0134181, 0.0103264, 0.0133407, -0.0034277, 0.0030917
6: -0.0048746, 0.0013797, -0.0046027, 0.0014196, -0.0062942, 0.0059825
7: 0.9734726, 0.9790246, 0.9741326, 0.9790525, -0.0055799, 0.0048921
8: -0.0148914, -0.0034434, -0.0142153, -0.0045965, -0.0102949, 0.0107719
9: -0.0020098, 0.0044622, -0.0013423, 0.0041193, -0.0061291, 0.0058046

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061068, upper bound: 0.0061444
time: 1.86 seconds

## Relational analysis of NS_A1_B1_B1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061059, upper bound: 0.0061444
time: 1.80 seconds

## BFS NS instance: NS_A1_B1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0029666, 0.0007363, -0.0030066, 0.0007381, -0.0037047, 0.0037429
1: -0.0057108, -0.0024422, -0.0057414, -0.0024376, -0.0028903, 0.0029277
2: 0.0310764, 0.0350373, 0.0310532, 0.0350746, -0.0039982, 0.0039841
3: -0.0030183, 0.0007683, -0.0030236, 0.0008037, -0.0038219, 0.0037919
4: -0.0047019, 0.0000381, -0.0047330, 0.0000578, -0.0047597, 0.0047710
5: 0.0099364, 0.0134137, 0.0099031, 0.0134199, -0.0034835, 0.0035106
6: -0.0048592, 0.0013632, -0.0048811, 0.0014081, -0.0062673, 0.0062443
7: 0.9735098, 0.9790132, 0.9734567, 0.9790445, -0.0055346, 0.0055565
8: -0.0148531, -0.0035085, -0.0149076, -0.0034157, -0.0114374, 0.0113991
9: -0.0019721, 0.0044428, -0.0020258, 0.0044704, -0.0064425, 0.0064686

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061093, upper bound: 0.0061245
time: 2.05 seconds

## Relational analysis of NS_A1_B1_B1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061059, upper bound: 0.0061192
time: 2.15 seconds

## BFS NS instance: NS_A1_B1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0028734, 0.0007321, -0.0024091, 0.0007111, -0.0035845, 0.0031411
1: -0.0057520, -0.0024529, -0.0057490, -0.0025061, -0.0028392, 0.0027493
2: 0.0311304, 0.0349505, 0.0313996, 0.0345180, -0.0033876, 0.0035509
3: -0.0030059, 0.0008160, -0.0029442, 0.0008125, -0.0038184, 0.0037602
4: -0.0047438, -0.0000080, -0.0047407, -0.0002376, -0.0045062, 0.0047327
5: 0.0100138, 0.0133992, 0.0103999, 0.0133269, -0.0033131, 0.0029993
6: -0.0048083, 0.0014238, -0.0045544, 0.0014193, -0.0062276, 0.0059782
7: 0.9736335, 0.9790555, 0.9742500, 0.9790523, -0.0054188, 0.0048055
8: -0.0147265, -0.0037246, -0.0140951, -0.0048015, -0.0099250, 0.0103705
9: -0.0018470, 0.0043786, -0.0012237, 0.0040583, -0.0059053, 0.0056023

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061054, upper bound: 0.0061441
time: 1.90 seconds

## Relational analysis of NS_A1_B1_B1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061042, upper bound: 0.0061441
time: 2.26 seconds

## BFS NS instance: NS_A1_B1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0028456, 0.0007308, -0.0029119, 0.0007338, -0.0035794, 0.0036427
1: -0.0057405, -0.0024561, -0.0057412, -0.0024485, -0.0028775, 0.0028904
2: 0.0311466, 0.0349246, 0.0311082, 0.0349863, -0.0038397, 0.0038164
3: -0.0030022, 0.0008027, -0.0030110, 0.0008034, -0.0038056, 0.0038137
4: -0.0047321, -0.0000218, -0.0047328, 0.0000110, -0.0047431, 0.0047110
5: 0.0100370, 0.0133949, 0.0099819, 0.0134052, -0.0033682, 0.0034130
6: -0.0047931, 0.0014069, -0.0048293, 0.0014078, -0.0062009, 0.0062362
7: 0.9736705, 0.9790437, 0.9735825, 0.9790444, -0.0053739, 0.0054612
8: -0.0146887, -0.0037892, -0.0147787, -0.0036355, -0.0110532, 0.0109896
9: -0.0018096, 0.0043594, -0.0018986, 0.0044051, -0.0062147, 0.0062580

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061066, upper bound: 0.0061213
time: 2.00 seconds

## Relational analysis of NS_A1_B1_B1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061046, upper bound: 0.0061167
time: 1.73 seconds

## BFS NS instance: NS_A1_B1_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0034668, 0.0008816, -0.0027536, 0.0007266, -0.0041934, 0.0036352
1: -0.0056884, -0.0022612, -0.0056886, -0.0024666, -0.0030709, 0.0034274
2: 0.0307864, 0.0360471, 0.0311999, 0.0348389, -0.0040525, 0.0048472
3: -0.0030847, 0.0010463, -0.0029900, 0.0007425, -0.0038273, 0.0040363
4: -0.0046791, 0.0004960, -0.0046793, -0.0000673, -0.0046118, 0.0051753
5: 0.0095205, 0.0138240, 0.0101135, 0.0133806, -0.0038601, 0.0037105
6: -0.0061332, 0.0013303, -0.0047428, 0.0013305, -0.0074637, 0.0060730
7: 0.9724222, 0.9789901, 0.9737927, 0.9789903, -0.0065681, 0.0051974
8: -0.0157263, -0.0023484, -0.0145635, -0.0040026, -0.0117237, 0.0122151
9: -0.0026436, 0.0050470, -0.0016861, 0.0042959, -0.0069395, 0.0067332

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B1_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061607, upper bound: 0.0061526
time: 2.16 seconds

## Relational analysis of NS_A1_B1_B1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B1_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061592, upper bound: 0.0061526
time: 1.93 seconds

## BFS NS instance: NS_A1_B1_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0039662, 0.0010629, -0.0027288, 0.0007255, -0.0046918, 0.0037917
1: -0.0056831, -0.0020441, -0.0056769, -0.0024695, -0.0032136, 0.0036328
2: 0.0304969, 0.0372152, 0.0312143, 0.0348159, -0.0043190, 0.0060009
3: -0.0031511, 0.0015880, -0.0029867, 0.0007290, -0.0038801, 0.0045746
4: -0.0046962, 0.0010152, -0.0046674, -0.0000795, -0.0046168, 0.0056826
5: 0.0091053, 0.0143314, 0.0101340, 0.0133767, -0.0042714, 0.0041974
6: -0.0076992, 0.0013224, -0.0047293, 0.0013133, -0.0090125, 0.0060516
7: 0.9712118, 0.9789846, 0.9738255, 0.9789781, -0.0077663, 0.0051591
8: -0.0166547, -0.0011902, -0.0145299, -0.0040599, -0.0125948, 0.0133397
9: -0.0033141, 0.0057265, -0.0016529, 0.0042789, -0.0075930, 0.0073794

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061580, upper bound: 0.0061594
time: 2.13 seconds

## Relational analysis of NS_A1_B1_B1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061561, upper bound: 0.0061591
time: 2.14 seconds

## BFS NS instance: NS_A1_B1_B1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0034187, 0.0008642, -0.0025817, 0.0007189, -0.0041376, 0.0034459
1: -0.0057241, -0.0022821, -0.0056838, -0.0024863, -0.0030592, 0.0034017
2: 0.0308143, 0.0359346, 0.0312996, 0.0346788, -0.0038645, 0.0046350
3: -0.0030783, 0.0009942, -0.0029671, 0.0007370, -0.0038153, 0.0039613
4: -0.0047154, 0.0004460, -0.0046744, -0.0001522, -0.0045631, 0.0051204
5: 0.0095605, 0.0137751, 0.0102564, 0.0133538, -0.0037933, 0.0035188
6: -0.0059823, 0.0013827, -0.0046488, 0.0013234, -0.0073058, 0.0060315
7: 0.9725388, 0.9790267, 0.9740208, 0.9789853, -0.0064465, 0.0050059
8: -0.0156368, -0.0024600, -0.0143298, -0.0044011, -0.0112357, 0.0118698
9: -0.0025790, 0.0049816, -0.0014554, 0.0041774, -0.0067564, 0.0064370

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B1_B2_B2_B1_A1

### Relational analysis result of NS_A1_B1_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061518, upper bound: 0.0061625
time: 1.56 seconds

## Relational analysis of NS_A1_B1_B1_B2_B2_B1_A2

### Relational analysis result of NS_A1_B1_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061518, upper bound: 0.0061614
time: 1.72 seconds

## BFS NS instance: NS_A1_B1_B1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0033904, 0.0008539, -0.0031073, 0.0007426, -0.0041330, 0.0039612
1: -0.0057129, -0.0022944, -0.0056774, -0.0024261, -0.0030970, 0.0033830
2: 0.0308307, 0.0358684, 0.0309949, 0.0351684, -0.0043377, 0.0048736
3: -0.0030746, 0.0009635, -0.0030370, 0.0007296, -0.0038042, 0.0040004
4: -0.0047040, 0.0004166, -0.0046679, 0.0001076, -0.0048116, 0.0050845
5: 0.0095840, 0.0137464, 0.0098194, 0.0134356, -0.0038516, 0.0039270
6: -0.0058936, 0.0013662, -0.0049362, 0.0013141, -0.0072078, 0.0063024
7: 0.9726074, 0.9790152, 0.9733230, 0.9789788, -0.0063714, 0.0056922
8: -0.0155842, -0.0025256, -0.0150445, -0.0031822, -0.0124020, 0.0125189
9: -0.0025410, 0.0049431, -0.0021610, 0.0045399, -0.0070809, 0.0071041

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061561, upper bound: 0.0061597
time: 2.07 seconds

## Relational analysis of NS_A1_B1_B1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061560, upper bound: 0.0061582
time: 2.10 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0025251, 0.0007163, -0.0031726, 0.0007749, -0.0033000, 0.0038889
1: -0.0058184, -0.0024928, -0.0059162, -0.0023891, -0.0034293, 0.0029983
2: 0.0313324, 0.0346261, 0.0309570, 0.0353590, -0.0040266, 0.0036691
3: -0.0029596, 0.0008929, -0.0030456, 0.0010062, -0.0039658, 0.0039385
4: -0.0048113, -0.0001802, -0.0049108, 0.0001902, -0.0050015, 0.0047306
5: 0.0103034, 0.0133450, 0.0097651, 0.0135251, -0.0032217, 0.0035799
6: -0.0046179, 0.0015213, -0.0052106, 0.0016652, -0.0062830, 0.0067319
7: 0.9740960, 0.9791238, 0.9731353, 0.9792244, -0.0051284, 0.0059885
8: -0.0142529, -0.0045324, -0.0151793, -0.0030308, -0.0112221, 0.0106469
9: -0.0013794, 0.0041384, -0.0022486, 0.0046468, -0.0060262, 0.0063870

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061689, upper bound: 0.0061282
time: 2.33 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061682, upper bound: 0.0061281
time: 2.37 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0024165, 0.0007114, -0.0030587, 0.0007404, -0.0031569, 0.0037701
1: -0.0058478, -0.0025053, -0.0059160, -0.0024316, -0.0028414, 0.0029601
2: 0.0313954, 0.0345248, 0.0310231, 0.0351231, -0.0037277, 0.0035018
3: -0.0029452, 0.0009270, -0.0030305, 0.0010060, -0.0039512, 0.0039575
4: -0.0048413, -0.0002339, -0.0049106, 0.0000836, -0.0049248, 0.0046767
5: 0.0103938, 0.0133281, 0.0098598, 0.0134280, -0.0030343, 0.0034683
6: -0.0045584, 0.0015646, -0.0049096, 0.0016649, -0.0062233, 0.0064742
7: 0.9742402, 0.9791541, 0.9733875, 0.9792243, -0.0049840, 0.0057665
8: -0.0141051, -0.0047844, -0.0149784, -0.0032950, -0.0108101, 0.0101940
9: -0.0012335, 0.0040634, -0.0020957, 0.0045064, -0.0057399, 0.0061591

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061649, upper bound: 0.0061284
time: 1.77 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061646, upper bound: 0.0061272
time: 2.14 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0027319, 0.0007257, -0.0035716, 0.0009197, -0.0036516, 0.0042973
1: -0.0058071, -0.0024691, -0.0059084, -0.0022156, -0.0035915, 0.0031389
2: 0.0312125, 0.0348188, 0.0307257, 0.0362921, -0.0050797, 0.0040931
3: -0.0029871, 0.0008798, -0.0030986, 0.0011600, -0.0041471, 0.0039785
4: -0.0047998, -0.0000780, -0.0049028, 0.0006050, -0.0054048, 0.0048249
5: 0.0101315, 0.0133772, 0.0094334, 0.0139305, -0.0037990, 0.0039438
6: -0.0047309, 0.0015048, -0.0064618, 0.0016536, -0.0063846, 0.0079665
7: 0.9738214, 0.9791120, 0.9721683, 0.9792163, -0.0053949, 0.0069438
8: -0.0145341, -0.0040528, -0.0159210, -0.0021054, -0.0124287, 0.0118683
9: -0.0016571, 0.0042810, -0.0027843, 0.0051896, -0.0068467, 0.0070653

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061708, upper bound: 0.0061096
time: 1.90 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061677, upper bound: 0.0061064
time: 1.61 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0026424, 0.0007216, -0.0034009, 0.0008577, -0.0035002, 0.0041225
1: -0.0058069, -0.0024794, -0.0059281, -0.0022898, -0.0035171, 0.0031041
2: 0.0312644, 0.0347354, 0.0308247, 0.0358928, -0.0046285, 0.0039107
3: -0.0029752, 0.0008796, -0.0030760, 0.0010200, -0.0039952, 0.0039556
4: -0.0047996, -0.0001222, -0.0049229, 0.0004275, -0.0052271, 0.0048007
5: 0.0102059, 0.0133633, 0.0095753, 0.0137570, -0.0035511, 0.0037879
6: -0.0046820, 0.0015045, -0.0059264, 0.0016826, -0.0063646, 0.0074309
7: 0.9739401, 0.9791120, 0.9725819, 0.9792366, -0.0052965, 0.0065300
8: -0.0144124, -0.0042603, -0.0156037, -0.0025014, -0.0119111, 0.0113434
9: -0.0015369, 0.0042193, -0.0025551, 0.0049573, -0.0064943, 0.0067744

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061707, upper bound: 0.0061069
time: 1.55 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061677, upper bound: 0.0061043
time: 1.93 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0029595, 0.0007360, -0.0034664, 0.0008815, -0.0038410, 0.0042024
1: -0.0057231, -0.0024430, -0.0059179, -0.0022613, -0.0034618, 0.0031914
2: 0.0310805, 0.0350308, 0.0307867, 0.0360462, -0.0049657, 0.0042441
3: -0.0030173, 0.0007825, -0.0030847, 0.0010459, -0.0040633, 0.0038672
4: -0.0047144, 0.0000346, -0.0049125, 0.0004956, -0.0052100, 0.0049471
5: 0.0099422, 0.0134126, 0.0095208, 0.0138236, -0.0038814, 0.0038918
6: -0.0048554, 0.0013812, -0.0061320, 0.0016676, -0.0065230, 0.0075133
7: 0.9735191, 0.9790258, 0.9724231, 0.9792261, -0.0057069, 0.0066026
8: -0.0148436, -0.0035249, -0.0157256, -0.0023493, -0.0124943, 0.0122007
9: -0.0019626, 0.0044380, -0.0026431, 0.0050466, -0.0070092, 0.0070811

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061847, upper bound: 0.0061540
time: 2.14 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061845, upper bound: 0.0061540
time: 2.21 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0029314, 0.0007347, -0.0040981, 0.0011107, -0.0040421, 0.0048327
1: -0.0057119, -0.0024462, -0.0059108, -0.0019868, -0.0037251, 0.0033914
2: 0.0310969, 0.0350045, 0.0304205, 0.0375235, -0.0064266, 0.0045841
3: -0.0030136, 0.0007695, -0.0031686, 0.0017309, -0.0047445, 0.0039381
4: -0.0047030, 0.0000206, -0.0049053, 0.0011522, -0.0058552, 0.0049259
5: 0.0099656, 0.0134082, 0.0089957, 0.0144654, -0.0044997, 0.0044126
6: -0.0048400, 0.0013648, -0.0081126, 0.0016572, -0.0064972, 0.0094774
7: 0.9735566, 0.9790142, 0.9708924, 0.9792189, -0.0056623, 0.0081218
8: -0.0148053, -0.0035902, -0.0168998, -0.0008844, -0.0139209, 0.0133096
9: -0.0019248, 0.0044186, -0.0034911, 0.0059058, -0.0078307, 0.0079096

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061950, upper bound: 0.0061530
time: 1.68 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061949, upper bound: 0.0061530
time: 2.12 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0028284, 0.0007300, -0.0033474, 0.0008383, -0.0036667, 0.0040774
1: -0.0057530, -0.0024580, -0.0059177, -0.0023131, -0.0034400, 0.0031499
2: 0.0311566, 0.0349086, 0.0308557, 0.0357678, -0.0046112, 0.0040529
3: -0.0029999, 0.0008172, -0.0030689, 0.0010080, -0.0040079, 0.0038861
4: -0.0047448, -0.0000303, -0.0049123, 0.0003719, -0.0051167, 0.0048820
5: 0.0100513, 0.0133922, 0.0096198, 0.0137027, -0.0036514, 0.0037724
6: -0.0047837, 0.0014253, -0.0057588, 0.0016674, -0.0064510, 0.0071841
7: 0.9736934, 0.9790566, 0.9727116, 0.9792260, -0.0055326, 0.0063450
8: -0.0146652, -0.0038292, -0.0155043, -0.0026253, -0.0120399, 0.0116751
9: -0.0017865, 0.0043475, -0.0024833, 0.0048846, -0.0066711, 0.0068308

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061840, upper bound: 0.0061540
time: 1.46 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061835, upper bound: 0.0061540
time: 1.72 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0027997, 0.0007287, -0.0039789, 0.0010675, -0.0038672, 0.0047076
1: -0.0057416, -0.0024613, -0.0059106, -0.0020386, -0.0037029, 0.0033428
2: 0.0311732, 0.0348819, 0.0304896, 0.0372446, -0.0060714, 0.0043923
3: -0.0029961, 0.0008039, -0.0031528, 0.0016016, -0.0045977, 0.0039567
4: -0.0047332, -0.0000444, -0.0049051, 0.0010283, -0.0057615, 0.0048607
5: 0.0100751, 0.0133877, 0.0090948, 0.0143442, -0.0042691, 0.0042930
6: -0.0047680, 0.0014084, -0.0077387, 0.0016569, -0.0064249, 0.0091471
7: 0.9737313, 0.9790448, 0.9711813, 0.9792188, -0.0054875, 0.0078635
8: -0.0146263, -0.0038955, -0.0166781, -0.0011609, -0.0134654, 0.0127827
9: -0.0017481, 0.0043278, -0.0033310, 0.0057436, -0.0074917, 0.0076588

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061934, upper bound: 0.0061530
time: 1.68 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061921, upper bound: 0.0061530
time: 2.08 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0030048, 0.0007380, -0.0026892, 0.0007237, -0.0037285, 0.0034272
1: -0.0057248, -0.0024378, -0.0057590, -0.0024740, -0.0028850, 0.0028384
2: 0.0310543, 0.0350729, 0.0312373, 0.0347789, -0.0037246, 0.0038356
3: -0.0030233, 0.0007845, -0.0029814, 0.0008241, -0.0038475, 0.0037659
4: -0.0047162, 0.0000569, -0.0047509, -0.0000991, -0.0046171, 0.0048079
5: 0.0099046, 0.0134196, 0.0101670, 0.0133705, -0.0034659, 0.0032526
6: -0.0048801, 0.0013838, -0.0047076, 0.0014341, -0.0063142, 0.0060914
7: 0.9734591, 0.9790276, 0.9738782, 0.9790627, -0.0056036, 0.0051494
8: -0.0149051, -0.0034199, -0.0144759, -0.0041519, -0.0107532, 0.0110560
9: -0.0020234, 0.0044692, -0.0015997, 0.0042515, -0.0062749, 0.0060689

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061068, upper bound: 0.0061444
time: 1.65 seconds

## Relational analysis of NS_A1_B2_B1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061059, upper bound: 0.0061443
time: 2.20 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0029764, 0.0007367, -0.0032043, 0.0007470, -0.0037234, 0.0039410
1: -0.0057139, -0.0024411, -0.0057539, -0.0024149, -0.0029249, 0.0029979
2: 0.0310707, 0.0350465, 0.0309387, 0.0352587, -0.0041879, 0.0041078
3: -0.0030196, 0.0007719, -0.0030498, 0.0008182, -0.0038378, 0.0038217
4: -0.0047050, 0.0000429, -0.0047457, 0.0001556, -0.0048606, 0.0047886
5: 0.0099282, 0.0134152, 0.0097388, 0.0134507, -0.0035225, 0.0036764
6: -0.0048646, 0.0013678, -0.0049892, 0.0014266, -0.0062912, 0.0063570
7: 0.9734967, 0.9790164, 0.9731944, 0.9790574, -0.0055606, 0.0058220
8: -0.0148666, -0.0034857, -0.0151763, -0.0029574, -0.0119092, 0.0116906
9: -0.0019853, 0.0044496, -0.0022911, 0.0046068, -0.0065921, 0.0067408

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061099, upper bound: 0.0061245
time: 1.89 seconds

## Relational analysis of NS_A1_B2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061064, upper bound: 0.0061192
time: 2.18 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0028842, 0.0007325, -0.0025998, 0.0007197, -0.0036038, 0.0033324
1: -0.0057542, -0.0024516, -0.0057588, -0.0024843, -0.0028734, 0.0028061
2: 0.0311242, 0.0349605, 0.0312891, 0.0346956, -0.0035714, 0.0036715
3: -0.0030073, 0.0008186, -0.0029695, 0.0008239, -0.0038312, 0.0037881
4: -0.0047460, -0.0000027, -0.0047507, -0.0001433, -0.0046028, 0.0047480
5: 0.0100049, 0.0134009, 0.0102413, 0.0133566, -0.0033517, 0.0031596
6: -0.0048142, 0.0014270, -0.0046587, 0.0014338, -0.0062480, 0.0060857
7: 0.9736192, 0.9790578, 0.9739968, 0.9790626, -0.0054433, 0.0050611
8: -0.0147411, -0.0036997, -0.0143544, -0.0043592, -0.0103819, 0.0106547
9: -0.0018614, 0.0043860, -0.0014797, 0.0041899, -0.0060513, 0.0058657

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061066, upper bound: 0.0061463
time: 1.71 seconds

## Relational analysis of NS_A1_B2_B1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061037, upper bound: 0.0061441
time: 2.03 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0028560, 0.0007313, -0.0031091, 0.0007427, -0.0035988, 0.0038404
1: -0.0057426, -0.0024549, -0.0057537, -0.0024258, -0.0029116, 0.0029587
2: 0.0311405, 0.0349344, 0.0309938, 0.0351701, -0.0040296, 0.0039405
3: -0.0030036, 0.0008051, -0.0030372, 0.0008180, -0.0038216, 0.0038423
4: -0.0047342, -0.0000166, -0.0047455, 0.0001085, -0.0048427, 0.0047289
5: 0.0100283, 0.0133965, 0.0098179, 0.0134359, -0.0034076, 0.0035786
6: -0.0047988, 0.0014099, -0.0049372, 0.0014263, -0.0062251, 0.0063470
7: 0.9736565, 0.9790457, 0.9733206, 0.9790573, -0.0054007, 0.0057251
8: -0.0147029, -0.0037649, -0.0150469, -0.0031781, -0.0115248, 0.0112820
9: -0.0018237, 0.0043666, -0.0021634, 0.0045411, -0.0063648, 0.0065300

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061070, upper bound: 0.0061213
time: 1.96 seconds

## Relational analysis of NS_A1_B2_B1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061043, upper bound: 0.0061167
time: 1.85 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0034764, 0.0008851, -0.0029322, 0.0007347, -0.0042111, 0.0038173
1: -0.0056912, -0.0022570, -0.0056980, -0.0024461, -0.0031048, 0.0034410
2: 0.0307809, 0.0360694, 0.0310964, 0.0350053, -0.0042244, 0.0049731
3: -0.0030860, 0.0010567, -0.0030137, 0.0007534, -0.0038394, 0.0040704
4: -0.0046819, 0.0005060, -0.0046888, 0.0000211, -0.0047029, 0.0051948
5: 0.0095125, 0.0138337, 0.0099649, 0.0134083, -0.0038958, 0.0038688
6: -0.0061631, 0.0013343, -0.0048405, 0.0013443, -0.0075074, 0.0061747
7: 0.9723991, 0.9789928, 0.9735554, 0.9790000, -0.0066009, 0.0054374
8: -0.0157440, -0.0023263, -0.0148064, -0.0035882, -0.0121558, 0.0124802
9: -0.0026564, 0.0050600, -0.0019260, 0.0044192, -0.0070756, 0.0069860

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061613, upper bound: 0.0061539
time: 1.60 seconds

## Relational analysis of NS_A1_B2_B1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061614, upper bound: 0.0061526
time: 1.69 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0039746, 0.0010659, -0.0029046, 0.0007335, -0.0047081, 0.0039706
1: -0.0056864, -0.0020405, -0.0056871, -0.0024493, -0.0032371, 0.0036466
2: 0.0304920, 0.0372347, 0.0311123, 0.0349796, -0.0044876, 0.0061224
3: -0.0031522, 0.0015970, -0.0030100, 0.0007408, -0.0038930, 0.0046071
4: -0.0047057, 0.0010239, -0.0046778, 0.0000074, -0.0047131, 0.0057016
5: 0.0090983, 0.0143399, 0.0099879, 0.0134041, -0.0043057, 0.0043521
6: -0.0077255, 0.0013273, -0.0048254, 0.0013283, -0.0090538, 0.0061527
7: 0.9711915, 0.9789881, 0.9735920, 0.9789887, -0.0077972, 0.0053960
8: -0.0166703, -0.0011708, -0.0147689, -0.0036522, -0.0130181, 0.0135982
9: -0.0033253, 0.0057379, -0.0018889, 0.0044001, -0.0077254, 0.0076268

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061583, upper bound: 0.0061603
time: 1.70 seconds

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061582, upper bound: 0.0061591
time: 2.20 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0033791, 0.0008498, -0.0028008, 0.0007288, -0.0041079, 0.0036506
1: -0.0056910, -0.0022993, -0.0057271, -0.0024612, -0.0030622, 0.0034278
2: 0.0308373, 0.0358420, 0.0311725, 0.0348829, -0.0040456, 0.0046694
3: -0.0030731, 0.0009512, -0.0029962, 0.0007872, -0.0038603, 0.0039475
4: -0.0046817, 0.0004049, -0.0047185, -0.0000439, -0.0046378, 0.0051234
5: 0.0095934, 0.0137349, 0.0100742, 0.0133879, -0.0037945, 0.0036607
6: -0.0058582, 0.0013340, -0.0047686, 0.0013872, -0.0072453, 0.0061026
7: 0.9726347, 0.9789926, 0.9737298, 0.9790299, -0.0063952, 0.0052628
8: -0.0155632, -0.0025518, -0.0146278, -0.0038930, -0.0116703, 0.0120760
9: -0.0025259, 0.0049277, -0.0017495, 0.0043285, -0.0068544, 0.0066773

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061626, upper bound: 0.0061527
time: 2.13 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061614, upper bound: 0.0061527
time: 2.44 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0038741, 0.0010295, -0.0027728, 0.0007275, -0.0046016, 0.0038023
1: -0.0056862, -0.0020841, -0.0057154, -0.0024644, -0.0032218, 0.0036313
2: 0.0305503, 0.0369997, 0.0311888, 0.0348568, -0.0043065, 0.0058109
3: -0.0031388, 0.0014881, -0.0029925, 0.0007736, -0.0039124, 0.0044806
4: -0.0046769, 0.0009194, -0.0047066, -0.0000578, -0.0046191, 0.0056260
5: 0.0091819, 0.0142378, 0.0100975, 0.0133835, -0.0042017, 0.0041403
6: -0.0074104, 0.0013270, -0.0047533, 0.0013700, -0.0087803, 0.0060803
7: 0.9714350, 0.9789879, 0.9737671, 0.9790179, -0.0075829, 0.0052208
8: -0.0164835, -0.0014038, -0.0145896, -0.0039580, -0.0125255, 0.0131858
9: -0.0031904, 0.0056012, -0.0017119, 0.0043092, -0.0074996, 0.0073131

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061597, upper bound: 0.0061583
time: 2.30 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061582, upper bound: 0.0061582
time: 2.55 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0027716, 0.0007275, -0.0031148, 0.0007539, -0.0035255, 0.0038423
1: -0.0058219, -0.0024646, -0.0059212, -0.0024142, -0.0034077, 0.0030105
2: 0.0311895, 0.0348557, 0.0309905, 0.0352239, -0.0040344, 0.0038652
3: -0.0029923, 0.0008970, -0.0030380, 0.0010120, -0.0040044, 0.0039349
4: -0.0048149, -0.0000584, -0.0049159, 0.0001301, -0.0049450, 0.0048575
5: 0.0100985, 0.0133834, 0.0098131, 0.0134664, -0.0033679, 0.0035702
6: -0.0047526, 0.0015265, -0.0050295, 0.0016725, -0.0064251, 0.0065560
7: 0.9737688, 0.9791273, 0.9732752, 0.9792295, -0.0054607, 0.0058522
8: -0.0145880, -0.0039609, -0.0150720, -0.0031648, -0.0114232, 0.0111111
9: -0.0017103, 0.0043083, -0.0021711, 0.0045682, -0.0062785, 0.0064794

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061670, upper bound: 0.0061281
time: 2.36 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061653, upper bound: 0.0061278
time: 1.89 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0026822, 0.0007234, -0.0029486, 0.0007355, -0.0034177, 0.0036720
1: -0.0058217, -0.0024748, -0.0059410, -0.0024443, -0.0028922, 0.0029845
2: 0.0312413, 0.0347724, 0.0310869, 0.0350206, -0.0037793, 0.0036856
3: -0.0029805, 0.0008967, -0.0030159, 0.0010350, -0.0040155, 0.0039126
4: -0.0048147, -0.0001025, -0.0049361, 0.0000292, -0.0048438, 0.0048335
5: 0.0101728, 0.0133694, 0.0099513, 0.0134109, -0.0032381, 0.0034181
6: -0.0047037, 0.0015262, -0.0048494, 0.0017016, -0.0064054, 0.0063756
7: 0.9738874, 0.9791272, 0.9735337, 0.9792500, -0.0053626, 0.0055935
8: -0.0144665, -0.0041680, -0.0148287, -0.0035502, -0.0109162, 0.0106607
9: -0.0015903, 0.0042467, -0.0019479, 0.0044304, -0.0060208, 0.0061946

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061669, upper bound: 0.0061273
time: 2.05 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061652, upper bound: 0.0061271
time: 2.16 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0027432, 0.0007262, -0.0037270, 0.0009761, -0.0037193, 0.0044532
1: -0.0058108, -0.0024678, -0.0059148, -0.0021481, -0.0036627, 0.0031948
2: 0.0312059, 0.0348293, 0.0306356, 0.0366556, -0.0054496, 0.0041936
3: -0.0029886, 0.0008841, -0.0031193, 0.0013285, -0.0043171, 0.0040034
4: -0.0048036, -0.0000724, -0.0049094, 0.0007665, -0.0055701, 0.0048370
5: 0.0101221, 0.0133789, 0.0093042, 0.0140883, -0.0039663, 0.0040748
6: -0.0047371, 0.0015102, -0.0069490, 0.0016631, -0.0064002, 0.0084592
7: 0.9738064, 0.9791160, 0.9717916, 0.9792231, -0.0054167, 0.0073244
8: -0.0145494, -0.0040266, -0.0162100, -0.0017450, -0.0128044, 0.0121834
9: -0.0016722, 0.0042888, -0.0029929, 0.0054010, -0.0070732, 0.0072817

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061715, upper bound: 0.0061096
time: 1.93 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061689, upper bound: 0.0061064
time: 2.03 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0026539, 0.0007221, -0.0035525, 0.0009127, -0.0035667, 0.0042746
1: -0.0058106, -0.0024781, -0.0059339, -0.0022239, -0.0035867, 0.0031562
2: 0.0312577, 0.0347461, 0.0307368, 0.0362475, -0.0049898, 0.0040093
3: -0.0029767, 0.0008839, -0.0030961, 0.0011393, -0.0041160, 0.0039800
4: -0.0048034, -0.0001165, -0.0049288, 0.0005851, -0.0053885, 0.0048122
5: 0.0101963, 0.0133650, 0.0094493, 0.0139111, -0.0037148, 0.0039158
6: -0.0046883, 0.0015099, -0.0064018, 0.0016911, -0.0063794, 0.0079117
7: 0.9739249, 0.9791157, 0.9722145, 0.9792425, -0.0053176, 0.0069012
8: -0.0144280, -0.0042336, -0.0158856, -0.0021497, -0.0122783, 0.0116519
9: -0.0015524, 0.0042272, -0.0027586, 0.0051636, -0.0067160, 0.0069858

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B2_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061715, upper bound: 0.0061069
time: 2.36 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061688, upper bound: 0.0061043
time: 2.12 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0029693, 0.0007364, -0.0036325, 0.0009418, -0.0039111, 0.0043689
1: -0.0057259, -0.0024419, -0.0059233, -0.0021892, -0.0035367, 0.0032453
2: 0.0310749, 0.0350399, 0.0306904, 0.0364347, -0.0053598, 0.0043495
3: -0.0030186, 0.0007857, -0.0031067, 0.0012261, -0.0042447, 0.0038925
4: -0.0047172, 0.0000394, -0.0049180, 0.0006683, -0.0053855, 0.0049574
5: 0.0099341, 0.0134141, 0.0093827, 0.0139924, -0.0040583, 0.0040314
6: -0.0048607, 0.0013854, -0.0066528, 0.0016756, -0.0065363, 0.0080382
7: 0.9735062, 0.9790286, 0.9720206, 0.9792317, -0.0057256, 0.0070080
8: -0.0148569, -0.0035021, -0.0160343, -0.0019641, -0.0128928, 0.0125322
9: -0.0019758, 0.0044447, -0.0028661, 0.0052725, -0.0072483, 0.0073108

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061853, upper bound: 0.0061540
time: 1.53 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061852, upper bound: 0.0061540
time: 2.17 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0029409, 0.0007351, -0.0042426, 0.0011632, -0.0041041, 0.0049777
1: -0.0057150, -0.0024451, -0.0059173, -0.0019239, -0.0037910, 0.0034469
2: 0.0310913, 0.0350134, 0.0303367, 0.0378617, -0.0067703, 0.0046767
3: -0.0030148, 0.0007731, -0.0031878, 0.0018877, -0.0049026, 0.0039609
4: -0.0047061, 0.0000253, -0.0050069, 0.0013025, -0.0060087, 0.0050323
5: 0.0099577, 0.0134097, 0.0088755, 0.0146123, -0.0046545, 0.0045342
6: -0.0048452, 0.0013693, -0.0085660, 0.0016668, -0.0065120, 0.0099353
7: 0.9735438, 0.9790174, 0.9705418, 0.9792256, -0.0056818, 0.0084755
8: -0.0148182, -0.0035681, -0.0171686, -0.0005491, -0.0142692, 0.0136005
9: -0.0019376, 0.0044251, -0.0036851, 0.0061025, -0.0080401, 0.0081103

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061961, upper bound: 0.0061530
time: 2.07 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061959, upper bound: 0.0061530
time: 1.69 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0028390, 0.0007305, -0.0035117, 0.0008979, -0.0037369, 0.0042422
1: -0.0057553, -0.0024568, -0.0059231, -0.0022417, -0.0035136, 0.0032020
2: 0.0311504, 0.0349184, 0.0307604, 0.0361521, -0.0050016, 0.0041580
3: -0.0030013, 0.0008198, -0.0030907, 0.0010950, -0.0040963, 0.0039105
4: -0.0047471, -0.0000250, -0.0049179, 0.0005427, -0.0052898, 0.0048928
5: 0.0100425, 0.0133938, 0.0094832, 0.0138696, -0.0038271, 0.0039107
6: -0.0047895, 0.0014286, -0.0062740, 0.0016753, -0.0064648, 0.0077025
7: 0.9736793, 0.9790589, 0.9723135, 0.9792316, -0.0055523, 0.0067455
8: -0.0146796, -0.0038045, -0.0158097, -0.0022443, -0.0124353, 0.0120052
9: -0.0018007, 0.0043548, -0.0027039, 0.0051081, -0.0069089, 0.0070587

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061843, upper bound: 0.0061540
time: 1.76 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061839, upper bound: 0.0061540
time: 1.69 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0028101, 0.0007292, -0.0041184, 0.0011181, -0.0039282, 0.0048476
1: -0.0057437, -0.0024601, -0.0059171, -0.0019779, -0.0037657, 0.0033954
2: 0.0311672, 0.0348915, 0.0304087, 0.0375712, -0.0064040, 0.0044828
3: -0.0029975, 0.0008063, -0.0031713, 0.0017530, -0.0047505, 0.0039776
4: -0.0047353, -0.0000393, -0.0049118, 0.0011734, -0.0059087, 0.0048724
5: 0.0100665, 0.0133893, 0.0089787, 0.0144861, -0.0044196, 0.0044106
6: -0.0047737, 0.0014115, -0.0081765, 0.0016666, -0.0064402, 0.0095879
7: 0.9737176, 0.9790469, 0.9708429, 0.9792255, -0.0055079, 0.0082040
8: -0.0146403, -0.0038715, -0.0169377, -0.0008372, -0.0138032, 0.0130662
9: -0.0017620, 0.0043349, -0.0035184, 0.0059336, -0.0076955, 0.0078533

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061944, upper bound: 0.0061530
time: 1.93 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061932, upper bound: 0.0061530
time: 2.07 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0031726, 0.0007749, -0.0025251, 0.0007163, -0.0038889, 0.0033000
1: -0.0059162, -0.0023891, -0.0058184, -0.0024928, -0.0029983, 0.0034293
2: 0.0309570, 0.0353590, 0.0313324, 0.0346261, -0.0036691, 0.0040266
3: -0.0030456, 0.0010062, -0.0029596, 0.0008929, -0.0039385, 0.0039658
4: -0.0049108, 0.0001902, -0.0048113, -0.0001802, -0.0047306, 0.0050015
5: 0.0097651, 0.0135251, 0.0103034, 0.0133450, -0.0035799, 0.0032217
6: -0.0052106, 0.0016652, -0.0046179, 0.0015213, -0.0067319, 0.0062830
7: 0.9731353, 0.9792244, 0.9740960, 0.9791238, -0.0059885, 0.0051284
8: -0.0151793, -0.0030308, -0.0142529, -0.0045324, -0.0106469, 0.0112221
9: -0.0022486, 0.0046468, -0.0013794, 0.0041384, -0.0063870, 0.0060262

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061282, upper bound: 0.0061689
time: 1.94 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061280, upper bound: 0.0061682
time: 1.90 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0030587, 0.0007404, -0.0024165, 0.0007114, -0.0037701, 0.0031569
1: -0.0059160, -0.0024316, -0.0058478, -0.0025053, -0.0029601, 0.0028414
2: 0.0310231, 0.0351231, 0.0313954, 0.0345248, -0.0035018, 0.0037277
3: -0.0030305, 0.0010060, -0.0029452, 0.0009270, -0.0039575, 0.0039512
4: -0.0049106, 0.0000836, -0.0048413, -0.0002339, -0.0046767, 0.0049248
5: 0.0098598, 0.0134280, 0.0103938, 0.0133281, -0.0034683, 0.0030343
6: -0.0049096, 0.0016649, -0.0045584, 0.0015646, -0.0064742, 0.0062233
7: 0.9733875, 0.9792243, 0.9742402, 0.9791541, -0.0057665, 0.0049840
8: -0.0149784, -0.0032950, -0.0141051, -0.0047844, -0.0101940, 0.0108101
9: -0.0020957, 0.0045064, -0.0012335, 0.0040634, -0.0061591, 0.0057399

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061284, upper bound: 0.0061649
time: 1.94 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061271, upper bound: 0.0061646
time: 2.10 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0035716, 0.0009197, -0.0027319, 0.0007257, -0.0042973, 0.0036516
1: -0.0059084, -0.0022156, -0.0058071, -0.0024691, -0.0031389, 0.0035915
2: 0.0307257, 0.0362921, 0.0312125, 0.0348188, -0.0040931, 0.0050797
3: -0.0030986, 0.0011600, -0.0029871, 0.0008798, -0.0039785, 0.0041471
4: -0.0049028, 0.0006050, -0.0047998, -0.0000780, -0.0048249, 0.0054048
5: 0.0094334, 0.0139305, 0.0101315, 0.0133772, -0.0039438, 0.0037990
6: -0.0064618, 0.0016536, -0.0047309, 0.0015048, -0.0079665, 0.0063846
7: 0.9721683, 0.9792163, 0.9738214, 0.9791120, -0.0069438, 0.0053949
8: -0.0159210, -0.0021054, -0.0145341, -0.0040528, -0.0118683, 0.0124287
9: -0.0027843, 0.0051896, -0.0016571, 0.0042810, -0.0070653, 0.0068467

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061096, upper bound: 0.0061707
time: 2.01 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061064, upper bound: 0.0061677
time: 1.80 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0034009, 0.0008577, -0.0026424, 0.0007216, -0.0041225, 0.0035002
1: -0.0059281, -0.0022898, -0.0058069, -0.0024794, -0.0031041, 0.0035171
2: 0.0308247, 0.0358928, 0.0312644, 0.0347354, -0.0039107, 0.0046285
3: -0.0030760, 0.0010200, -0.0029752, 0.0008796, -0.0039556, 0.0039952
4: -0.0049229, 0.0004275, -0.0047996, -0.0001222, -0.0048007, 0.0052271
5: 0.0095753, 0.0137570, 0.0102059, 0.0133633, -0.0037879, 0.0035511
6: -0.0059264, 0.0016826, -0.0046820, 0.0015045, -0.0074309, 0.0063646
7: 0.9725819, 0.9792366, 0.9739401, 0.9791120, -0.0065300, 0.0052965
8: -0.0156037, -0.0025014, -0.0144124, -0.0042603, -0.0113434, 0.0119111
9: -0.0025551, 0.0049573, -0.0015369, 0.0042193, -0.0067744, 0.0064943

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061069, upper bound: 0.0061707
time: 2.02 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061043, upper bound: 0.0061677
time: 1.82 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0034664, 0.0008815, -0.0029595, 0.0007360, -0.0042024, 0.0038410
1: -0.0059179, -0.0022613, -0.0057231, -0.0024430, -0.0031914, 0.0034618
2: 0.0307867, 0.0360462, 0.0310805, 0.0350308, -0.0042441, 0.0049657
3: -0.0030847, 0.0010459, -0.0030173, 0.0007825, -0.0038672, 0.0040633
4: -0.0049125, 0.0004956, -0.0047144, 0.0000346, -0.0049471, 0.0052100
5: 0.0095208, 0.0138236, 0.0099422, 0.0134126, -0.0038918, 0.0038814
6: -0.0061320, 0.0016676, -0.0048554, 0.0013812, -0.0075133, 0.0065230
7: 0.9724231, 0.9792261, 0.9735191, 0.9790258, -0.0066026, 0.0057069
8: -0.0157256, -0.0023493, -0.0148436, -0.0035249, -0.0122007, 0.0124943
9: -0.0026431, 0.0050466, -0.0019626, 0.0044380, -0.0070811, 0.0070092

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061540, upper bound: 0.0061847
time: 2.12 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061540, upper bound: 0.0061845
time: 2.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0040981, 0.0011107, -0.0029314, 0.0007347, -0.0048327, 0.0040421
1: -0.0059108, -0.0019868, -0.0057119, -0.0024462, -0.0033914, 0.0037251
2: 0.0304205, 0.0375235, 0.0310969, 0.0350045, -0.0045841, 0.0064266
3: -0.0031686, 0.0017309, -0.0030136, 0.0007695, -0.0039381, 0.0047445
4: -0.0049053, 0.0011522, -0.0047030, 0.0000206, -0.0049259, 0.0058552
5: 0.0089957, 0.0144654, 0.0099656, 0.0134082, -0.0044126, 0.0044997
6: -0.0081126, 0.0016572, -0.0048400, 0.0013648, -0.0094774, 0.0064972
7: 0.9708924, 0.9792189, 0.9735566, 0.9790142, -0.0081218, 0.0056623
8: -0.0168998, -0.0008844, -0.0148053, -0.0035902, -0.0133096, 0.0139209
9: -0.0034911, 0.0059058, -0.0019248, 0.0044186, -0.0079096, 0.0078307

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061530, upper bound: 0.0061951
time: 2.01 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061530, upper bound: 0.0061949
time: 1.86 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0033474, 0.0008383, -0.0028284, 0.0007300, -0.0040774, 0.0036667
1: -0.0059177, -0.0023131, -0.0057530, -0.0024580, -0.0031499, 0.0034400
2: 0.0308557, 0.0357678, 0.0311566, 0.0349086, -0.0040529, 0.0046112
3: -0.0030689, 0.0010080, -0.0029999, 0.0008172, -0.0038861, 0.0040079
4: -0.0049123, 0.0003719, -0.0047448, -0.0000303, -0.0048820, 0.0051167
5: 0.0096198, 0.0137027, 0.0100513, 0.0133922, -0.0037724, 0.0036514
6: -0.0057588, 0.0016674, -0.0047837, 0.0014253, -0.0071841, 0.0064510
7: 0.9727116, 0.9792260, 0.9736934, 0.9790566, -0.0063450, 0.0055326
8: -0.0155043, -0.0026253, -0.0146652, -0.0038292, -0.0116751, 0.0120399
9: -0.0024833, 0.0048846, -0.0017865, 0.0043475, -0.0068308, 0.0066711

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B2_B2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061540, upper bound: 0.0061840
time: 2.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_A1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061540, upper bound: 0.0061835
time: 2.00 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0039789, 0.0010675, -0.0027997, 0.0007287, -0.0047076, 0.0038672
1: -0.0059106, -0.0020386, -0.0057416, -0.0024613, -0.0033428, 0.0037029
2: 0.0304896, 0.0372446, 0.0311732, 0.0348819, -0.0043923, 0.0060714
3: -0.0031528, 0.0016016, -0.0029961, 0.0008039, -0.0039567, 0.0045977
4: -0.0049051, 0.0010283, -0.0047332, -0.0000444, -0.0048607, 0.0057615
5: 0.0090948, 0.0143442, 0.0100751, 0.0133877, -0.0042930, 0.0042691
6: -0.0077387, 0.0016569, -0.0047680, 0.0014084, -0.0091471, 0.0064249
7: 0.9711813, 0.9792188, 0.9737313, 0.9790448, -0.0078635, 0.0054875
8: -0.0166781, -0.0011609, -0.0146263, -0.0038955, -0.0127827, 0.0134654
9: -0.0033310, 0.0057436, -0.0017481, 0.0043278, -0.0076588, 0.0074917

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B2_B2_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061530, upper bound: 0.0061934
time: 1.81 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061530, upper bound: 0.0061920
time: 2.23 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0031148, 0.0007539, -0.0027716, 0.0007275, -0.0038423, 0.0035255
1: -0.0059212, -0.0024142, -0.0058219, -0.0024646, -0.0030105, 0.0034077
2: 0.0309905, 0.0352239, 0.0311895, 0.0348557, -0.0038652, 0.0040344
3: -0.0030380, 0.0010120, -0.0029923, 0.0008970, -0.0039349, 0.0040044
4: -0.0049159, 0.0001301, -0.0048149, -0.0000584, -0.0048575, 0.0049450
5: 0.0098131, 0.0134664, 0.0100985, 0.0133834, -0.0035702, 0.0033679
6: -0.0050295, 0.0016725, -0.0047526, 0.0015265, -0.0065560, 0.0064251
7: 0.9732752, 0.9792295, 0.9737688, 0.9791273, -0.0058522, 0.0054607
8: -0.0150720, -0.0031648, -0.0145880, -0.0039609, -0.0111111, 0.0114232
9: -0.0021711, 0.0045682, -0.0017103, 0.0043083, -0.0064794, 0.0062785

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A2_B1_A1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061281, upper bound: 0.0061670
time: 1.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061278, upper bound: 0.0061652
time: 1.97 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0029486, 0.0007355, -0.0026822, 0.0007234, -0.0036720, 0.0034177
1: -0.0059410, -0.0024443, -0.0058217, -0.0024748, -0.0029845, 0.0028922
2: 0.0310869, 0.0350206, 0.0312413, 0.0347724, -0.0036856, 0.0037793
3: -0.0030159, 0.0010350, -0.0029805, 0.0008967, -0.0039126, 0.0040155
4: -0.0049361, 0.0000292, -0.0048147, -0.0001025, -0.0048335, 0.0048438
5: 0.0099513, 0.0134109, 0.0101728, 0.0133694, -0.0034181, 0.0032381
6: -0.0048494, 0.0017016, -0.0047037, 0.0015262, -0.0063756, 0.0064054
7: 0.9735337, 0.9792500, 0.9738874, 0.9791272, -0.0055935, 0.0053626
8: -0.0148287, -0.0035502, -0.0144665, -0.0041680, -0.0106607, 0.0109162
9: -0.0019479, 0.0044304, -0.0015903, 0.0042467, -0.0061946, 0.0060208

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061274, upper bound: 0.0061669
time: 2.26 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061271, upper bound: 0.0061652
time: 1.63 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0037270, 0.0009761, -0.0027432, 0.0007262, -0.0044532, 0.0037193
1: -0.0059148, -0.0021481, -0.0058108, -0.0024678, -0.0031948, 0.0036627
2: 0.0306356, 0.0366556, 0.0312059, 0.0348293, -0.0041936, 0.0054496
3: -0.0031193, 0.0013285, -0.0029886, 0.0008841, -0.0040034, 0.0043171
4: -0.0049094, 0.0007665, -0.0048036, -0.0000724, -0.0048370, 0.0055701
5: 0.0093042, 0.0140883, 0.0101221, 0.0133789, -0.0040748, 0.0039663
6: -0.0069490, 0.0016631, -0.0047371, 0.0015102, -0.0084592, 0.0064002
7: 0.9717916, 0.9792231, 0.9738064, 0.9791160, -0.0073244, 0.0054167
8: -0.0162100, -0.0017450, -0.0145494, -0.0040266, -0.0121834, 0.0128044
9: -0.0029929, 0.0054010, -0.0016722, 0.0042888, -0.0072817, 0.0070732

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A2_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061096, upper bound: 0.0061715
time: 1.87 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_A1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061064, upper bound: 0.0061689
time: 1.85 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0035525, 0.0009127, -0.0026539, 0.0007221, -0.0042746, 0.0035667
1: -0.0059339, -0.0022239, -0.0058106, -0.0024781, -0.0031562, 0.0035867
2: 0.0307368, 0.0362475, 0.0312577, 0.0347461, -0.0040093, 0.0049898
3: -0.0030961, 0.0011393, -0.0029767, 0.0008839, -0.0039800, 0.0041160
4: -0.0049288, 0.0005851, -0.0048034, -0.0001165, -0.0048122, 0.0053885
5: 0.0094493, 0.0139111, 0.0101963, 0.0133650, -0.0039158, 0.0037148
6: -0.0064018, 0.0016911, -0.0046883, 0.0015099, -0.0079117, 0.0063794
7: 0.9722145, 0.9792425, 0.9739249, 0.9791157, -0.0069012, 0.0053176
8: -0.0158856, -0.0021497, -0.0144280, -0.0042336, -0.0116519, 0.0122783
9: -0.0027586, 0.0051636, -0.0015524, 0.0042272, -0.0069858, 0.0067160

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A2_B1_A2_A2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061069, upper bound: 0.0061715
time: 1.94 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_A2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061043, upper bound: 0.0061689
time: 2.09 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.63 + 596.96 = 601.58 seconds
