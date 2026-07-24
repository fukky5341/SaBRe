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
execution time: IAR + RelationalAnalysis = 1.52 + 2.95 = 4.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0062749, upper bound: 0.0062749

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 186

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062569, upper bound: 0.0062173
time: 2.20 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062586, upper bound: 0.0062585
time: 1.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.80 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.80
Output dim: 7, lower bound: -0.0062569, upper bound: 0.0062173
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.80
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
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 186

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062174, upper bound: 0.0062173
time: 1.37 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062174, upper bound: 0.0062173
time: 2.13 seconds

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

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 186

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062172, upper bound: 0.0062569
time: 2.01 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062172, upper bound: 0.0062585
time: 2.08 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.79 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.79
Output dim: 7, lower bound: -0.0062174, upper bound: 0.0062173
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.79
Output dim: 7, lower bound: -0.0062174, upper bound: 0.0062173
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.79
Output dim: 7, lower bound: -0.0062172, upper bound: 0.0062569
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.79
Output dim: 7, lower bound: -0.0062172, upper bound: 0.0062585

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0037215, 0.0009741, -0.0037215, 0.0009741, -0.0046956, 0.0046956
1: -0.0057615, -0.0021505, -0.0057615, -0.0021505, -0.0036110, 0.0036110
2: 0.0306388, 0.0366429, 0.0306388, 0.0366429, -0.0060041, 0.0060041
3: -0.0031186, 0.0013226, -0.0031186, 0.0013226, -0.0044412, 0.0044412
4: -0.0047534, 0.0007608, -0.0047534, 0.0007608, -0.0055143, 0.0055143
5: 0.0093087, 0.0140828, 0.0093087, 0.0140828, -0.0047741, 0.0047741
6: -0.0069319, 0.0014377, -0.0069319, 0.0014377, -0.0083696, 0.0083696
7: 0.9718048, 0.9790652, 0.9718048, 0.9790652, -0.0072604, 0.0072604
8: -0.0161998, -0.0017576, -0.0161998, -0.0017576, -0.0144422, 0.0144422
9: -0.0029856, 0.0053936, -0.0029856, 0.0053936, -0.0083792, 0.0083792

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062194, upper bound: 0.0062073
time: 1.54 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062194, upper bound: 0.0062095
time: 2.19 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0037215, 0.0009741, -0.0039181, 0.0010454, -0.0047670, 0.0048921
1: -0.0057615, -0.0021505, -0.0060239, -0.0020650, -0.0036964, 0.0038735
2: 0.0306388, 0.0366429, 0.0305248, 0.0371025, -0.0064637, 0.0061180
3: -0.0031186, 0.0013226, -0.0031447, 0.0015357, -0.0046543, 0.0044673
4: -0.0047534, 0.0007608, -0.0050203, 0.0009651, -0.0057185, 0.0057812
5: 0.0093087, 0.0140828, 0.0091453, 0.0142825, -0.0049738, 0.0049375
6: -0.0069319, 0.0014377, -0.0075481, 0.0018235, -0.0087554, 0.0089858
7: 0.9718048, 0.9790652, 0.9713286, 0.9793352, -0.0075305, 0.0077366
8: -0.0161998, -0.0017576, -0.0165651, -0.0013019, -0.0148979, 0.0148075
9: -0.0029856, 0.0053936, -0.0032494, 0.0056609, -0.0086465, 0.0086430

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062194, upper bound: 0.0062073
time: 1.85 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062194, upper bound: 0.0062095
time: 1.96 seconds

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

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062473
time: 1.92 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062486
time: 2.03 seconds

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

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

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
time: 2.06 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.76 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 7, lower bound: -0.0062194, upper bound: 0.0062073
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 7, lower bound: -0.0062194, upper bound: 0.0062095
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 7, lower bound: -0.0062194, upper bound: 0.0062073
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 7, lower bound: -0.0062194, upper bound: 0.0062095
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062473
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062486
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062486
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.76
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062500

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0034787, 0.0008860, -0.0036853, 0.0009609, -0.0044396, 0.0045713
1: -0.0057049, -0.0022560, -0.0057395, -0.0021662, -0.0035387, 0.0034835
2: 0.0307795, 0.0360750, 0.0306598, 0.0365581, -0.0057786, 0.0054152
3: -0.0030863, 0.0010593, -0.0031138, 0.0012833, -0.0043696, 0.0041730
4: -0.0046959, 0.0005084, -0.0047311, 0.0007232, -0.0054190, 0.0052395
5: 0.0095106, 0.0138361, 0.0093388, 0.0140460, -0.0045354, 0.0044973
6: -0.0061705, 0.0013545, -0.0068183, 0.0014054, -0.0075759, 0.0081728
7: 0.9723933, 0.9790071, 0.9718927, 0.9790427, -0.0066494, 0.0071144
8: -0.0157484, -0.0023208, -0.0161325, -0.0018417, -0.0139067, 0.0138116
9: -0.0026596, 0.0050632, -0.0029369, 0.0053443, -0.0080039, 0.0080002

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061978
time: 1.46 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062033, upper bound: 0.0062015
time: 1.67 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0036580, 0.0009510, -0.0036956, 0.0009647, -0.0046226, 0.0046466
1: -0.0057145, -0.0021781, -0.0057424, -0.0021617, -0.0035528, 0.0035643
2: 0.0306756, 0.0364942, 0.0306538, 0.0365822, -0.0059066, 0.0058404
3: -0.0031101, 0.0012537, -0.0031151, 0.0012945, -0.0044046, 0.0043688
4: -0.0047057, 0.0006948, -0.0047340, 0.0007339, -0.0054395, 0.0054288
5: 0.0093615, 0.0140182, 0.0093303, 0.0140565, -0.0046949, 0.0046880
6: -0.0067326, 0.0013686, -0.0068506, 0.0014097, -0.0081423, 0.0082192
7: 0.9719589, 0.9790169, 0.9718676, 0.9790456, -0.0070868, 0.0071493
8: -0.0160816, -0.0019051, -0.0161516, -0.0018178, -0.0142638, 0.0142466
9: -0.0029002, 0.0053071, -0.0029508, 0.0053583, -0.0082586, 0.0082579

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061994
time: 1.91 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062033, upper bound: 0.0062033
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0034787, 0.0008860, -0.0038784, 0.0010310, -0.0045097, 0.0047644
1: -0.0057049, -0.0022560, -0.0060027, -0.0020823, -0.0036226, 0.0037467
2: 0.0307795, 0.0360750, 0.0305478, 0.0370097, -0.0062302, 0.0055271
3: -0.0030863, 0.0010593, -0.0031394, 0.0014927, -0.0045790, 0.0041987
4: -0.0046959, 0.0005084, -0.0049988, 0.0009239, -0.0056197, 0.0055072
5: 0.0095106, 0.0138361, 0.0091783, 0.0142422, -0.0047316, 0.0046578
6: -0.0061705, 0.0013545, -0.0074237, 0.0017924, -0.0079629, 0.0087782
7: 0.9723933, 0.9790071, 0.9714248, 0.9793135, -0.0069202, 0.0075823
8: -0.0157484, -0.0023208, -0.0164914, -0.0013939, -0.0143545, 0.0141706
9: -0.0026596, 0.0050632, -0.0031961, 0.0056070, -0.0082666, 0.0082594

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062190, upper bound: 0.0061889
time: 2.15 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062326, upper bound: 0.0061918
time: 1.98 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0036580, 0.0009510, -0.0038893, 0.0010350, -0.0046930, 0.0048403
1: -0.0057145, -0.0021781, -0.0060039, -0.0020775, -0.0036370, 0.0038258
2: 0.0306756, 0.0364942, 0.0305415, 0.0370353, -0.0063597, 0.0059527
3: -0.0031101, 0.0012537, -0.0031409, 0.0015046, -0.0046147, 0.0043945
4: -0.0047057, 0.0006948, -0.0050000, 0.0009352, -0.0056409, 0.0056947
5: 0.0093615, 0.0140182, 0.0091692, 0.0142533, -0.0048917, 0.0048490
6: -0.0067326, 0.0013686, -0.0074580, 0.0017941, -0.0085267, 0.0088267
7: 0.9719589, 0.9790169, 0.9713982, 0.9793146, -0.0073557, 0.0076187
8: -0.0160816, -0.0019051, -0.0165117, -0.0013685, -0.0147131, 0.0146067
9: -0.0029002, 0.0053071, -0.0032108, 0.0056219, -0.0085221, 0.0085179

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062190, upper bound: 0.0061905
time: 2.04 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062327, upper bound: 0.0061937
time: 1.90 seconds

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

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061838, upper bound: 0.0062289
time: 2.05 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061937, upper bound: 0.0062314
time: 1.90 seconds

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

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061838, upper bound: 0.0062299
time: 1.77 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061937, upper bound: 0.0062327
time: 1.99 seconds

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

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061867, upper bound: 0.0062294
time: 1.95 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061977, upper bound: 0.0062328
time: 1.66 seconds

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

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061867, upper bound: 0.0062304
time: 1.96 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061977, upper bound: 0.0062341
time: 1.45 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.14 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061978
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0062033, upper bound: 0.0062015
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061994
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0062033, upper bound: 0.0062033
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0062190, upper bound: 0.0061889
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0062326, upper bound: 0.0061918
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0062190, upper bound: 0.0061905
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0062327, upper bound: 0.0061937
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0061838, upper bound: 0.0062289
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0061937, upper bound: 0.0062314
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0061838, upper bound: 0.0062299
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0061937, upper bound: 0.0062327
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0061867, upper bound: 0.0062294
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0061977, upper bound: 0.0062328
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0061867, upper bound: 0.0062304
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 7, lower bound: -0.0061977, upper bound: 0.0062341

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0031561, 0.0007689, -0.0029020, 0.0007333, -0.0038895, 0.0036709
1: -0.0057031, -0.0023962, -0.0058337, -0.0024496, -0.0029530, 0.0034374
2: 0.0309665, 0.0353205, 0.0311139, 0.0349772, -0.0040106, 0.0042066
3: -0.0030434, 0.0007593, -0.0030097, 0.0009106, -0.0039540, 0.0037690
4: -0.0046940, 0.0001731, -0.0048268, 0.0000061, -0.0047001, 0.0049999
5: 0.0097788, 0.0135084, 0.0099900, 0.0134036, -0.0036249, 0.0035183
6: -0.0051590, 0.0013518, -0.0048239, 0.0015438, -0.0067028, 0.0061758
7: 0.9731752, 0.9790052, 0.9735956, 0.9791394, -0.0059643, 0.0054095
8: -0.0151487, -0.0030689, -0.0147654, -0.0036583, -0.0114905, 0.0116964
9: -0.0022266, 0.0046244, -0.0018854, 0.0043983, -0.0066249, 0.0065098

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061253, upper bound: 0.0061512
time: 1.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061767, upper bound: 0.0061837
time: 1.70 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0034488, 0.0008751, -0.0033245, 0.0008300, -0.0042788, 0.0041996
1: -0.0057048, -0.0022690, -0.0057387, -0.0023230, -0.0033818, 0.0034697
2: 0.0307969, 0.0360050, 0.0308689, 0.0357144, -0.0049175, 0.0051361
3: -0.0030823, 0.0010268, -0.0030658, 0.0008920, -0.0039744, 0.0040927
4: -0.0046958, 0.0004773, -0.0047302, 0.0003481, -0.0050439, 0.0052075
5: 0.0095354, 0.0138057, 0.0096388, 0.0136795, -0.0041440, 0.0041669
6: -0.0060768, 0.0013544, -0.0056870, 0.0014041, -0.0074809, 0.0070414
7: 0.9724658, 0.9790069, 0.9727671, 0.9790418, -0.0065760, 0.0062399
8: -0.0156928, -0.0023901, -0.0154618, -0.0026784, -0.0130144, 0.0130716
9: -0.0026195, 0.0050226, -0.0024526, 0.0048535, -0.0074730, 0.0074752

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061985, upper bound: 0.0061904
time: 2.10 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061985, upper bound: 0.0062015
time: 2.08 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0033392, 0.0008353, -0.0029139, 0.0007339, -0.0040731, 0.0037492
1: -0.0057126, -0.0023166, -0.0058370, -0.0024482, -0.0030208, 0.0035204
2: 0.0308604, 0.0357487, 0.0311070, 0.0349882, -0.0041278, 0.0046417
3: -0.0030678, 0.0009080, -0.0030113, 0.0009145, -0.0039823, 0.0039193
4: -0.0047037, 0.0003634, -0.0048303, 0.0000120, -0.0047157, 0.0051937
5: 0.0096266, 0.0136944, 0.0099802, 0.0134055, -0.0037789, 0.0037142
6: -0.0057331, 0.0013658, -0.0048304, 0.0015487, -0.0072819, 0.0061963
7: 0.9727313, 0.9790150, 0.9735798, 0.9791430, -0.0064117, 0.0054352
8: -0.0154891, -0.0026443, -0.0147815, -0.0036308, -0.0118584, 0.0121372
9: -0.0024723, 0.0048735, -0.0019013, 0.0044065, -0.0068788, 0.0067748

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061245, upper bound: 0.0061514
time: 1.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061767, upper bound: 0.0061851
time: 1.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0036276, 0.0009400, -0.0033347, 0.0008337, -0.0044613, 0.0042747
1: -0.0057144, -0.0021913, -0.0057415, -0.0023186, -0.0033959, 0.0035503
2: 0.0306932, 0.0364232, 0.0308630, 0.0357382, -0.0050450, 0.0055602
3: -0.0031061, 0.0012208, -0.0030672, 0.0009031, -0.0040092, 0.0042879
4: -0.0047056, 0.0006632, -0.0047331, 0.0003587, -0.0050643, 0.0053963
5: 0.0093868, 0.0139874, 0.0096303, 0.0136898, -0.0043030, 0.0043571
6: -0.0066375, 0.0013685, -0.0057190, 0.0014083, -0.0080458, 0.0070875
7: 0.9720324, 0.9790168, 0.9727424, 0.9790447, -0.0070123, 0.0062744
8: -0.0160252, -0.0019754, -0.0154807, -0.0026548, -0.0133705, 0.0135053
9: -0.0028595, 0.0052658, -0.0024663, 0.0048673, -0.0077269, 0.0077321

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061985, upper bound: 0.0061911
time: 2.13 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061985, upper bound: 0.0062033
time: 1.84 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0031561, 0.0007689, -0.0031450, 0.0007648, -0.0039210, 0.0039139
1: -0.0057031, -0.0023962, -0.0060964, -0.0024011, -0.0033020, 0.0037002
2: 0.0309665, 0.0353205, 0.0309730, 0.0352944, -0.0043278, 0.0043475
3: -0.0030434, 0.0007593, -0.0030420, 0.0012149, -0.0042584, 0.0038013
4: -0.0046940, 0.0001731, -0.0050941, 0.0001614, -0.0048555, 0.0052671
5: 0.0097788, 0.0135084, 0.0097881, 0.0134970, -0.0037182, 0.0037203
6: -0.0051590, 0.0013518, -0.0051240, 0.0019300, -0.0070891, 0.0064758
7: 0.9731752, 0.9790052, 0.9732021, 0.9794098, -0.0062346, 0.0058030
8: -0.0151487, -0.0030689, -0.0151279, -0.0030949, -0.0120539, 0.0120590
9: -0.0022266, 0.0046244, -0.0022115, 0.0046092, -0.0068357, 0.0068359

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061578, upper bound: 0.0061457
time: 1.87 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062034, upper bound: 0.0061762
time: 1.43 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0034488, 0.0008751, -0.0035229, 0.0009020, -0.0043508, 0.0043980
1: -0.0057048, -0.0022690, -0.0060019, -0.0022368, -0.0034680, 0.0037329
2: 0.0307969, 0.0360050, 0.0307539, 0.0361783, -0.0053815, 0.0052511
3: -0.0030823, 0.0010268, -0.0030922, 0.0011072, -0.0041895, 0.0041190
4: -0.0046958, 0.0004773, -0.0049980, 0.0005544, -0.0052501, 0.0054753
5: 0.0095354, 0.0138057, 0.0094738, 0.0138810, -0.0043456, 0.0043319
6: -0.0060768, 0.0013544, -0.0063091, 0.0017911, -0.0078679, 0.0076635
7: 0.9724658, 0.9790069, 0.9722862, 0.9793125, -0.0068467, 0.0067207
8: -0.0156928, -0.0023901, -0.0158306, -0.0022183, -0.0134745, 0.0134405
9: -0.0026195, 0.0050226, -0.0027189, 0.0051234, -0.0077429, 0.0077415

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062277, upper bound: 0.0061826
time: 1.44 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062277, upper bound: 0.0061919
time: 1.28 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0033392, 0.0008353, -0.0031576, 0.0007694, -0.0041086, 0.0039929
1: -0.0057126, -0.0023166, -0.0060986, -0.0023956, -0.0033170, 0.0037820
2: 0.0308604, 0.0357487, 0.0309657, 0.0353238, -0.0044634, 0.0047830
3: -0.0030678, 0.0009080, -0.0030436, 0.0012175, -0.0042853, 0.0039516
4: -0.0047037, 0.0003634, -0.0050963, 0.0001745, -0.0048783, 0.0054597
5: 0.0096266, 0.0136944, 0.0097776, 0.0135098, -0.0038832, 0.0039168
6: -0.0057331, 0.0013658, -0.0051634, 0.0019333, -0.0076665, 0.0065293
7: 0.9727313, 0.9790150, 0.9731716, 0.9794121, -0.0066808, 0.0058434
8: -0.0154891, -0.0026443, -0.0151513, -0.0030657, -0.0124235, 0.0125070
9: -0.0024723, 0.0048735, -0.0022285, 0.0046263, -0.0070986, 0.0071019

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061565, upper bound: 0.0061459
time: 1.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062034, upper bound: 0.0061775
time: 1.84 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0036276, 0.0009400, -0.0035337, 0.0009059, -0.0045336, 0.0044737
1: -0.0057144, -0.0021913, -0.0060031, -0.0022321, -0.0034823, 0.0038118
2: 0.0306932, 0.0364232, 0.0307477, 0.0362035, -0.0055103, 0.0056755
3: -0.0031061, 0.0012208, -0.0030936, 0.0011189, -0.0042250, 0.0043144
4: -0.0047056, 0.0006632, -0.0049991, 0.0005655, -0.0052711, 0.0056623
5: 0.0093868, 0.0139874, 0.0094649, 0.0138920, -0.0045052, 0.0045225
6: -0.0066375, 0.0013685, -0.0063429, 0.0017928, -0.0084303, 0.0077114
7: 0.9720324, 0.9790168, 0.9722601, 0.9793137, -0.0072813, 0.0067567
8: -0.0160252, -0.0019754, -0.0158506, -0.0021933, -0.0138319, 0.0138752
9: -0.0028595, 0.0052658, -0.0027334, 0.0051380, -0.0079975, 0.0079992

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062277, upper bound: 0.0061838
time: 1.53 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062277, upper bound: 0.0061938
time: 2.05 seconds

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

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061177, upper bound: 0.0061750
time: 1.89 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061707, upper bound: 0.0062138
time: 1.97 seconds

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

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061891, upper bound: 0.0062187
time: 1.96 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061892, upper bound: 0.0062314
time: 2.10 seconds

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

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061164, upper bound: 0.0061755
time: 1.89 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061707, upper bound: 0.0062147
time: 1.44 seconds

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

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061892, upper bound: 0.0062190
time: 2.09 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061892, upper bound: 0.0062327
time: 1.96 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0033599, 0.0008429, -0.0031450, 0.0007648, -0.0041248, 0.0039878
1: -0.0059674, -0.0023076, -0.0060964, -0.0024011, -0.0035663, 0.0037887
2: 0.0308484, 0.0357972, 0.0309730, 0.0352944, -0.0044460, 0.0048241
3: -0.0030705, 0.0010655, -0.0030420, 0.0012149, -0.0042854, 0.0041075
4: -0.0049629, 0.0003849, -0.0050941, 0.0001614, -0.0051243, 0.0054790
5: 0.0096093, 0.0137154, 0.0097881, 0.0134970, -0.0038877, 0.0039274
6: -0.0057981, 0.0017404, -0.0051240, 0.0019300, -0.0077281, 0.0068644
7: 0.9726812, 0.9792771, 0.9732021, 0.9794098, -0.0067286, 0.0060750
8: -0.0155276, -0.0025963, -0.0151279, -0.0030949, -0.0124327, 0.0125316
9: -0.0025001, 0.0049016, -0.0022115, 0.0046092, -0.0071093, 0.0071132

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061177, upper bound: 0.0061750
time: 1.45 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061734, upper bound: 0.0062142
time: 1.97 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0036529, 0.0009492, -0.0035229, 0.0009020, -0.0045549, 0.0044721
1: -0.0059690, -0.0021803, -0.0060019, -0.0022368, -0.0037323, 0.0038216
2: 0.0306786, 0.0364824, 0.0307539, 0.0361783, -0.0054998, 0.0057285
3: -0.0031094, 0.0012482, -0.0030922, 0.0011072, -0.0042166, 0.0043404
4: -0.0049645, 0.0006895, -0.0049980, 0.0005544, -0.0055189, 0.0056875
5: 0.0093658, 0.0140131, 0.0094738, 0.0138810, -0.0045153, 0.0045393
6: -0.0067168, 0.0017428, -0.0063091, 0.0017911, -0.0085079, 0.0080519
7: 0.9719711, 0.9792788, 0.9722862, 0.9793125, -0.0073414, 0.0069926
8: -0.0160723, -0.0019168, -0.0158306, -0.0022183, -0.0138540, 0.0139138
9: -0.0028934, 0.0053002, -0.0027189, 0.0051234, -0.0080168, 0.0080192

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061923, upper bound: 0.0062196
time: 1.97 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061924, upper bound: 0.0062327
time: 2.06 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0035353, 0.0009065, -0.0031576, 0.0007694, -0.0043047, 0.0040641
1: -0.0059729, -0.0022314, -0.0060986, -0.0023956, -0.0035773, 0.0038672
2: 0.0307467, 0.0362073, 0.0309657, 0.0353238, -0.0045770, 0.0052416
3: -0.0030938, 0.0011206, -0.0030436, 0.0012175, -0.0043113, 0.0041643
4: -0.0049684, 0.0005672, -0.0050963, 0.0001745, -0.0051430, 0.0056636
5: 0.0094635, 0.0138936, 0.0097776, 0.0135098, -0.0040463, 0.0041160
6: -0.0063480, 0.0017485, -0.0051634, 0.0019333, -0.0082813, 0.0069119
7: 0.9722562, 0.9792826, 0.9731716, 0.9794121, -0.0071560, 0.0061110
8: -0.0158536, -0.0021896, -0.0151513, -0.0030657, -0.0127879, 0.0129618
9: -0.0027356, 0.0051402, -0.0022285, 0.0046263, -0.0073619, 0.0073687

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061164, upper bound: 0.0061755
time: 1.96 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061733, upper bound: 0.0062151
time: 1.96 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0038170, 0.0010087, -0.0035337, 0.0009059, -0.0047229, 0.0045424
1: -0.0059746, -0.0021090, -0.0060031, -0.0022321, -0.0037425, 0.0038941
2: 0.0305834, 0.0368662, 0.0307477, 0.0362035, -0.0056201, 0.0061185
3: -0.0031313, 0.0014261, -0.0030936, 0.0011189, -0.0042501, 0.0045198
4: -0.0049702, 0.0008601, -0.0049991, 0.0005655, -0.0055357, 0.0058592
5: 0.0092293, 0.0141798, 0.0094649, 0.0138920, -0.0046626, 0.0047149
6: -0.0072313, 0.0017510, -0.0063429, 0.0017928, -0.0090242, 0.0080939
7: 0.9715734, 0.9792844, 0.9722601, 0.9793137, -0.0077403, 0.0070243
8: -0.0163773, -0.0015362, -0.0158506, -0.0021933, -0.0141840, 0.0143143
9: -0.0031137, 0.0055235, -0.0027334, 0.0051380, -0.0082518, 0.0082569

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061924, upper bound: 0.0062198
time: 1.36 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061924, upper bound: 0.0062340
time: 1.95 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.00 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061253, upper bound: 0.0061512
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061767, upper bound: 0.0061837
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061985, upper bound: 0.0061904
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061985, upper bound: 0.0062015
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061245, upper bound: 0.0061514
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061767, upper bound: 0.0061851
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061985, upper bound: 0.0061911
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061985, upper bound: 0.0062033
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061578, upper bound: 0.0061457
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0062034, upper bound: 0.0061762
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0062277, upper bound: 0.0061826
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0062277, upper bound: 0.0061919
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061565, upper bound: 0.0061459
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0062034, upper bound: 0.0061775
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0062277, upper bound: 0.0061838
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0062277, upper bound: 0.0061938
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061177, upper bound: 0.0061750
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061707, upper bound: 0.0062138
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061891, upper bound: 0.0062187
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061892, upper bound: 0.0062314
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061164, upper bound: 0.0061755
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061707, upper bound: 0.0062147
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061892, upper bound: 0.0062190
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061892, upper bound: 0.0062327
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061177, upper bound: 0.0061750
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061734, upper bound: 0.0062142
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061923, upper bound: 0.0062196
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061924, upper bound: 0.0062327
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061164, upper bound: 0.0061755
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061733, upper bound: 0.0062151
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061924, upper bound: 0.0062198
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 7, lower bound: -0.0061924, upper bound: 0.0062340

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0023246, 0.0007073, -0.0025687, 0.0007183, -0.0030428, 0.0032760
1: -0.0057927, -0.0025158, -0.0058305, -0.0024878, -0.0026975, 0.0027506
2: 0.0314362, 0.0344393, 0.0313071, 0.0346667, -0.0032305, 0.0031322
3: -0.0029330, 0.0008631, -0.0029654, 0.0009070, -0.0038399, 0.0038285
4: -0.0047851, -0.0002794, -0.0048236, -0.0001587, -0.0046265, 0.0045443
5: 0.0104701, 0.0133138, 0.0102672, 0.0133518, -0.0028817, 0.0030466
6: -0.0045082, 0.0014835, -0.0046417, 0.0015392, -0.0060474, 0.0061252
7: 0.9743622, 0.9790973, 0.9740381, 0.9791363, -0.0047741, 0.0050592
8: -0.0139802, -0.0049975, -0.0143121, -0.0044313, -0.0095489, 0.0093146
9: -0.0011102, 0.0040000, -0.0014379, 0.0041684, -0.0052786, 0.0054380

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059018, upper bound: 0.0060315
time: 1.35 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0058991, upper bound: 0.0060190
time: 1.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0028778, 0.0007323, -0.0028939, 0.0007330, -0.0036108, 0.0036262
1: -0.0057016, -0.0024524, -0.0058336, -0.0024505, -0.0028455, 0.0029010
2: 0.0311279, 0.0349546, 0.0311186, 0.0349696, -0.0038417, 0.0038361
3: -0.0030065, 0.0007576, -0.0030086, 0.0009105, -0.0039170, 0.0037662
4: -0.0046925, -0.0000058, -0.0048268, 0.0000021, -0.0046946, 0.0048209
5: 0.0100102, 0.0133999, 0.0099968, 0.0134024, -0.0033922, 0.0034031
6: -0.0048107, 0.0013496, -0.0048195, 0.0015437, -0.0063544, 0.0061691
7: 0.9736277, 0.9790036, 0.9736063, 0.9791394, -0.0055118, 0.0053973
8: -0.0147324, -0.0037144, -0.0147543, -0.0036770, -0.0110554, 0.0110399
9: -0.0018529, 0.0043816, -0.0018745, 0.0043927, -0.0062456, 0.0062561

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061597, upper bound: 0.0061677
time: 2.13 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061597, upper bound: 0.0061675
time: 1.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0026877, 0.0007237, -0.0033245, 0.0008300, -0.0035177, 0.0040482
1: -0.0057992, -0.0024742, -0.0057387, -0.0023230, -0.0034761, 0.0030005
2: 0.0312381, 0.0347775, 0.0308689, 0.0357144, -0.0044762, 0.0039086
3: -0.0029812, 0.0008706, -0.0030658, 0.0008920, -0.0038732, 0.0039364
4: -0.0047917, -0.0000998, -0.0047302, 0.0003481, -0.0051399, 0.0046303
5: 0.0101683, 0.0133703, 0.0096388, 0.0136795, -0.0035112, 0.0037315
6: -0.0047067, 0.0014931, -0.0056870, 0.0014041, -0.0061108, 0.0071801
7: 0.9738801, 0.9791039, 0.9727671, 0.9790418, -0.0051616, 0.0063369
8: -0.0144739, -0.0041554, -0.0154618, -0.0026784, -0.0117955, 0.0113064
9: -0.0015976, 0.0042505, -0.0024526, 0.0048535, -0.0064511, 0.0067031

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061231, upper bound: 0.0061252
time: 1.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061767, upper bound: 0.0061761
time: 1.93 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0031220, 0.0007565, -0.0033245, 0.0008300, -0.0039520, 0.0040810
1: -0.0057040, -0.0024111, -0.0057387, -0.0023230, -0.0033810, 0.0033276
2: 0.0309863, 0.0352406, 0.0308689, 0.0357144, -0.0047280, 0.0043717
3: -0.0030389, 0.0007604, -0.0030658, 0.0008920, -0.0039310, 0.0038262
4: -0.0046950, 0.0001376, -0.0047302, 0.0003481, -0.0050431, 0.0048678
5: 0.0098072, 0.0134737, 0.0096388, 0.0136795, -0.0038723, 0.0038349
6: -0.0050519, 0.0013532, -0.0056870, 0.0014041, -0.0064560, 0.0070402
7: 0.9732578, 0.9790061, 0.9727671, 0.9790418, -0.0057839, 0.0062391
8: -0.0150852, -0.0031481, -0.0154618, -0.0026784, -0.0124068, 0.0123136
9: -0.0021807, 0.0045779, -0.0024526, 0.0048535, -0.0070342, 0.0070305

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061231, upper bound: 0.0061775
time: 1.82 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061767, upper bound: 0.0061876
time: 1.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0025075, 0.0007155, -0.0025805, 0.0007188, -0.0032263, 0.0032960
1: -0.0058026, -0.0024948, -0.0058339, -0.0024865, -0.0027501, 0.0027854
2: 0.0313426, 0.0346096, 0.0313003, 0.0346777, -0.0033351, 0.0033094
3: -0.0029573, 0.0008746, -0.0029670, 0.0009108, -0.0038681, 0.0038416
4: -0.0047953, -0.0001889, -0.0048270, -0.0001528, -0.0046425, 0.0046381
5: 0.0103181, 0.0133423, 0.0102573, 0.0133536, -0.0030356, 0.0030849
6: -0.0046082, 0.0014982, -0.0046481, 0.0015441, -0.0061523, 0.0061463
7: 0.9741194, 0.9791076, 0.9740223, 0.9791397, -0.0050203, 0.0050853
8: -0.0142289, -0.0045733, -0.0143282, -0.0044039, -0.0098250, 0.0097549
9: -0.0013558, 0.0041262, -0.0014538, 0.0041766, -0.0055323, 0.0055800

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0058797, upper bound: 0.0060250
time: 1.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0058778, upper bound: 0.0060119
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0030563, 0.0007403, -0.0029058, 0.0007335, -0.0037898, 0.0036461
1: -0.0057111, -0.0024319, -0.0058370, -0.0024492, -0.0029069, 0.0029364
2: 0.0310244, 0.0351209, 0.0311117, 0.0349807, -0.0039563, 0.0040092
3: -0.0030302, 0.0007686, -0.0030102, 0.0009144, -0.0039446, 0.0037788
4: -0.0047022, 0.0000824, -0.0048302, 0.0000080, -0.0047102, 0.0049126
5: 0.0098618, 0.0134276, 0.0099869, 0.0134042, -0.0035425, 0.0034407
6: -0.0049083, 0.0013636, -0.0048260, 0.0015487, -0.0064570, 0.0061896
7: 0.9733906, 0.9790134, 0.9735905, 0.9791428, -0.0057522, 0.0054229
8: -0.0149752, -0.0033005, -0.0147705, -0.0036495, -0.0113257, 0.0114700
9: -0.0020925, 0.0045047, -0.0018905, 0.0044009, -0.0064934, 0.0063952

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061597, upper bound: 0.0061694
time: 1.76 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061597, upper bound: 0.0061690
time: 2.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0028785, 0.0007323, -0.0033347, 0.0008337, -0.0037122, 0.0040670
1: -0.0058096, -0.0024523, -0.0057415, -0.0023186, -0.0034910, 0.0030348
2: 0.0311275, 0.0349553, 0.0308630, 0.0357382, -0.0046107, 0.0040923
3: -0.0030066, 0.0008826, -0.0030672, 0.0009031, -0.0039097, 0.0039498
4: -0.0048023, -0.0000055, -0.0047331, 0.0003587, -0.0051610, 0.0047276
5: 0.0100096, 0.0134000, 0.0096303, 0.0136898, -0.0036802, 0.0037697
6: -0.0048111, 0.0015083, -0.0057190, 0.0014083, -0.0062194, 0.0072273
7: 0.9736267, 0.9791147, 0.9727424, 0.9790447, -0.0054181, 0.0063723
8: -0.0147334, -0.0037128, -0.0154807, -0.0026548, -0.0120786, 0.0117679
9: -0.0018539, 0.0043821, -0.0024663, 0.0048673, -0.0067212, 0.0068484

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061224, upper bound: 0.0061245
time: 1.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061768, upper bound: 0.0061767
time: 2.06 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0032971, 0.0008200, -0.0033347, 0.0008337, -0.0041308, 0.0041547
1: -0.0057136, -0.0023350, -0.0057415, -0.0023186, -0.0033950, 0.0034066
2: 0.0308849, 0.0356500, 0.0308630, 0.0357382, -0.0048533, 0.0047870
3: -0.0030622, 0.0008622, -0.0030672, 0.0009031, -0.0039653, 0.0039294
4: -0.0047047, 0.0003195, -0.0047331, 0.0003587, -0.0050634, 0.0050527
5: 0.0096616, 0.0136515, 0.0096303, 0.0136898, -0.0040282, 0.0040212
6: -0.0056008, 0.0013673, -0.0057190, 0.0014083, -0.0070092, 0.0070863
7: 0.9728337, 0.9790161, 0.9727424, 0.9790447, -0.0062110, 0.0062737
8: -0.0154107, -0.0027422, -0.0154807, -0.0026548, -0.0127559, 0.0127385
9: -0.0024157, 0.0048161, -0.0024663, 0.0048673, -0.0072830, 0.0072824

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061224, upper bound: 0.0061789
time: 1.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061768, upper bound: 0.0061892
time: 2.08 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0023246, 0.0007073, -0.0027880, 0.0007282, -0.0030528, 0.0034953
1: -0.0057927, -0.0025158, -0.0060931, -0.0024627, -0.0027264, 0.0029678
2: 0.0314362, 0.0344393, 0.0311800, 0.0348710, -0.0034348, 0.0032593
3: -0.0029330, 0.0008631, -0.0029945, 0.0012112, -0.0041441, 0.0038576
4: -0.0047851, -0.0002794, -0.0050908, -0.0000502, -0.0047349, 0.0048114
5: 0.0104701, 0.0133138, 0.0100848, 0.0133859, -0.0029158, 0.0032290
6: -0.0045082, 0.0014835, -0.0047616, 0.0019253, -0.0064335, 0.0062451
7: 0.9743622, 0.9790973, 0.9737468, 0.9794064, -0.0050442, 0.0053505
8: -0.0139802, -0.0049975, -0.0146104, -0.0039226, -0.0100575, 0.0096129
9: -0.0011102, 0.0040000, -0.0017324, 0.0043197, -0.0054299, 0.0057324

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059245, upper bound: 0.0060789
time: 1.40 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059186, upper bound: 0.0059486
time: 1.55 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0028778, 0.0007323, -0.0031359, 0.0007615, -0.0036393, 0.0038681
1: -0.0057016, -0.0024524, -0.0060963, -0.0024050, -0.0032966, 0.0031257
2: 0.0311279, 0.0349546, 0.0309783, 0.0352731, -0.0041452, 0.0039763
3: -0.0030065, 0.0007576, -0.0030408, 0.0012149, -0.0042213, 0.0037983
4: -0.0046925, -0.0000058, -0.0050940, 0.0001520, -0.0048445, 0.0050882
5: 0.0100102, 0.0133999, 0.0097956, 0.0134878, -0.0034776, 0.0036043
6: -0.0048107, 0.0013496, -0.0050955, 0.0019300, -0.0067407, 0.0064451
7: 0.9736277, 0.9790036, 0.9732242, 0.9794096, -0.0057819, 0.0057794
8: -0.0147324, -0.0037144, -0.0151111, -0.0031159, -0.0116165, 0.0113967
9: -0.0018529, 0.0043816, -0.0021994, 0.0045968, -0.0064497, 0.0065810

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061712, upper bound: 0.0061552
time: 1.47 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061762, upper bound: 0.0061508
time: 1.47 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0026877, 0.0007237, -0.0035229, 0.0009020, -0.0035897, 0.0042466
1: -0.0057992, -0.0024742, -0.0060019, -0.0022368, -0.0035624, 0.0031772
2: 0.0312381, 0.0347775, 0.0307539, 0.0361783, -0.0049402, 0.0040236
3: -0.0029812, 0.0008706, -0.0030922, 0.0011072, -0.0040884, 0.0039628
4: -0.0047917, -0.0000998, -0.0049980, 0.0005544, -0.0053461, 0.0048981
5: 0.0101683, 0.0133703, 0.0094738, 0.0138810, -0.0037128, 0.0038965
6: -0.0047067, 0.0014931, -0.0063091, 0.0017911, -0.0064979, 0.0078022
7: 0.9738801, 0.9791039, 0.9722862, 0.9793125, -0.0054324, 0.0068177
8: -0.0144739, -0.0041554, -0.0158306, -0.0022183, -0.0122556, 0.0116752
9: -0.0015976, 0.0042505, -0.0027189, 0.0051234, -0.0067210, 0.0069694

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061459, upper bound: 0.0061171
time: 1.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062034, upper bound: 0.0061695
time: 1.97 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0031220, 0.0007565, -0.0035229, 0.0009020, -0.0040240, 0.0042794
1: -0.0057040, -0.0024111, -0.0060019, -0.0022368, -0.0034672, 0.0035908
2: 0.0309863, 0.0352406, 0.0307539, 0.0361783, -0.0051920, 0.0044867
3: -0.0030389, 0.0007604, -0.0030922, 0.0011072, -0.0041461, 0.0038526
4: -0.0046950, 0.0001376, -0.0049980, 0.0005544, -0.0052493, 0.0051355
5: 0.0098072, 0.0134737, 0.0094738, 0.0138810, -0.0040738, 0.0039998
6: -0.0050519, 0.0013532, -0.0063091, 0.0017911, -0.0068431, 0.0076623
7: 0.9732578, 0.9790061, 0.9722862, 0.9793125, -0.0060546, 0.0067199
8: -0.0150852, -0.0031481, -0.0158306, -0.0022183, -0.0128670, 0.0126825
9: -0.0021807, 0.0045779, -0.0027189, 0.0051234, -0.0073041, 0.0072969

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061459, upper bound: 0.0061674
time: 1.87 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062034, upper bound: 0.0061791
time: 1.92 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0025075, 0.0007155, -0.0028005, 0.0007288, -0.0032362, 0.0035161
1: -0.0058026, -0.0024948, -0.0060954, -0.0024612, -0.0027794, 0.0030024
2: 0.0313426, 0.0346096, 0.0311727, 0.0348827, -0.0035400, 0.0034369
3: -0.0029573, 0.0008746, -0.0029962, 0.0012138, -0.0041711, 0.0038708
4: -0.0047953, -0.0001889, -0.0050931, -0.0000440, -0.0047512, 0.0049042
5: 0.0103181, 0.0133423, 0.0100744, 0.0133878, -0.0030698, 0.0032679
6: -0.0046082, 0.0014982, -0.0047684, 0.0019286, -0.0065368, 0.0062666
7: 0.9741194, 0.9791076, 0.9737303, 0.9794088, -0.0052894, 0.0053773
8: -0.0142289, -0.0045733, -0.0146274, -0.0038936, -0.0103353, 0.0100541
9: -0.0013558, 0.0041262, -0.0017492, 0.0043283, -0.0056841, 0.0058754

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059242, upper bound: 0.0060790
time: 1.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059177, upper bound: 0.0059435
time: 1.44 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0030563, 0.0007403, -0.0031485, 0.0007661, -0.0038224, 0.0038888
1: -0.0057111, -0.0024319, -0.0060986, -0.0023996, -0.0033115, 0.0031609
2: 0.0310244, 0.0351209, 0.0309710, 0.0353026, -0.0042781, 0.0041499
3: -0.0030302, 0.0007686, -0.0030424, 0.0012175, -0.0042477, 0.0038110
4: -0.0047022, 0.0000824, -0.0050963, 0.0001651, -0.0048673, 0.0051787
5: 0.0098618, 0.0134276, 0.0097852, 0.0135006, -0.0036388, 0.0036425
6: -0.0049083, 0.0013636, -0.0051350, 0.0019333, -0.0068416, 0.0064986
7: 0.9733906, 0.9790134, 0.9731937, 0.9794120, -0.0060213, 0.0058197
8: -0.0149752, -0.0033005, -0.0151344, -0.0030867, -0.0118884, 0.0118340
9: -0.0020925, 0.0045047, -0.0022163, 0.0046140, -0.0067065, 0.0067210

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061712, upper bound: 0.0061566
time: 1.37 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061762, upper bound: 0.0061520
time: 1.31 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0028785, 0.0007323, -0.0035337, 0.0009059, -0.0037845, 0.0042660
1: -0.0058096, -0.0024523, -0.0060031, -0.0022321, -0.0035774, 0.0032107
2: 0.0311275, 0.0349553, 0.0307477, 0.0362035, -0.0050760, 0.0042076
3: -0.0030066, 0.0008826, -0.0030936, 0.0011189, -0.0041254, 0.0039763
4: -0.0048023, -0.0000055, -0.0049991, 0.0005655, -0.0053678, 0.0049937
5: 0.0100096, 0.0134000, 0.0094649, 0.0138920, -0.0038824, 0.0039351
6: -0.0048111, 0.0015083, -0.0063429, 0.0017928, -0.0066039, 0.0078513
7: 0.9736267, 0.9791147, 0.9722601, 0.9793137, -0.0056871, 0.0068546
8: -0.0147334, -0.0037128, -0.0158506, -0.0021933, -0.0125401, 0.0121378
9: -0.0018539, 0.0043821, -0.0027334, 0.0051380, -0.0069919, 0.0071155

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061453, upper bound: 0.0061164
time: 1.53 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062034, upper bound: 0.0061706
time: 1.88 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0032971, 0.0008200, -0.0035337, 0.0009059, -0.0042030, 0.0043537
1: -0.0057136, -0.0023350, -0.0060031, -0.0022321, -0.0034815, 0.0036681
2: 0.0308849, 0.0356500, 0.0307477, 0.0362035, -0.0053186, 0.0049024
3: -0.0030622, 0.0008622, -0.0030936, 0.0011189, -0.0041810, 0.0039558
4: -0.0047047, 0.0003195, -0.0049991, 0.0005655, -0.0052703, 0.0053187
5: 0.0096616, 0.0136515, 0.0094649, 0.0138920, -0.0042303, 0.0041866
6: -0.0056008, 0.0013673, -0.0063429, 0.0017928, -0.0073937, 0.0077102
7: 0.9728337, 0.9790161, 0.9722601, 0.9793137, -0.0064800, 0.0067559
8: -0.0154107, -0.0027422, -0.0158506, -0.0021933, -0.0132174, 0.0131084
9: -0.0024157, 0.0048161, -0.0027334, 0.0051380, -0.0075537, 0.0075495

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061453, upper bound: 0.0061689
time: 1.55 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062034, upper bound: 0.0061808
time: 1.93 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0024836, 0.0007144, -0.0025687, 0.0007183, -0.0032019, 0.0032831
1: -0.0060663, -0.0024976, -0.0058305, -0.0024878, -0.0029163, 0.0027772
2: 0.0312665, 0.0345874, 0.0313071, 0.0346667, -0.0034002, 0.0032803
3: -0.0029541, 0.0011800, -0.0029654, 0.0009070, -0.0038611, 0.0041454
4: -0.0050634, -0.0002007, -0.0048236, -0.0001587, -0.0049047, 0.0046229
5: 0.0103379, 0.0133385, 0.0102672, 0.0133518, -0.0030139, 0.0030714
6: -0.0045951, 0.0018858, -0.0046417, 0.0015392, -0.0061343, 0.0065275
7: 0.9741510, 0.9793788, 0.9740381, 0.9791363, -0.0049853, 0.0053406
8: -0.0141964, -0.0046287, -0.0143121, -0.0044313, -0.0097651, 0.0096835
9: -0.0013237, 0.0041097, -0.0014379, 0.0041684, -0.0054921, 0.0055476

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059013, upper bound: 0.0060627
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0058984, upper bound: 0.0060461
time: 1.42 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0030452, 0.0007398, -0.0028939, 0.0007330, -0.0037782, 0.0036338
1: -0.0059659, -0.0024332, -0.0058336, -0.0024505, -0.0030298, 0.0029341
2: 0.0310309, 0.0351106, 0.0311186, 0.0349696, -0.0039388, 0.0039920
3: -0.0030287, 0.0010637, -0.0030086, 0.0009105, -0.0039392, 0.0040724
4: -0.0049613, 0.0000769, -0.0048268, 0.0000021, -0.0049635, 0.0049037
5: 0.0098710, 0.0134259, 0.0099968, 0.0134024, -0.0035314, 0.0034291
6: -0.0049023, 0.0017382, -0.0048195, 0.0015437, -0.0064460, 0.0065577
7: 0.9734054, 0.9792755, 0.9736063, 0.9791394, -0.0057341, 0.0056692
8: -0.0149601, -0.0033261, -0.0147543, -0.0036770, -0.0112831, 0.0114282
9: -0.0020777, 0.0044971, -0.0018745, 0.0043927, -0.0064704, 0.0063716

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061556, upper bound: 0.0062017
time: 2.10 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061555, upper bound: 0.0061996
time: 1.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0029214, 0.0007342, -0.0033245, 0.0008300, -0.0037514, 0.0040587
1: -0.0060629, -0.0024474, -0.0057387, -0.0023230, -0.0037399, 0.0030265
2: 0.0311026, 0.0349953, 0.0308689, 0.0357144, -0.0046117, 0.0041263
3: -0.0030123, 0.0011761, -0.0030658, 0.0008920, -0.0039043, 0.0042420
4: -0.0050600, 0.0000157, -0.0047302, 0.0003481, -0.0054081, 0.0047459
5: 0.0099739, 0.0134067, 0.0096388, 0.0136795, -0.0037055, 0.0037679
6: -0.0048346, 0.0018808, -0.0056870, 0.0014041, -0.0062387, 0.0075678
7: 0.9735698, 0.9793753, 0.9727671, 0.9790418, -0.0054719, 0.0066082
8: -0.0147918, -0.0036133, -0.0154618, -0.0026784, -0.0121133, 0.0118485
9: -0.0019115, 0.0044117, -0.0024526, 0.0048535, -0.0067649, 0.0068643

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061211, upper bound: 0.0061574
time: 1.40 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061707, upper bound: 0.0062032
time: 2.06 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0033317, 0.0008326, -0.0033245, 0.0008300, -0.0041617, 0.0041571
1: -0.0059683, -0.0023199, -0.0057387, -0.0023230, -0.0036453, 0.0034188
2: 0.0308647, 0.0357312, 0.0308689, 0.0357144, -0.0048496, 0.0048622
3: -0.0030668, 0.0010665, -0.0030658, 0.0008920, -0.0039588, 0.0041324
4: -0.0049638, 0.0003556, -0.0047302, 0.0003481, -0.0053119, 0.0050858
5: 0.0096328, 0.0136868, 0.0096388, 0.0136795, -0.0040467, 0.0040480
6: -0.0057097, 0.0017417, -0.0056870, 0.0014041, -0.0071138, 0.0074288
7: 0.9727495, 0.9792778, 0.9727671, 0.9790418, -0.0062922, 0.0065108
8: -0.0154752, -0.0026617, -0.0154618, -0.0026784, -0.0127968, 0.0128001
9: -0.0024623, 0.0048633, -0.0024526, 0.0048535, -0.0073158, 0.0073159

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061211, upper bound: 0.0062074
time: 1.54 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061707, upper bound: 0.0062163
time: 2.05 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0026853, 0.0007236, -0.0025805, 0.0007188, -0.0034042, 0.0033041
1: -0.0060722, -0.0024744, -0.0058339, -0.0024865, -0.0029646, 0.0028144
2: 0.0312395, 0.0347753, 0.0313003, 0.0346777, -0.0034382, 0.0034751
3: -0.0029809, 0.0011869, -0.0029670, 0.0009108, -0.0038917, 0.0041539
4: -0.0050695, -0.0001010, -0.0048270, -0.0001528, -0.0049166, 0.0047260
5: 0.0101702, 0.0133699, 0.0102573, 0.0133536, -0.0031834, 0.0031126
6: -0.0047054, 0.0018945, -0.0046481, 0.0015441, -0.0062496, 0.0065426
7: 0.9738832, 0.9793848, 0.9740223, 0.9791397, -0.0052565, 0.0053626
8: -0.0144707, -0.0041608, -0.0143282, -0.0044039, -0.0100668, 0.0101674
9: -0.0015945, 0.0042489, -0.0014538, 0.0041766, -0.0057711, 0.0057027

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0058793, upper bound: 0.0060571
time: 1.40 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0058765, upper bound: 0.0060373
time: 1.80 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0032191, 0.0007917, -0.0029058, 0.0007335, -0.0039526, 0.0036975
1: -0.0059713, -0.0023689, -0.0058370, -0.0024492, -0.0030832, 0.0034681
2: 0.0309301, 0.0354677, 0.0311117, 0.0349807, -0.0040507, 0.0043561
3: -0.0030518, 0.0010701, -0.0030102, 0.0009144, -0.0039662, 0.0040803
4: -0.0049668, 0.0002385, -0.0048302, 0.0000080, -0.0049748, 0.0050687
5: 0.0097264, 0.0135723, 0.0099869, 0.0134042, -0.0036778, 0.0035854
6: -0.0053564, 0.0017462, -0.0048260, 0.0015487, -0.0069051, 0.0065722
7: 0.9730226, 0.9792811, 0.9735905, 0.9791428, -0.0061203, 0.0056906
8: -0.0152657, -0.0029230, -0.0147705, -0.0036495, -0.0116162, 0.0118476
9: -0.0023110, 0.0047100, -0.0018905, 0.0044009, -0.0067120, 0.0066005

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061556, upper bound: 0.0062027
time: 1.80 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061555, upper bound: 0.0062007
time: 2.14 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0031174, 0.0007548, -0.0033347, 0.0008337, -0.0039511, 0.0040895
1: -0.0060706, -0.0024131, -0.0057415, -0.0023186, -0.0037520, 0.0033285
2: 0.0309890, 0.0352298, 0.0308630, 0.0357382, -0.0047492, 0.0043668
3: -0.0030383, 0.0011851, -0.0030672, 0.0009031, -0.0039414, 0.0042522
4: -0.0050678, 0.0001328, -0.0047331, 0.0003587, -0.0054265, 0.0048659
5: 0.0098110, 0.0134690, 0.0096303, 0.0136898, -0.0038788, 0.0038387
6: -0.0050375, 0.0018921, -0.0057190, 0.0014083, -0.0064458, 0.0076111
7: 0.9732691, 0.9793832, 0.9727424, 0.9790447, -0.0057756, 0.0066409
8: -0.0150767, -0.0031588, -0.0154807, -0.0026548, -0.0124219, 0.0123219
9: -0.0021745, 0.0045717, -0.0024663, 0.0048673, -0.0070419, 0.0070379

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061204, upper bound: 0.0061565
time: 1.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061707, upper bound: 0.0062034
time: 2.17 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0034918, 0.0008907, -0.0033347, 0.0008337, -0.0043255, 0.0042254
1: -0.0059738, -0.0022503, -0.0057415, -0.0023186, -0.0036552, 0.0034912
2: 0.0307720, 0.0361055, 0.0308630, 0.0357382, -0.0049662, 0.0052425
3: -0.0030880, 0.0010734, -0.0030672, 0.0009031, -0.0039911, 0.0041406
4: -0.0049694, 0.0005220, -0.0047331, 0.0003587, -0.0053281, 0.0052551
5: 0.0094997, 0.0138494, 0.0096303, 0.0136898, -0.0041901, 0.0042191
6: -0.0062115, 0.0017498, -0.0057190, 0.0014083, -0.0076199, 0.0074688
7: 0.9723616, 0.9792837, 0.9727424, 0.9790447, -0.0066832, 0.0065413
8: -0.0157727, -0.0022905, -0.0154807, -0.0026548, -0.0131179, 0.0131902
9: -0.0026772, 0.0050810, -0.0024663, 0.0048673, -0.0075445, 0.0075473

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061204, upper bound: 0.0062084
time: 1.76 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061707, upper bound: 0.0062174
time: 2.04 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0024836, 0.0007144, -0.0027880, 0.0007282, -0.0032118, 0.0035025
1: -0.0060663, -0.0024976, -0.0060931, -0.0024627, -0.0028505, 0.0029073
2: 0.0312665, 0.0345874, 0.0311800, 0.0348710, -0.0036045, 0.0034075
3: -0.0029541, 0.0011800, -0.0029945, 0.0012112, -0.0041653, 0.0041746
4: -0.0050634, -0.0002007, -0.0050908, -0.0000502, -0.0050132, 0.0048900
5: 0.0103379, 0.0133385, 0.0100848, 0.0133859, -0.0030480, 0.0032537
6: -0.0045951, 0.0018858, -0.0047616, 0.0019253, -0.0065204, 0.0066474
7: 0.9741510, 0.9793788, 0.9737468, 0.9794064, -0.0052553, 0.0056319
8: -0.0141964, -0.0046287, -0.0146104, -0.0039226, -0.0102738, 0.0099817
9: -0.0013237, 0.0041097, -0.0017324, 0.0043197, -0.0056434, 0.0058421

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0058840, upper bound: 0.0061098
time: 1.42 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0058684, upper bound: 0.0059719
time: 1.41 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0030452, 0.0007398, -0.0031359, 0.0007615, -0.0038068, 0.0038757
1: -0.0059659, -0.0024332, -0.0060963, -0.0024050, -0.0035608, 0.0030679
2: 0.0310309, 0.0351106, 0.0309783, 0.0352731, -0.0042423, 0.0041323
3: -0.0030287, 0.0010637, -0.0030408, 0.0012149, -0.0042436, 0.0041045
4: -0.0049613, 0.0000769, -0.0050940, 0.0001520, -0.0051133, 0.0051710
5: 0.0098710, 0.0134259, 0.0097956, 0.0134878, -0.0036168, 0.0036303
6: -0.0049023, 0.0017382, -0.0050955, 0.0019300, -0.0068322, 0.0068337
7: 0.9734054, 0.9792755, 0.9732242, 0.9794096, -0.0060043, 0.0060514
8: -0.0149601, -0.0033261, -0.0151111, -0.0031159, -0.0118442, 0.0117850
9: -0.0020777, 0.0044971, -0.0021994, 0.0045968, -0.0066745, 0.0066965

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061430, upper bound: 0.0061937
time: 1.87 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061456, upper bound: 0.0061911
time: 1.93 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0029214, 0.0007342, -0.0035229, 0.0009020, -0.0038234, 0.0042572
1: -0.0060629, -0.0024474, -0.0060019, -0.0022368, -0.0038261, 0.0031505
2: 0.0311026, 0.0349953, 0.0307539, 0.0361783, -0.0050757, 0.0042413
3: -0.0030123, 0.0011761, -0.0030922, 0.0011072, -0.0041195, 0.0042683
4: -0.0050600, 0.0000157, -0.0049980, 0.0005544, -0.0056144, 0.0050137
5: 0.0099739, 0.0134067, 0.0094738, 0.0138810, -0.0039071, 0.0039328
6: -0.0048346, 0.0018808, -0.0063091, 0.0017911, -0.0066257, 0.0081899
7: 0.9735698, 0.9793753, 0.9722862, 0.9793125, -0.0057427, 0.0070891
8: -0.0147918, -0.0036133, -0.0158306, -0.0022183, -0.0125735, 0.0122173
9: -0.0019115, 0.0044117, -0.0027189, 0.0051234, -0.0070348, 0.0071306

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061212, upper bound: 0.0061574
time: 1.94 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061734, upper bound: 0.0062041
time: 2.11 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0033317, 0.0008326, -0.0035229, 0.0009020, -0.0042337, 0.0043556
1: -0.0059683, -0.0023199, -0.0060019, -0.0022368, -0.0037315, 0.0036820
2: 0.0308647, 0.0357312, 0.0307539, 0.0361783, -0.0053136, 0.0049773
3: -0.0030668, 0.0010665, -0.0030922, 0.0011072, -0.0041740, 0.0041587
4: -0.0049638, 0.0003556, -0.0049980, 0.0005544, -0.0055181, 0.0053536
5: 0.0096328, 0.0136868, 0.0094738, 0.0138810, -0.0042482, 0.0042129
6: -0.0057097, 0.0017417, -0.0063091, 0.0017911, -0.0075008, 0.0080508
7: 0.9727495, 0.9792778, 0.9722862, 0.9793125, -0.0065629, 0.0069916
8: -0.0154752, -0.0026617, -0.0158306, -0.0022183, -0.0132569, 0.0131689
9: -0.0024623, 0.0048633, -0.0027189, 0.0051234, -0.0075857, 0.0075822

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061212, upper bound: 0.0062076
time: 1.76 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061734, upper bound: 0.0062176
time: 1.56 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0026853, 0.0007236, -0.0028005, 0.0007288, -0.0034141, 0.0035241
1: -0.0060722, -0.0024744, -0.0060954, -0.0024612, -0.0029048, 0.0029464
2: 0.0312395, 0.0347753, 0.0311727, 0.0348827, -0.0036432, 0.0036026
3: -0.0029809, 0.0011869, -0.0029962, 0.0012138, -0.0041947, 0.0041831
4: -0.0050695, -0.0001010, -0.0050931, -0.0000440, -0.0050254, 0.0049921
5: 0.0101702, 0.0133699, 0.0100744, 0.0133878, -0.0032176, 0.0032955
6: -0.0047054, 0.0018945, -0.0047684, 0.0019286, -0.0066341, 0.0066630
7: 0.9738832, 0.9793848, 0.9737303, 0.9794088, -0.0055256, 0.0056546
8: -0.0144707, -0.0041608, -0.0146274, -0.0038936, -0.0105771, 0.0104666
9: -0.0015945, 0.0042489, -0.0017492, 0.0043283, -0.0059228, 0.0059980

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0058828, upper bound: 0.0061104
time: 1.45 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0058660, upper bound: 0.0059660
time: 1.44 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0032191, 0.0007917, -0.0031485, 0.0007661, -0.0039852, 0.0039402
1: -0.0059713, -0.0023689, -0.0060986, -0.0023996, -0.0035718, 0.0037297
2: 0.0309301, 0.0354677, 0.0309710, 0.0353026, -0.0043725, 0.0044967
3: -0.0030518, 0.0010701, -0.0030424, 0.0012175, -0.0042693, 0.0041125
4: -0.0049668, 0.0002385, -0.0050963, 0.0001651, -0.0051319, 0.0053348
5: 0.0097264, 0.0135723, 0.0097852, 0.0135006, -0.0037741, 0.0037872
6: -0.0053564, 0.0017462, -0.0051350, 0.0019333, -0.0072897, 0.0068812
7: 0.9730226, 0.9792811, 0.9731937, 0.9794120, -0.0063894, 0.0060874
8: -0.0152657, -0.0029230, -0.0151344, -0.0030867, -0.0121790, 0.0122115
9: -0.0023110, 0.0047100, -0.0022163, 0.0046140, -0.0069250, 0.0069263

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061430, upper bound: 0.0061949
time: 1.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061456, upper bound: 0.0061920
time: 2.00 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0031174, 0.0007548, -0.0035337, 0.0009059, -0.0040233, 0.0042885
1: -0.0060706, -0.0024131, -0.0060031, -0.0022321, -0.0038385, 0.0035900
2: 0.0309890, 0.0352298, 0.0307477, 0.0362035, -0.0052145, 0.0044822
3: -0.0030383, 0.0011851, -0.0030936, 0.0011189, -0.0041572, 0.0042787
4: -0.0050678, 0.0001328, -0.0049991, 0.0005655, -0.0056334, 0.0051319
5: 0.0098110, 0.0134690, 0.0094649, 0.0138920, -0.0040810, 0.0040041
6: -0.0050375, 0.0018921, -0.0063429, 0.0017928, -0.0068303, 0.0082350
7: 0.9732691, 0.9793832, 0.9722601, 0.9793137, -0.0060446, 0.0071231
8: -0.0150767, -0.0031588, -0.0158506, -0.0021933, -0.0128833, 0.0126918
9: -0.0021745, 0.0045717, -0.0027334, 0.0051380, -0.0073125, 0.0073051

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061206, upper bound: 0.0061565
time: 1.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061734, upper bound: 0.0062042
time: 2.20 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0034918, 0.0008907, -0.0035337, 0.0009059, -0.0043977, 0.0044244
1: -0.0059738, -0.0022503, -0.0060031, -0.0022321, -0.0037417, 0.0037527
2: 0.0307720, 0.0361055, 0.0307477, 0.0362035, -0.0054316, 0.0053579
3: -0.0030880, 0.0010734, -0.0030936, 0.0011189, -0.0042069, 0.0041671
4: -0.0049694, 0.0005220, -0.0049991, 0.0005655, -0.0055349, 0.0055211
5: 0.0094997, 0.0138494, 0.0094649, 0.0138920, -0.0043923, 0.0043845
6: -0.0062115, 0.0017498, -0.0063429, 0.0017928, -0.0080044, 0.0080928
7: 0.9723616, 0.9792837, 0.9722601, 0.9793137, -0.0069522, 0.0070236
8: -0.0157727, -0.0022905, -0.0158506, -0.0021933, -0.0135794, 0.0135601
9: -0.0026772, 0.0050810, -0.0027334, 0.0051380, -0.0078152, 0.0078144

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061206, upper bound: 0.0062085
time: 1.55 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061734, upper bound: 0.0062188
time: 2.05 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.52 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0059018, upper bound: 0.0060315
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0058991, upper bound: 0.0060190
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061597, upper bound: 0.0061677
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061597, upper bound: 0.0061675
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061231, upper bound: 0.0061252
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061767, upper bound: 0.0061761
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061231, upper bound: 0.0061775
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061767, upper bound: 0.0061876
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0058797, upper bound: 0.0060250
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0058778, upper bound: 0.0060119
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061597, upper bound: 0.0061694
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061597, upper bound: 0.0061690
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061224, upper bound: 0.0061245
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061768, upper bound: 0.0061767
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061224, upper bound: 0.0061789
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061768, upper bound: 0.0061892
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0059245, upper bound: 0.0060789
NS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0059186, upper bound: 0.0059486
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061712, upper bound: 0.0061552
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061762, upper bound: 0.0061508
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061459, upper bound: 0.0061171
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0062034, upper bound: 0.0061695
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061459, upper bound: 0.0061674
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0062034, upper bound: 0.0061791
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0059242, upper bound: 0.0060790
NS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0059177, upper bound: 0.0059435
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061712, upper bound: 0.0061566
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061762, upper bound: 0.0061520
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061453, upper bound: 0.0061164
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0062034, upper bound: 0.0061706
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061453, upper bound: 0.0061689
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0062034, upper bound: 0.0061808
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0059013, upper bound: 0.0060627
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0058984, upper bound: 0.0060461
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061556, upper bound: 0.0062017
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061555, upper bound: 0.0061996
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061211, upper bound: 0.0061574
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061707, upper bound: 0.0062032
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061211, upper bound: 0.0062074
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061707, upper bound: 0.0062163
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0058793, upper bound: 0.0060571
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0058765, upper bound: 0.0060373
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061556, upper bound: 0.0062027
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061555, upper bound: 0.0062007
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061204, upper bound: 0.0061565
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061707, upper bound: 0.0062034
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061204, upper bound: 0.0062084
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061707, upper bound: 0.0062174
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0058840, upper bound: 0.0061098
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0058684, upper bound: 0.0059719
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061430, upper bound: 0.0061937
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061456, upper bound: 0.0061911
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061212, upper bound: 0.0061574
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061734, upper bound: 0.0062041
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061212, upper bound: 0.0062076
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061734, upper bound: 0.0062176
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0058828, upper bound: 0.0061104
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0058660, upper bound: 0.0059660
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061430, upper bound: 0.0061949
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061456, upper bound: 0.0061920
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061206, upper bound: 0.0061565
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061734, upper bound: 0.0062042
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061206, upper bound: 0.0062085
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.52
Output dim: 7, lower bound: -0.0061734, upper bound: 0.0062188

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0022119, 0.0007022, -0.0022173, 0.0007024, -0.0029143, 0.0029195
1: -0.0057925, -0.0025287, -0.0058300, -0.0025281, -0.0026263, 0.0026470
2: 0.0314363, 0.0343343, 0.0314131, 0.0343394, -0.0029030, 0.0029213
3: -0.0029180, 0.0008629, -0.0029187, 0.0009063, -0.0038243, 0.0037816
4: -0.0047850, -0.0003351, -0.0048231, -0.0003324, -0.0044526, 0.0044880
5: 0.0105638, 0.0132963, 0.0105593, 0.0132971, -0.0027333, 0.0027370
6: -0.0044466, 0.0014833, -0.0044495, 0.0015384, -0.0059850, 0.0059328
7: 0.9745117, 0.9790971, 0.9745045, 0.9791357, -0.0046239, 0.0045927
8: -0.0138270, -0.0052588, -0.0138343, -0.0052462, -0.0085807, 0.0085755
9: -0.0009589, 0.0039223, -0.0009662, 0.0039260, -0.0048850, 0.0048885

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0056196, upper bound: 0.0056115
time: 1.37 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055260, upper bound: 0.0056040
time: 1.40 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0021006, 0.0006971, -0.0020963, 0.0006969, -0.0027976, 0.0027934
1: -0.0057923, -0.0025415, -0.0058594, -0.0025420, -0.0025954, 0.0026491
2: 0.0314364, 0.0342306, 0.0313948, 0.0342266, -0.0027902, 0.0028358
3: -0.0029032, 0.0008627, -0.0029026, 0.0009405, -0.0038437, 0.0037653
4: -0.0047848, -0.0003901, -0.0048531, -0.0003923, -0.0043925, 0.0044630
5: 0.0106563, 0.0132790, 0.0106600, 0.0132783, -0.0026220, 0.0026190
6: -0.0043857, 0.0014830, -0.0043833, 0.0015817, -0.0059674, 0.0058664
7: 0.9746596, 0.9790969, 0.9746654, 0.9791661, -0.0045065, 0.0044315
8: -0.0136756, -0.0055168, -0.0136697, -0.0055270, -0.0081486, 0.0081528
9: -0.0008096, 0.0038455, -0.0008037, 0.0038425, -0.0046521, 0.0046492

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0056103, upper bound: 0.0056010
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055207, upper bound: 0.0055938
time: 1.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0027549, 0.0007267, -0.0025374, 0.0007169, -0.0034718, 0.0032641
1: -0.0057014, -0.0024665, -0.0058331, -0.0024914, -0.0027629, 0.0027804
2: 0.0311992, 0.0348401, 0.0313253, 0.0346375, -0.0034384, 0.0035149
3: -0.0029901, 0.0007574, -0.0029612, 0.0009099, -0.0039000, 0.0037186
4: -0.0046923, -0.0000666, -0.0048262, -0.0001741, -0.0045182, 0.0047596
5: 0.0101124, 0.0133808, 0.0102932, 0.0133469, -0.0032345, 0.0030876
6: -0.0047435, 0.0013494, -0.0046246, 0.0015429, -0.0062864, 0.0059739
7: 0.9737909, 0.9790034, 0.9740797, 0.9791389, -0.0053480, 0.0049237
8: -0.0145653, -0.0039995, -0.0142696, -0.0045039, -0.0100614, 0.0102700
9: -0.0016879, 0.0042968, -0.0013959, 0.0041468, -0.0058347, 0.0056928

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061294, upper bound: 0.0061202
time: 1.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061092, upper bound: 0.0061221
time: 1.89 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0026705, 0.0007229, -0.0024285, 0.0007119, -0.0033824, 0.0031514
1: -0.0057012, -0.0024762, -0.0058625, -0.0025039, -0.0027315, 0.0027760
2: 0.0312481, 0.0347615, 0.0313884, 0.0345361, -0.0032879, 0.0033731
3: -0.0029789, 0.0007572, -0.0029468, 0.0009440, -0.0039229, 0.0037039
4: -0.0046921, -0.0001084, -0.0048562, -0.0002280, -0.0044641, 0.0047478
5: 0.0101826, 0.0133676, 0.0103838, 0.0133300, -0.0031474, 0.0029839
6: -0.0046973, 0.0013491, -0.0045650, 0.0015862, -0.0062835, 0.0059141
7: 0.9739030, 0.9790034, 0.9742242, 0.9791692, -0.0052662, 0.0047792
8: -0.0144505, -0.0041953, -0.0141215, -0.0047565, -0.0096940, 0.0099261
9: -0.0015746, 0.0042386, -0.0012497, 0.0040717, -0.0056462, 0.0054883

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061287, upper bound: 0.0061182
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061069, upper bound: 0.0061195
time: 2.02 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0023479, 0.0007083, -0.0024984, 0.0007151, -0.0030631, 0.0032067
1: -0.0057960, -0.0025131, -0.0058283, -0.0024959, -0.0027008, 0.0027358
2: 0.0314341, 0.0344610, 0.0313479, 0.0346012, -0.0031671, 0.0031132
3: -0.0029361, 0.0008670, -0.0029561, 0.0009044, -0.0038405, 0.0038231
4: -0.0047886, -0.0002678, -0.0048214, -0.0001934, -0.0045951, 0.0045536
5: 0.0104507, 0.0133174, 0.0103256, 0.0133409, -0.0028901, 0.0029918
6: -0.0045210, 0.0014885, -0.0046032, 0.0015359, -0.0060569, 0.0060917
7: 0.9743312, 0.9791008, 0.9741315, 0.9791340, -0.0048028, 0.0049694
8: -0.0140119, -0.0049433, -0.0142166, -0.0045943, -0.0094176, 0.0092733
9: -0.0011416, 0.0040161, -0.0013436, 0.0041199, -0.0052615, 0.0053597

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060396, upper bound: 0.0058874
time: 1.59 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060255, upper bound: 0.0058841
time: 1.50 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0026797, 0.0007233, -0.0030390, 0.0007395, -0.0034193, 0.0037623
1: -0.0057991, -0.0024751, -0.0057371, -0.0024339, -0.0028463, 0.0028886
2: 0.0312428, 0.0347701, 0.0310345, 0.0351047, -0.0038620, 0.0037356
3: -0.0029801, 0.0008706, -0.0030279, 0.0007988, -0.0037789, 0.0038984
4: -0.0047917, -0.0001038, -0.0047286, 0.0000738, -0.0048655, 0.0046249
5: 0.0101749, 0.0133691, 0.0098762, 0.0134250, -0.0032501, 0.0034929
6: -0.0047024, 0.0014930, -0.0048988, 0.0014019, -0.0061043, 0.0063918
7: 0.9738907, 0.9791039, 0.9734138, 0.9790402, -0.0051495, 0.0056902
8: -0.0144631, -0.0041739, -0.0149516, -0.0033407, -0.0111224, 0.0107777
9: -0.0015870, 0.0042450, -0.0020692, 0.0044928, -0.0060797, 0.0063142

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061693, upper bound: 0.0061592
time: 1.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061689, upper bound: 0.0061591
time: 2.04 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0027942, 0.0007285, -0.0024984, 0.0007151, -0.0035093, 0.0032269
1: -0.0057010, -0.0024620, -0.0058283, -0.0024959, -0.0028053, 0.0028187
2: 0.0311764, 0.0348767, 0.0313479, 0.0346012, -0.0034248, 0.0035288
3: -0.0029954, 0.0007569, -0.0029561, 0.0009044, -0.0038998, 0.0037129
4: -0.0046919, -0.0000472, -0.0048214, -0.0001934, -0.0044984, 0.0047742
5: 0.0100797, 0.0133869, 0.0103256, 0.0133409, -0.0032611, 0.0030612
6: -0.0047650, 0.0013487, -0.0046032, 0.0015359, -0.0063009, 0.0059519
7: 0.9737387, 0.9790031, 0.9741315, 0.9791340, -0.0053952, 0.0048716
8: -0.0146187, -0.0039084, -0.0142166, -0.0045943, -0.0100244, 0.0103082
9: -0.0017406, 0.0043239, -0.0013436, 0.0041199, -0.0058606, 0.0056675

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061488, upper bound: 0.0061519
time: 2.11 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061480, upper bound: 0.0061517
time: 1.47 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0031138, 0.0007535, -0.0030390, 0.0007395, -0.0038534, 0.0037925
1: -0.0057040, -0.0024146, -0.0057371, -0.0024339, -0.0029580, 0.0033225
2: 0.0309911, 0.0352216, 0.0310345, 0.0351047, -0.0041137, 0.0041871
3: -0.0030378, 0.0007604, -0.0030279, 0.0007988, -0.0038366, 0.0037883
4: -0.0046949, 0.0001291, -0.0047286, 0.0000738, -0.0047688, 0.0048578
5: 0.0098140, 0.0134654, 0.0098762, 0.0134250, -0.0036110, 0.0035892
6: -0.0050264, 0.0013532, -0.0048988, 0.0014019, -0.0064282, 0.0062520
7: 0.9732777, 0.9790061, 0.9734138, 0.9790402, -0.0057625, 0.0055923
8: -0.0150701, -0.0031670, -0.0149516, -0.0033407, -0.0117294, 0.0117845
9: -0.0021698, 0.0045668, -0.0020692, 0.0044928, -0.0066625, 0.0066361

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061720, upper bound: 0.0061755
time: 1.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061715, upper bound: 0.0061754
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0023937, 0.0007104, -0.0022290, 0.0007029, -0.0030967, 0.0029393
1: -0.0058025, -0.0025079, -0.0058333, -0.0025268, -0.0026766, 0.0026827
2: 0.0314085, 0.0345037, 0.0314110, 0.0343502, -0.0029417, 0.0030927
3: -0.0029421, 0.0008744, -0.0029203, 0.0009102, -0.0038524, 0.0037947
4: -0.0047951, -0.0002452, -0.0048265, -0.0003266, -0.0044685, 0.0045814
5: 0.0104126, 0.0133246, 0.0105496, 0.0132989, -0.0028863, 0.0027749
6: -0.0045460, 0.0014979, -0.0044559, 0.0015433, -0.0060893, 0.0059538
7: 0.9742704, 0.9791073, 0.9744892, 0.9791392, -0.0048688, 0.0046181
8: -0.0140742, -0.0048371, -0.0138501, -0.0052193, -0.0088550, 0.0090131
9: -0.0012031, 0.0040477, -0.0009818, 0.0039341, -0.0051372, 0.0050296

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055958, upper bound: 0.0056021
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055165, upper bound: 0.0055973
time: 1.39 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0022849, 0.0007055, -0.0021083, 0.0006975, -0.0029824, 0.0028138
1: -0.0058023, -0.0025204, -0.0058625, -0.0025406, -0.0026446, 0.0026822
2: 0.0314302, 0.0344023, 0.0313929, 0.0342379, -0.0028076, 0.0030093
3: -0.0029277, 0.0008743, -0.0029042, 0.0009440, -0.0038716, 0.0037785
4: -0.0047949, -0.0002990, -0.0048561, -0.0003863, -0.0044086, 0.0045571
5: 0.0105031, 0.0133076, 0.0106499, 0.0132802, -0.0027770, 0.0026577
6: -0.0044865, 0.0014977, -0.0043899, 0.0015862, -0.0060726, 0.0058876
7: 0.9744149, 0.9791072, 0.9746493, 0.9791691, -0.0047542, 0.0044580
8: -0.0139262, -0.0050895, -0.0136862, -0.0054990, -0.0084272, 0.0085966
9: -0.0010569, 0.0039726, -0.0008199, 0.0038509, -0.0049078, 0.0047925

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055860, upper bound: 0.0055920
time: 1.42 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055100, upper bound: 0.0055871
time: 1.37 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0029360, 0.0007349, -0.0025491, 0.0007174, -0.0036534, 0.0032840
1: -0.0057109, -0.0024457, -0.0058364, -0.0024901, -0.0028232, 0.0028168
2: 0.0310942, 0.0350089, 0.0313185, 0.0346484, -0.0035543, 0.0036904
3: -0.0030142, 0.0007684, -0.0029628, 0.0009138, -0.0039280, 0.0037312
4: -0.0047020, 0.0000230, -0.0048297, -0.0001684, -0.0045336, 0.0048526
5: 0.0099618, 0.0134089, 0.0102835, 0.0133487, -0.0033870, 0.0031255
6: -0.0048425, 0.0013633, -0.0046310, 0.0015479, -0.0063904, 0.0059943
7: 0.9735504, 0.9790131, 0.9740641, 0.9791423, -0.0055919, 0.0049490
8: -0.0148116, -0.0035794, -0.0142855, -0.0044768, -0.0103348, 0.0107061
9: -0.0019311, 0.0044218, -0.0014116, 0.0041549, -0.0060860, 0.0058334

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061295, upper bound: 0.0061209
time: 2.05 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061092, upper bound: 0.0061237
time: 1.99 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0028445, 0.0007308, -0.0024408, 0.0007125, -0.0035570, 0.0031716
1: -0.0057107, -0.0024562, -0.0058655, -0.0025025, -0.0027881, 0.0028107
2: 0.0311472, 0.0349236, 0.0313813, 0.0345476, -0.0034003, 0.0035423
3: -0.0030020, 0.0007682, -0.0029484, 0.0009475, -0.0039496, 0.0037166
4: -0.0047018, -0.0000223, -0.0048593, -0.0002219, -0.0044799, 0.0048370
5: 0.0100379, 0.0133947, 0.0103735, 0.0133319, -0.0032940, 0.0030212
6: -0.0047925, 0.0013631, -0.0045717, 0.0015907, -0.0063832, 0.0059348
7: 0.9736719, 0.9790131, 0.9742079, 0.9791723, -0.0055004, 0.0048051
8: -0.0146872, -0.0037917, -0.0141382, -0.0047279, -0.0099593, 0.0103466
9: -0.0018082, 0.0043586, -0.0012663, 0.0040802, -0.0058884, 0.0056249

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061287, upper bound: 0.0061188
time: 2.07 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061069, upper bound: 0.0061208
time: 2.01 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0025452, 0.0007172, -0.0025089, 0.0007156, -0.0032608, 0.0032262
1: -0.0058063, -0.0024905, -0.0058314, -0.0024947, -0.0027556, 0.0027711
2: 0.0313207, 0.0346448, 0.0313418, 0.0346110, -0.0032903, 0.0033030
3: -0.0029623, 0.0008790, -0.0029575, 0.0009080, -0.0038703, 0.0038364
4: -0.0047991, -0.0001703, -0.0048246, -0.0001882, -0.0046108, 0.0046543
5: 0.0102867, 0.0133481, 0.0103169, 0.0133425, -0.0030558, 0.0030313
6: -0.0046288, 0.0015036, -0.0046090, 0.0015406, -0.0061694, 0.0061126
7: 0.9740694, 0.9791113, 0.9741175, 0.9791372, -0.0050678, 0.0049938
8: -0.0142802, -0.0044857, -0.0142309, -0.0045699, -0.0097103, 0.0097451
9: -0.0014064, 0.0041522, -0.0013577, 0.0041272, -0.0055336, 0.0055099

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060250, upper bound: 0.0058797
time: 1.87 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060119, upper bound: 0.0058778
time: 1.74 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0028704, 0.0007319, -0.0030491, 0.0007400, -0.0036104, 0.0037811
1: -0.0058095, -0.0024532, -0.0057400, -0.0024327, -0.0029071, 0.0029227
2: 0.0311322, 0.0349477, 0.0310286, 0.0351142, -0.0039820, 0.0039191
3: -0.0030055, 0.0008826, -0.0030292, 0.0008021, -0.0038076, 0.0039118
4: -0.0048023, -0.0000095, -0.0047316, 0.0000789, -0.0048811, 0.0047221
5: 0.0100163, 0.0133987, 0.0098677, 0.0134265, -0.0034102, 0.0035310
6: -0.0048067, 0.0015083, -0.0049044, 0.0014061, -0.0062128, 0.0064127
7: 0.9736375, 0.9791147, 0.9734002, 0.9790432, -0.0054057, 0.0057145
8: -0.0147224, -0.0037316, -0.0149654, -0.0033171, -0.0114053, 0.0112339
9: -0.0018430, 0.0043765, -0.0020829, 0.0044998, -0.0063428, 0.0064594

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061694, upper bound: 0.0061597
time: 1.92 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061690, upper bound: 0.0061597
time: 2.27 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0029693, 0.0007364, -0.0025089, 0.0007156, -0.0036849, 0.0032453
1: -0.0057105, -0.0024419, -0.0058314, -0.0024947, -0.0028640, 0.0028534
2: 0.0310748, 0.0350399, 0.0313418, 0.0346110, -0.0035362, 0.0036981
3: -0.0030186, 0.0007680, -0.0029575, 0.0009080, -0.0039267, 0.0037254
4: -0.0047016, 0.0000394, -0.0048246, -0.0001882, -0.0045134, 0.0048640
5: 0.0099341, 0.0134141, 0.0103169, 0.0133425, -0.0034084, 0.0030973
6: -0.0048607, 0.0013628, -0.0046090, 0.0015406, -0.0064013, 0.0059718
7: 0.9735062, 0.9790128, 0.9741175, 0.9791372, -0.0056310, 0.0048953
8: -0.0148569, -0.0035022, -0.0142309, -0.0045699, -0.0102870, 0.0107287
9: -0.0019758, 0.0044447, -0.0013577, 0.0041272, -0.0061030, 0.0058024

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061488, upper bound: 0.0061532
time: 1.53 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061480, upper bound: 0.0061529
time: 1.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0032888, 0.0008170, -0.0030491, 0.0007400, -0.0040288, 0.0038662
1: -0.0057136, -0.0023386, -0.0057400, -0.0024327, -0.0030189, 0.0034015
2: 0.0308896, 0.0356308, 0.0310286, 0.0351142, -0.0042246, 0.0046022
3: -0.0030611, 0.0008533, -0.0030292, 0.0008021, -0.0038632, 0.0038825
4: -0.0047047, 0.0003110, -0.0047316, 0.0000789, -0.0047836, 0.0050426
5: 0.0096685, 0.0136431, 0.0098677, 0.0134265, -0.0037581, 0.0037754
6: -0.0055750, 0.0013672, -0.0049044, 0.0014061, -0.0069811, 0.0062716
7: 0.9728536, 0.9790160, 0.9734002, 0.9790432, -0.0061896, 0.0056158
8: -0.0153954, -0.0027613, -0.0149654, -0.0033171, -0.0120783, 0.0122042
9: -0.0024047, 0.0048049, -0.0020829, 0.0044998, -0.0069044, 0.0068878

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061720, upper bound: 0.0061770
time: 1.61 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061715, upper bound: 0.0061770
time: 2.06 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0023045, 0.0007063, -0.0027194, 0.0007251, -0.0030296, 0.0034258
1: -0.0057782, -0.0025181, -0.0060424, -0.0024705, -0.0027032, 0.0029127
2: 0.0314452, 0.0344206, 0.0312197, 0.0348071, -0.0033619, 0.0032008
3: -0.0029303, 0.0008463, -0.0029854, 0.0011524, -0.0040827, 0.0038317
4: -0.0047704, -0.0002893, -0.0050392, -0.0000841, -0.0046863, 0.0047499
5: 0.0104868, 0.0133107, 0.0101418, 0.0133752, -0.0028884, 0.0031688
6: -0.0044972, 0.0014622, -0.0047241, 0.0018507, -0.0063479, 0.0061863
7: 0.9743889, 0.9790824, 0.9738380, 0.9793542, -0.0049653, 0.0052444
8: -0.0139529, -0.0050441, -0.0145171, -0.0040817, -0.0098711, 0.0094730
9: -0.0010832, 0.0039862, -0.0016403, 0.0042724, -0.0053556, 0.0056265

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0057232, upper bound: 0.0058307
time: 1.56 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055985, upper bound: 0.0057520
time: 1.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0028576, 0.0007313, -0.0030673, 0.0007408, -0.0035984, 0.0037986
1: -0.0056866, -0.0024547, -0.0060455, -0.0024306, -0.0028519, 0.0030714
2: 0.0311396, 0.0349358, 0.0310181, 0.0351311, -0.0039915, 0.0039177
3: -0.0030038, 0.0007403, -0.0030316, 0.0011561, -0.0041598, 0.0037719
4: -0.0046773, -0.0000158, -0.0050424, 0.0000878, -0.0047651, 0.0050265
5: 0.0100270, 0.0133967, 0.0098527, 0.0134294, -0.0034024, 0.0035441
6: -0.0047997, 0.0013276, -0.0049143, 0.0018553, -0.0066550, 0.0062419
7: 0.9736545, 0.9789882, 0.9733761, 0.9793575, -0.0057030, 0.0056121
8: -0.0147050, -0.0037613, -0.0149901, -0.0032751, -0.0114299, 0.0112288
9: -0.0018258, 0.0043677, -0.0021072, 0.0045123, -0.0063381, 0.0064749

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061522, upper bound: 0.0061376
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061465, upper bound: 0.0061365
time: 2.07 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0028327, 0.0007302, -0.0036914, 0.0009632, -0.0037959, 0.0044216
1: -0.0056749, -0.0024575, -0.0060392, -0.0021635, -0.0035114, 0.0032383
2: 0.0311540, 0.0349126, 0.0306562, 0.0365725, -0.0054184, 0.0042564
3: -0.0030005, 0.0007267, -0.0031146, 0.0012899, -0.0042904, 0.0038413
4: -0.0046654, -0.0000281, -0.0050359, 0.0007295, -0.0053949, 0.0050078
5: 0.0100476, 0.0133929, 0.0093337, 0.0140522, -0.0040046, 0.0040591
6: -0.0047861, 0.0013104, -0.0068375, 0.0018459, -0.0066320, 0.0081479
7: 0.9736875, 0.9789762, 0.9718778, 0.9793509, -0.0056635, 0.0070984
8: -0.0146712, -0.0038190, -0.0161439, -0.0018275, -0.0128437, 0.0123249
9: -0.0017924, 0.0043505, -0.0029452, 0.0053526, -0.0071450, 0.0072957

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061555, upper bound: 0.0061228
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061485, upper bound: 0.0061195
time: 1.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0023479, 0.0007083, -0.0026846, 0.0007235, -0.0030715, 0.0033929
1: -0.0057960, -0.0025131, -0.0061010, -0.0024745, -0.0027310, 0.0029568
2: 0.0314341, 0.0344610, 0.0312399, 0.0347746, -0.0033405, 0.0032211
3: -0.0029361, 0.0008670, -0.0029808, 0.0012203, -0.0041563, 0.0038478
4: -0.0047886, -0.0002678, -0.0050987, -0.0001014, -0.0046872, 0.0048309
5: 0.0104507, 0.0133174, 0.0101708, 0.0133698, -0.0029191, 0.0031466
6: -0.0045210, 0.0014885, -0.0047051, 0.0019368, -0.0064578, 0.0061935
7: 0.9743312, 0.9791008, 0.9738841, 0.9794145, -0.0050833, 0.0052167
8: -0.0140119, -0.0049433, -0.0144697, -0.0041625, -0.0098494, 0.0095264
9: -0.0011416, 0.0040161, -0.0015935, 0.0042484, -0.0053899, 0.0056097

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060681, upper bound: 0.0058865
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060487, upper bound: 0.0058827
time: 1.40 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0026797, 0.0007233, -0.0031928, 0.0007822, -0.0034619, 0.0039161
1: -0.0057991, -0.0024751, -0.0060004, -0.0023803, -0.0034188, 0.0030658
2: 0.0312428, 0.0347701, 0.0309453, 0.0354062, -0.0041634, 0.0038248
3: -0.0029801, 0.0008706, -0.0030483, 0.0011037, -0.0040839, 0.0039189
4: -0.0047917, -0.0001038, -0.0049964, 0.0002112, -0.0050029, 0.0048926
5: 0.0101749, 0.0133691, 0.0097483, 0.0135456, -0.0033707, 0.0036207
6: -0.0047024, 0.0014930, -0.0052739, 0.0017889, -0.0064913, 0.0067669
7: 0.9738907, 0.9791039, 0.9730864, 0.9793110, -0.0054203, 0.0060176
8: -0.0144631, -0.0041739, -0.0152169, -0.0029840, -0.0114791, 0.0110430
9: -0.0015870, 0.0042450, -0.0022757, 0.0046743, -0.0062612, 0.0065207

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062028, upper bound: 0.0061545
time: 1.91 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062006, upper bound: 0.0061544
time: 2.08 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0027942, 0.0007285, -0.0026846, 0.0007235, -0.0035177, 0.0034131
1: -0.0057010, -0.0024620, -0.0061010, -0.0024745, -0.0028325, 0.0030464
2: 0.0311764, 0.0348767, 0.0312399, 0.0347746, -0.0035983, 0.0036368
3: -0.0029954, 0.0007569, -0.0029808, 0.0012203, -0.0042156, 0.0037377
4: -0.0046919, -0.0000472, -0.0050987, -0.0001014, -0.0045905, 0.0050516
5: 0.0100797, 0.0133869, 0.0101708, 0.0133698, -0.0032901, 0.0032161
6: -0.0047650, 0.0013487, -0.0047051, 0.0019368, -0.0067018, 0.0060538
7: 0.9737387, 0.9790031, 0.9738841, 0.9794145, -0.0056758, 0.0051190
8: -0.0146187, -0.0039084, -0.0144697, -0.0041625, -0.0104562, 0.0105613
9: -0.0017406, 0.0043239, -0.0015935, 0.0042484, -0.0059890, 0.0059175

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061794, upper bound: 0.0061446
time: 2.17 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061758, upper bound: 0.0061435
time: 2.58 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0031138, 0.0007535, -0.0031928, 0.0007822, -0.0038960, 0.0039463
1: -0.0057040, -0.0024146, -0.0060004, -0.0023803, -0.0033237, 0.0035857
2: 0.0309911, 0.0352216, 0.0309453, 0.0354062, -0.0044151, 0.0042763
3: -0.0030378, 0.0007604, -0.0030483, 0.0011037, -0.0041415, 0.0038087
4: -0.0046949, 0.0001291, -0.0049964, 0.0002112, -0.0049061, 0.0051255
5: 0.0098140, 0.0134654, 0.0097483, 0.0135456, -0.0037316, 0.0037171
6: -0.0050264, 0.0013532, -0.0052739, 0.0017889, -0.0068152, 0.0066271
7: 0.9732777, 0.9790061, 0.9730864, 0.9793110, -0.0060334, 0.0059197
8: -0.0150701, -0.0031670, -0.0152169, -0.0029840, -0.0120861, 0.0120498
9: -0.0021698, 0.0045668, -0.0022757, 0.0046743, -0.0068440, 0.0068426

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062000, upper bound: 0.0061676
time: 2.33 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061969, upper bound: 0.0061675
time: 1.66 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0024876, 0.0007146, -0.0027312, 0.0007256, -0.0032132, 0.0034459
1: -0.0057882, -0.0024971, -0.0060453, -0.0024692, -0.0027563, 0.0029472
2: 0.0313541, 0.0345911, 0.0312129, 0.0348181, -0.0034640, 0.0033782
3: -0.0029546, 0.0008579, -0.0029870, 0.0011558, -0.0041104, 0.0038449
4: -0.0047806, -0.0001988, -0.0050421, -0.0000783, -0.0047023, 0.0048433
5: 0.0103346, 0.0133392, 0.0101320, 0.0133771, -0.0030425, 0.0032071
6: -0.0045973, 0.0014769, -0.0047306, 0.0018550, -0.0064523, 0.0062075
7: 0.9741458, 0.9790927, 0.9738222, 0.9793573, -0.0052115, 0.0052704
8: -0.0142018, -0.0046195, -0.0145332, -0.0040543, -0.0101475, 0.0099137
9: -0.0013290, 0.0041124, -0.0016562, 0.0042805, -0.0056096, 0.0057686

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0057220, upper bound: 0.0058304
time: 1.47 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055963, upper bound: 0.0057519
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0030362, 0.0007394, -0.0030791, 0.0007414, -0.0037776, 0.0038185
1: -0.0056960, -0.0024342, -0.0060485, -0.0024293, -0.0029136, 0.0031062
2: 0.0310361, 0.0351022, 0.0310112, 0.0351421, -0.0041061, 0.0040910
3: -0.0030275, 0.0007511, -0.0030332, 0.0011594, -0.0041869, 0.0037843
4: -0.0046868, 0.0000725, -0.0050453, 0.0000937, -0.0047805, 0.0051178
5: 0.0098785, 0.0134245, 0.0098428, 0.0134312, -0.0035527, 0.0035817
6: -0.0048973, 0.0013414, -0.0049208, 0.0018596, -0.0067569, 0.0062621
7: 0.9734174, 0.9789978, 0.9733605, 0.9793605, -0.0059431, 0.0056373
8: -0.0149479, -0.0033470, -0.0150062, -0.0032475, -0.0117003, 0.0116591
9: -0.0020656, 0.0044909, -0.0021232, 0.0045205, -0.0065860, 0.0066140

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061522, upper bound: 0.0061395
time: 1.38 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061465, upper bound: 0.0061390
time: 1.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0030089, 0.0007382, -0.0037022, 0.0009671, -0.0039759, 0.0044403
1: -0.0056851, -0.0024373, -0.0060410, -0.0021589, -0.0035262, 0.0032723
2: 0.0310519, 0.0350767, 0.0306500, 0.0365975, -0.0055456, 0.0044267
3: -0.0030239, 0.0007385, -0.0031160, 0.0013016, -0.0043255, 0.0038544
4: -0.0046757, 0.0000589, -0.0050377, 0.0007407, -0.0054164, 0.0050967
5: 0.0099012, 0.0134203, 0.0093248, 0.0140631, -0.0041619, 0.0040954
6: -0.0048824, 0.0013253, -0.0068711, 0.0018486, -0.0067310, 0.0081964
7: 0.9734536, 0.9789866, 0.9718519, 0.9793527, -0.0058991, 0.0071347
8: -0.0149107, -0.0034105, -0.0161638, -0.0018026, -0.0131081, 0.0127533
9: -0.0020288, 0.0044720, -0.0029596, 0.0053672, -0.0073961, 0.0074316

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061555, upper bound: 0.0061240
time: 1.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061485, upper bound: 0.0061208
time: 1.84 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0025452, 0.0007172, -0.0026953, 0.0007240, -0.0032692, 0.0034125
1: -0.0058063, -0.0024905, -0.0061022, -0.0024733, -0.0027859, 0.0029919
2: 0.0313207, 0.0346448, 0.0312337, 0.0347846, -0.0034639, 0.0034111
3: -0.0029623, 0.0008790, -0.0029822, 0.0012217, -0.0041840, 0.0038612
4: -0.0047991, -0.0001703, -0.0051000, -0.0000961, -0.0047030, 0.0049297
5: 0.0102867, 0.0133481, 0.0101620, 0.0133715, -0.0030848, 0.0031862
6: -0.0046288, 0.0015036, -0.0047109, 0.0019386, -0.0065675, 0.0062145
7: 0.9740694, 0.9791113, 0.9738700, 0.9794158, -0.0053464, 0.0052413
8: -0.0142802, -0.0044857, -0.0144842, -0.0041378, -0.0101424, 0.0099985
9: -0.0014064, 0.0041522, -0.0016079, 0.0042557, -0.0056621, 0.0057601

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060571, upper bound: 0.0058793
time: 1.62 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060373, upper bound: 0.0058765
time: 1.49 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0028704, 0.0007319, -0.0032036, 0.0007861, -0.0036565, 0.0039355
1: -0.0058095, -0.0024532, -0.0060015, -0.0023756, -0.0034339, 0.0030997
2: 0.0311322, 0.0349477, 0.0309391, 0.0354314, -0.0042992, 0.0040087
3: -0.0030055, 0.0008826, -0.0030497, 0.0011051, -0.0041105, 0.0039323
4: -0.0048023, -0.0000095, -0.0049976, 0.0002224, -0.0050246, 0.0049881
5: 0.0100163, 0.0133987, 0.0097393, 0.0135566, -0.0035402, 0.0036594
6: -0.0048067, 0.0015083, -0.0053078, 0.0017906, -0.0065973, 0.0068160
7: 0.9736375, 0.9791147, 0.9730603, 0.9793122, -0.0056747, 0.0060545
8: -0.0147224, -0.0037316, -0.0152369, -0.0029589, -0.0117635, 0.0115054
9: -0.0018430, 0.0043765, -0.0022902, 0.0046889, -0.0065319, 0.0066668

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062027, upper bound: 0.0061556
time: 1.86 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062006, upper bound: 0.0061555
time: 2.14 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0029693, 0.0007364, -0.0026953, 0.0007240, -0.0036933, 0.0034317
1: -0.0057105, -0.0024419, -0.0061022, -0.0024733, -0.0028913, 0.0030809
2: 0.0310748, 0.0350399, 0.0312337, 0.0347846, -0.0037097, 0.0038061
3: -0.0030186, 0.0007680, -0.0029822, 0.0012217, -0.0042403, 0.0037502
4: -0.0047016, 0.0000394, -0.0051000, -0.0000961, -0.0046055, 0.0051394
5: 0.0099341, 0.0134141, 0.0101620, 0.0133715, -0.0034374, 0.0032522
6: -0.0048607, 0.0013628, -0.0047109, 0.0019386, -0.0067994, 0.0060737
7: 0.9735062, 0.9790128, 0.9738700, 0.9794158, -0.0059096, 0.0051428
8: -0.0148569, -0.0035022, -0.0144842, -0.0041378, -0.0107191, 0.0109821
9: -0.0019758, 0.0044447, -0.0016079, 0.0042557, -0.0062315, 0.0060526

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061794, upper bound: 0.0061460
time: 1.99 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061758, upper bound: 0.0061451
time: 2.17 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0032888, 0.0008170, -0.0032036, 0.0007861, -0.0040749, 0.0040206
1: -0.0057136, -0.0023386, -0.0060015, -0.0023756, -0.0033380, 0.0036630
2: 0.0308896, 0.0356308, 0.0309391, 0.0354314, -0.0045418, 0.0046917
3: -0.0030611, 0.0008533, -0.0030497, 0.0011051, -0.0041661, 0.0039030
4: -0.0047047, 0.0003110, -0.0049976, 0.0002224, -0.0049271, 0.0053086
5: 0.0096685, 0.0136431, 0.0097393, 0.0135566, -0.0038881, 0.0039038
6: -0.0055750, 0.0013672, -0.0053078, 0.0017906, -0.0073656, 0.0066750
7: 0.9728536, 0.9790160, 0.9730603, 0.9793122, -0.0064586, 0.0059558
8: -0.0153954, -0.0027613, -0.0152369, -0.0029589, -0.0124364, 0.0124756
9: -0.0024047, 0.0048049, -0.0022902, 0.0046889, -0.0070936, 0.0070951

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062000, upper bound: 0.0061691
time: 1.74 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061969, upper bound: 0.0061690
time: 2.34 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0023683, 0.0007092, -0.0022173, 0.0007024, -0.0030707, 0.0029265
1: -0.0060661, -0.0025108, -0.0058300, -0.0025281, -0.0028499, 0.0026740
2: 0.0312666, 0.0344801, 0.0314131, 0.0343394, -0.0030728, 0.0030670
3: -0.0029388, 0.0011798, -0.0029187, 0.0009063, -0.0038451, 0.0040985
4: -0.0050632, -0.0002577, -0.0048231, -0.0003324, -0.0047308, 0.0045654
5: 0.0104338, 0.0133206, 0.0105593, 0.0132971, -0.0028633, 0.0027613
6: -0.0045321, 0.0018855, -0.0044495, 0.0015384, -0.0060705, 0.0063350
7: 0.9743041, 0.9793786, 0.9745045, 0.9791357, -0.0048316, 0.0048741
8: -0.0140397, -0.0048960, -0.0138343, -0.0052462, -0.0087934, 0.0089383
9: -0.0011689, 0.0040302, -0.0009662, 0.0039260, -0.0050950, 0.0049964

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0056196, upper bound: 0.0056564
time: 1.25 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055260, upper bound: 0.0056543
time: 1.55 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0022386, 0.0007034, -0.0020963, 0.0006969, -0.0029355, 0.0027996
1: -0.0060659, -0.0025257, -0.0058594, -0.0025420, -0.0028219, 0.0026714
2: 0.0312667, 0.0343592, 0.0313948, 0.0342266, -0.0029599, 0.0029644
3: -0.0029215, 0.0011797, -0.0029026, 0.0009405, -0.0038620, 0.0040823
4: -0.0050631, -0.0003219, -0.0048531, -0.0003923, -0.0046708, 0.0045312
5: 0.0105416, 0.0133004, 0.0106600, 0.0132783, -0.0027366, 0.0026405
6: -0.0044612, 0.0018852, -0.0043833, 0.0015817, -0.0060429, 0.0062686
7: 0.9744763, 0.9793785, 0.9746654, 0.9791661, -0.0046898, 0.0047131
8: -0.0138633, -0.0051969, -0.0136697, -0.0055270, -0.0083363, 0.0084728
9: -0.0009948, 0.0039407, -0.0008037, 0.0038425, -0.0048373, 0.0047444

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0056102, upper bound: 0.0056355
time: 1.30 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055207, upper bound: 0.0056341
time: 1.31 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0029221, 0.0007343, -0.0025374, 0.0007169, -0.0036390, 0.0032716
1: -0.0059657, -0.0024473, -0.0058331, -0.0024914, -0.0029517, 0.0028141
2: 0.0311022, 0.0349959, 0.0313253, 0.0346375, -0.0035353, 0.0036706
3: -0.0030124, 0.0010636, -0.0029612, 0.0009099, -0.0039222, 0.0040248
4: -0.0049611, 0.0000161, -0.0048262, -0.0001741, -0.0047870, 0.0048423
5: 0.0099733, 0.0134068, 0.0102932, 0.0133469, -0.0033736, 0.0031136
6: -0.0048349, 0.0017379, -0.0046246, 0.0015429, -0.0063778, 0.0063625
7: 0.9735689, 0.9792753, 0.9740797, 0.9791389, -0.0055701, 0.0051956
8: -0.0147927, -0.0036117, -0.0142696, -0.0045039, -0.0102888, 0.0106579
9: -0.0019124, 0.0044122, -0.0013959, 0.0041468, -0.0060592, 0.0058081

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061273, upper bound: 0.0061665
time: 1.81 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061092, upper bound: 0.0061724
time: 1.86 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0028138, 0.0007294, -0.0024285, 0.0007119, -0.0035257, 0.0031578
1: -0.0059655, -0.0024597, -0.0058625, -0.0025039, -0.0029194, 0.0028048
2: 0.0311650, 0.0348950, 0.0313884, 0.0345361, -0.0033710, 0.0035065
3: -0.0029980, 0.0010634, -0.0029468, 0.0009440, -0.0039420, 0.0040101
4: -0.0049610, -0.0000375, -0.0048562, -0.0002280, -0.0047330, 0.0048187
5: 0.0100634, 0.0133899, 0.0103838, 0.0133300, -0.0032665, 0.0030062
6: -0.0047757, 0.0017377, -0.0045650, 0.0015862, -0.0063619, 0.0063027
7: 0.9737127, 0.9792752, 0.9742242, 0.9791692, -0.0054565, 0.0050510
8: -0.0146454, -0.0038629, -0.0141215, -0.0047565, -0.0098888, 0.0102585
9: -0.0017669, 0.0043374, -0.0012497, 0.0040717, -0.0058386, 0.0055871

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061259, upper bound: 0.0061642
time: 1.94 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061069, upper bound: 0.0061684
time: 1.90 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0025628, 0.0007180, -0.0024984, 0.0007151, -0.0032779, 0.0032164
1: -0.0060596, -0.0024885, -0.0058283, -0.0024959, -0.0029216, 0.0027622
2: 0.0312706, 0.0346612, 0.0313479, 0.0346012, -0.0033306, 0.0033134
3: -0.0029646, 0.0011724, -0.0029561, 0.0009044, -0.0038690, 0.0041284
4: -0.0050567, -0.0001616, -0.0048214, -0.0001934, -0.0048633, 0.0046598
5: 0.0102720, 0.0133509, 0.0103256, 0.0133409, -0.0030688, 0.0030253
6: -0.0046385, 0.0018760, -0.0046032, 0.0015359, -0.0061744, 0.0064793
7: 0.9740458, 0.9793720, 0.9741315, 0.9791340, -0.0050882, 0.0052405
8: -0.0143042, -0.0044449, -0.0142166, -0.0045943, -0.0097098, 0.0097716
9: -0.0014301, 0.0041644, -0.0013436, 0.0041199, -0.0055500, 0.0055080

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060827, upper bound: 0.0059336
time: 1.70 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059443, upper bound: 0.0059246
time: 1.56 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0029124, 0.0007338, -0.0030390, 0.0007395, -0.0036519, 0.0037728
1: -0.0060628, -0.0024484, -0.0057371, -0.0024339, -0.0030754, 0.0029144
2: 0.0311079, 0.0349868, 0.0310345, 0.0351047, -0.0039969, 0.0039524
3: -0.0030111, 0.0011761, -0.0030279, 0.0007988, -0.0038098, 0.0042040
4: -0.0050600, 0.0000113, -0.0047286, 0.0000738, -0.0051338, 0.0047399
5: 0.0099814, 0.0134053, 0.0098762, 0.0134250, -0.0034435, 0.0035291
6: -0.0048296, 0.0018807, -0.0048988, 0.0014019, -0.0062315, 0.0067796
7: 0.9735818, 0.9793753, 0.9734138, 0.9790402, -0.0054584, 0.0059615
8: -0.0147795, -0.0036342, -0.0149516, -0.0033407, -0.0114388, 0.0113173
9: -0.0018993, 0.0044055, -0.0020692, 0.0044928, -0.0063921, 0.0064747

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061566, upper bound: 0.0061718
time: 1.90 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061520, upper bound: 0.0061762
time: 2.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0029860, 0.0007371, -0.0024984, 0.0007151, -0.0037011, 0.0032355
1: -0.0059651, -0.0024400, -0.0058283, -0.0024959, -0.0030046, 0.0028420
2: 0.0310652, 0.0350554, 0.0313479, 0.0346012, -0.0035361, 0.0037076
3: -0.0030208, 0.0010629, -0.0029561, 0.0009044, -0.0039253, 0.0040190
4: -0.0049606, 0.0000477, -0.0048214, -0.0001934, -0.0047672, 0.0048691
5: 0.0099202, 0.0134167, 0.0103256, 0.0133409, -0.0034206, 0.0030911
6: -0.0048699, 0.0017371, -0.0046032, 0.0015359, -0.0064058, 0.0063403
7: 0.9734840, 0.9792747, 0.9741315, 0.9791340, -0.0056499, 0.0051433
8: -0.0148796, -0.0034635, -0.0142166, -0.0045943, -0.0102853, 0.0107531
9: -0.0019982, 0.0044563, -0.0013436, 0.0041199, -0.0061181, 0.0057998

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061491, upper bound: 0.0061812
time: 1.88 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061480, upper bound: 0.0061899
time: 1.79 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0033226, 0.0008293, -0.0030390, 0.0007395, -0.0040621, 0.0038683
1: -0.0059682, -0.0023239, -0.0057371, -0.0024339, -0.0031636, 0.0034133
2: 0.0308701, 0.0357098, 0.0310345, 0.0351047, -0.0042347, 0.0046753
3: -0.0030656, 0.0010665, -0.0030279, 0.0007988, -0.0038643, 0.0040944
4: -0.0049637, 0.0003461, -0.0047286, 0.0000738, -0.0050376, 0.0050747
5: 0.0096404, 0.0136775, 0.0098762, 0.0134250, -0.0037846, 0.0038013
6: -0.0056809, 0.0017416, -0.0048988, 0.0014019, -0.0070828, 0.0066405
7: 0.9727718, 0.9792780, 0.9734138, 0.9790402, -0.0062684, 0.0058642
8: -0.0154582, -0.0026829, -0.0149516, -0.0033407, -0.0121175, 0.0122687
9: -0.0024500, 0.0048508, -0.0020692, 0.0044928, -0.0069427, 0.0069201

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061625, upper bound: 0.0061917
time: 2.35 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061624, upper bound: 0.0062024
time: 2.25 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0025695, 0.0007183, -0.0022290, 0.0007029, -0.0032724, 0.0029473
1: -0.0060720, -0.0024877, -0.0058333, -0.0025268, -0.0028958, 0.0027116
2: 0.0312629, 0.0346675, 0.0314110, 0.0343502, -0.0030873, 0.0032565
3: -0.0029655, 0.0011867, -0.0029203, 0.0009102, -0.0038757, 0.0041070
4: -0.0050693, -0.0001583, -0.0048265, -0.0003266, -0.0047426, 0.0046683
5: 0.0102665, 0.0133519, 0.0105496, 0.0132989, -0.0030324, 0.0028023
6: -0.0046421, 0.0018943, -0.0044559, 0.0015433, -0.0061855, 0.0063502
7: 0.9740369, 0.9793848, 0.9744892, 0.9791392, -0.0051023, 0.0048956
8: -0.0143132, -0.0044295, -0.0138501, -0.0052193, -0.0090940, 0.0094207
9: -0.0014390, 0.0041690, -0.0009818, 0.0039341, -0.0053731, 0.0051508

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055958, upper bound: 0.0056433
time: 1.45 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055165, upper bound: 0.0056430
time: 1.46 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0024416, 0.0007125, -0.0021083, 0.0006975, -0.0031390, 0.0028209
1: -0.0060719, -0.0025024, -0.0058625, -0.0025406, -0.0028658, 0.0027082
2: 0.0312630, 0.0345483, 0.0313929, 0.0342379, -0.0029749, 0.0031553
3: -0.0029485, 0.0011865, -0.0029042, 0.0009440, -0.0038925, 0.0040908
4: -0.0050691, -0.0002215, -0.0048561, -0.0003863, -0.0046829, 0.0046346
5: 0.0103729, 0.0133320, 0.0106499, 0.0132802, -0.0029073, 0.0026821
6: -0.0045721, 0.0018940, -0.0043899, 0.0015862, -0.0061583, 0.0062840
7: 0.9742069, 0.9793845, 0.9746493, 0.9791691, -0.0049622, 0.0047352
8: -0.0141392, -0.0047262, -0.0136862, -0.0054990, -0.0086402, 0.0089599
9: -0.0012672, 0.0040807, -0.0008199, 0.0038509, -0.0051181, 0.0049006

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055859, upper bound: 0.0056240
time: 1.65 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055100, upper bound: 0.0056234
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0030954, 0.0007469, -0.0025491, 0.0007174, -0.0038128, 0.0032960
1: -0.0059711, -0.0024226, -0.0058364, -0.0024901, -0.0030032, 0.0034138
2: 0.0310018, 0.0351784, 0.0313185, 0.0346484, -0.0036466, 0.0038599
3: -0.0030354, 0.0010699, -0.0029628, 0.0009138, -0.0039492, 0.0040327
4: -0.0049667, 0.0001099, -0.0048297, -0.0001684, -0.0047983, 0.0049396
5: 0.0098293, 0.0134466, 0.0102835, 0.0133487, -0.0035195, 0.0031632
6: -0.0049686, 0.0017459, -0.0046310, 0.0015479, -0.0065164, 0.0063769
7: 0.9733223, 0.9792809, 0.9740641, 0.9791423, -0.0058200, 0.0052168
8: -0.0150358, -0.0032098, -0.0142855, -0.0044768, -0.0105590, 0.0110756
9: -0.0021450, 0.0045417, -0.0014116, 0.0041549, -0.0062999, 0.0059534

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061273, upper bound: 0.0061671
time: 1.49 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061092, upper bound: 0.0061737
time: 1.84 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0029857, 0.0007371, -0.0024408, 0.0007125, -0.0036982, 0.0031779
1: -0.0059710, -0.0024400, -0.0058655, -0.0025025, -0.0029682, 0.0028435
2: 0.0310654, 0.0350551, 0.0313813, 0.0345476, -0.0034822, 0.0036739
3: -0.0030208, 0.0010697, -0.0029484, 0.0009475, -0.0039683, 0.0040181
4: -0.0049665, 0.0000475, -0.0048593, -0.0002219, -0.0047446, 0.0049068
5: 0.0099205, 0.0134167, 0.0103735, 0.0133319, -0.0034114, 0.0030432
6: -0.0048697, 0.0017457, -0.0045717, 0.0015907, -0.0064604, 0.0063174
7: 0.9734845, 0.9792807, 0.9742079, 0.9791723, -0.0056878, 0.0050728
8: -0.0148792, -0.0034642, -0.0141382, -0.0047279, -0.0101513, 0.0106740
9: -0.0019977, 0.0044560, -0.0012663, 0.0040802, -0.0060780, 0.0057223

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061259, upper bound: 0.0061645
time: 2.26 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061069, upper bound: 0.0061693
time: 1.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0027607, 0.0007270, -0.0025089, 0.0007156, -0.0034763, 0.0032359
1: -0.0060674, -0.0024658, -0.0058314, -0.0024947, -0.0029714, 0.0027997
2: 0.0311958, 0.0348456, 0.0313418, 0.0346110, -0.0034152, 0.0035038
3: -0.0029909, 0.0011813, -0.0029575, 0.0009080, -0.0038989, 0.0041388
4: -0.0050646, -0.0000637, -0.0048246, -0.0001882, -0.0048763, 0.0047609
5: 0.0101075, 0.0133817, 0.0103169, 0.0133425, -0.0032350, 0.0030648
6: -0.0047467, 0.0018874, -0.0046090, 0.0015406, -0.0062873, 0.0064964
7: 0.9737831, 0.9793799, 0.9741175, 0.9791372, -0.0053541, 0.0052624
8: -0.0145733, -0.0039859, -0.0142309, -0.0045699, -0.0100033, 0.0102449
9: -0.0016958, 0.0043009, -0.0013577, 0.0041272, -0.0058229, 0.0056586

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060790, upper bound: 0.0059242
time: 1.83 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059435, upper bound: 0.0059177
time: 1.63 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0031083, 0.0007516, -0.0030491, 0.0007400, -0.0038483, 0.0038007
1: -0.0060705, -0.0024170, -0.0057400, -0.0024327, -0.0031302, 0.0033230
2: 0.0309943, 0.0352086, 0.0310286, 0.0351142, -0.0041199, 0.0041800
3: -0.0030371, 0.0011850, -0.0030292, 0.0008021, -0.0038392, 0.0042142
4: -0.0050678, 0.0001233, -0.0047316, 0.0000789, -0.0051466, 0.0048549
5: 0.0098186, 0.0134598, 0.0098677, 0.0134265, -0.0036080, 0.0035920
6: -0.0050090, 0.0018921, -0.0049044, 0.0014061, -0.0064151, 0.0067965
7: 0.9732910, 0.9793832, 0.9734002, 0.9790432, -0.0057523, 0.0059830
8: -0.0150598, -0.0031799, -0.0149654, -0.0033171, -0.0117427, 0.0117855
9: -0.0021623, 0.0045593, -0.0020829, 0.0044998, -0.0066621, 0.0066422

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061566, upper bound: 0.0061712
time: 2.04 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061520, upper bound: 0.0061762
time: 2.00 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0031594, 0.0007701, -0.0025089, 0.0007156, -0.0038750, 0.0032790
1: -0.0059707, -0.0023948, -0.0058314, -0.0024947, -0.0030550, 0.0034367
2: 0.0309646, 0.0353282, 0.0313418, 0.0346110, -0.0036464, 0.0039865
3: -0.0030439, 0.0010693, -0.0029575, 0.0009080, -0.0039519, 0.0040268
4: -0.0049662, 0.0001765, -0.0048246, -0.0001882, -0.0047780, 0.0050011
5: 0.0097760, 0.0135117, 0.0103169, 0.0133425, -0.0035665, 0.0031948
6: -0.0051693, 0.0017453, -0.0046090, 0.0015406, -0.0067099, 0.0063543
7: 0.9731671, 0.9792805, 0.9741175, 0.9791372, -0.0059701, 0.0051630
8: -0.0151549, -0.0030613, -0.0142309, -0.0045699, -0.0105849, 0.0111696
9: -0.0022310, 0.0046289, -0.0013577, 0.0041272, -0.0063582, 0.0059866

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061491, upper bound: 0.0061816
time: 2.31 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061480, upper bound: 0.0061912
time: 2.27 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0034823, 0.0008873, -0.0030491, 0.0007400, -0.0042223, 0.0039364
1: -0.0059738, -0.0022545, -0.0057400, -0.0024327, -0.0032127, 0.0034856
2: 0.0307775, 0.0360833, 0.0310286, 0.0351142, -0.0043367, 0.0050547
3: -0.0030868, 0.0010729, -0.0030292, 0.0008021, -0.0038889, 0.0041021
4: -0.0049693, 0.0005121, -0.0047316, 0.0000789, -0.0050482, 0.0052437
5: 0.0095076, 0.0138397, 0.0098677, 0.0134265, -0.0039189, 0.0039720
6: -0.0061818, 0.0017498, -0.0049044, 0.0014061, -0.0075879, 0.0066542
7: 0.9723846, 0.9792836, 0.9734002, 0.9790432, -0.0066586, 0.0058834
8: -0.0157551, -0.0023125, -0.0149654, -0.0033171, -0.0124380, 0.0126529
9: -0.0026644, 0.0050681, -0.0020829, 0.0044998, -0.0071642, 0.0071510

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061624, upper bound: 0.0061922
time: 2.22 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061624, upper bound: 0.0062037
time: 2.22 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0024647, 0.0007136, -0.0027194, 0.0007251, -0.0031898, 0.0034330
1: -0.0060513, -0.0024998, -0.0060424, -0.0024705, -0.0028274, 0.0028513
2: 0.0312758, 0.0345698, 0.0312197, 0.0348071, -0.0035313, 0.0033501
3: -0.0029516, 0.0011628, -0.0029854, 0.0011524, -0.0041040, 0.0041482
4: -0.0050483, -0.0002101, -0.0050392, -0.0000841, -0.0049641, 0.0048291
5: 0.0103536, 0.0133356, 0.0101418, 0.0133752, -0.0030216, 0.0031937
6: -0.0045848, 0.0018638, -0.0047241, 0.0018507, -0.0064355, 0.0065879
7: 0.9741762, 0.9793634, 0.9738380, 0.9793542, -0.0051780, 0.0055254
8: -0.0141707, -0.0046725, -0.0145171, -0.0040817, -0.0100890, 0.0098446
9: -0.0012983, 0.0040967, -0.0016403, 0.0042724, -0.0055707, 0.0057370

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0056754, upper bound: 0.0058674
time: 1.83 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055798, upper bound: 0.0058169
time: 1.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0024402, 0.0007125, -0.0033422, 0.0007532, -0.0031934, 0.0040547
1: -0.0060393, -0.0025026, -0.0060358, -0.0023991, -0.0028834, 0.0030305
2: 0.0312832, 0.0345470, 0.0308587, 0.0353872, -0.0041041, 0.0036883
3: -0.0029483, 0.0011489, -0.0030682, 0.0011448, -0.0040931, 0.0042170
4: -0.0050361, -0.0002222, -0.0050324, 0.0002238, -0.0052598, 0.0048102
5: 0.0103740, 0.0133318, 0.0096241, 0.0134721, -0.0030981, 0.0037077
6: -0.0045714, 0.0018462, -0.0050647, 0.0018410, -0.0064124, 0.0069109
7: 0.9742087, 0.9793512, 0.9730111, 0.9793475, -0.0051388, 0.0063401
8: -0.0141374, -0.0047294, -0.0153639, -0.0026374, -0.0115000, 0.0106345
9: -0.0012654, 0.0040798, -0.0024763, 0.0047019, -0.0059673, 0.0065561

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0056562, upper bound: 0.0057230
time: 1.64 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055585, upper bound: 0.0056597
time: 1.36 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0030257, 0.0007389, -0.0030673, 0.0007408, -0.0037665, 0.0038062
1: -0.0059508, -0.0024354, -0.0060455, -0.0024306, -0.0029701, 0.0030127
2: 0.0310422, 0.0350923, 0.0310181, 0.0351311, -0.0040889, 0.0040743
3: -0.0030261, 0.0010463, -0.0030316, 0.0011561, -0.0041822, 0.0040779
4: -0.0049460, 0.0000673, -0.0050424, 0.0000878, -0.0050338, 0.0051096
5: 0.0098873, 0.0134229, 0.0098527, 0.0134294, -0.0035421, 0.0035702
6: -0.0048916, 0.0017160, -0.0049143, 0.0018553, -0.0067469, 0.0066303
7: 0.9734314, 0.9792600, 0.9733761, 0.9793575, -0.0059261, 0.0058839
8: -0.0149335, -0.0033715, -0.0149901, -0.0032751, -0.0116584, 0.0116185
9: -0.0020514, 0.0044836, -0.0021072, 0.0045123, -0.0065637, 0.0065908

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061194, upper bound: 0.0061794
time: 1.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061188, upper bound: 0.0061794
time: 1.82 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0030010, 0.0007378, -0.0036914, 0.0009632, -0.0039642, 0.0044292
1: -0.0059392, -0.0024382, -0.0060392, -0.0021635, -0.0037756, 0.0032081
2: 0.0310565, 0.0350694, 0.0306562, 0.0365725, -0.0055160, 0.0044132
3: -0.0030228, 0.0010328, -0.0031146, 0.0012899, -0.0043128, 0.0041474
4: -0.0049342, 0.0000551, -0.0050359, 0.0007295, -0.0056637, 0.0050910
5: 0.0099077, 0.0134190, 0.0093337, 0.0140522, -0.0041445, 0.0040853
6: -0.0048781, 0.0016989, -0.0068375, 0.0018459, -0.0067240, 0.0085365
7: 0.9734642, 0.9792480, 0.9718778, 0.9793509, -0.0058867, 0.0073702
8: -0.0149000, -0.0034287, -0.0161439, -0.0018275, -0.0130725, 0.0127152
9: -0.0020183, 0.0044666, -0.0029452, 0.0053526, -0.0073710, 0.0074118

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061206, upper bound: 0.0061714
time: 1.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061200, upper bound: 0.0061714
time: 1.68 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0025628, 0.0007180, -0.0026846, 0.0007235, -0.0032864, 0.0034026
1: -0.0060596, -0.0024885, -0.0061010, -0.0024745, -0.0028624, 0.0028895
2: 0.0312706, 0.0346612, 0.0312399, 0.0347746, -0.0035040, 0.0034213
3: -0.0029646, 0.0011724, -0.0029808, 0.0012203, -0.0041849, 0.0041532
4: -0.0050567, -0.0001616, -0.0050987, -0.0001014, -0.0049553, 0.0049372
5: 0.0102720, 0.0133509, 0.0101708, 0.0133698, -0.0030978, 0.0031801
6: -0.0046385, 0.0018760, -0.0047051, 0.0019368, -0.0065753, 0.0065811
7: 0.9740458, 0.9793720, 0.9738841, 0.9794145, -0.0053687, 0.0054879
8: -0.0143042, -0.0044449, -0.0144697, -0.0041625, -0.0101417, 0.0100248
9: -0.0014301, 0.0041644, -0.0015935, 0.0042484, -0.0056784, 0.0057579

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060836, upper bound: 0.0059336
time: 1.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059489, upper bound: 0.0059246
time: 1.39 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.47 + 598.43 = 602.90 seconds
