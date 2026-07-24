## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.012999239999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621)
1: (-0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673)
2: (0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0190535, 0.0190535)
3: (-0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474)
4: (0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0052041, 0.0052041)
5: (-0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063)
6: (-0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656)
7: (-0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616)
8: (-0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768)
9: (0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.79 + 2.68 = 4.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0144436, upper bound: 0.0144436

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139782, upper bound: 0.0141357
time: 1.70 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141097, upper bound: 0.0141097
time: 1.99 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.89 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.89
Output dim: 9, lower bound: -0.0139782, upper bound: 0.0141357
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.89
Output dim: 9, lower bound: -0.0141097, upper bound: 0.0141097

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0058681, 0.0028047, -0.0060496, 0.0032124, -0.0090805, 0.0088544
1: -0.0033364, 0.0129989, -0.0036231, 0.0133442, -0.0166806, 0.0166221
2: 0.0049230, 0.0235649, 0.0049072, 0.0242831, -0.0190379, 0.0182517
3: -0.0082160, -0.0018313, -0.0086418, -0.0017943, -0.0064217, 0.0068105
4: 0.0029355, 0.0077320, 0.0025300, 0.0077342, -0.0047987, 0.0052019
5: -0.0060685, 0.0014632, -0.0062876, 0.0019187, -0.0079872, 0.0077508
6: -0.0073471, -0.0045518, -0.0074834, -0.0043177, -0.0030293, 0.0029316
7: -0.0062436, 0.0013010, -0.0063982, 0.0016634, -0.0079071, 0.0076992
8: -0.0101357, -0.0013230, -0.0105992, -0.0012224, -0.0089133, 0.0092762
9: 0.9919425, 1.0125992, 0.9911500, 1.0128294, -0.0208870, 0.0214493

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136063, upper bound: 0.0138772
time: 1.96 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136056, upper bound: 0.0138773
time: 1.92 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0058934, 0.0028588, -0.0060204, 0.0031462, -0.0090396, 0.0088792
1: -0.0037534, 0.0130331, -0.0035707, 0.0132863, -0.0170397, 0.0166038
2: 0.0046538, 0.0236478, 0.0049108, 0.0241652, -0.0191501, 0.0184153
3: -0.0082821, -0.0017038, -0.0085749, -0.0018010, -0.0064811, 0.0068711
4: 0.0028747, 0.0079229, 0.0026008, 0.0077337, -0.0048590, 0.0053222
5: -0.0062997, 0.0015317, -0.0062506, 0.0018431, -0.0081429, 0.0077823
6: -0.0073662, -0.0045228, -0.0074617, -0.0043643, -0.0030019, 0.0029389
7: -0.0063086, 0.0013925, -0.0063666, 0.0015993, -0.0079079, 0.0077591
8: -0.0101879, -0.0013065, -0.0105219, -0.0012388, -0.0089491, 0.0092153
9: 0.9917983, 1.0132831, 0.9912813, 1.0127872, -0.0209889, 0.0220018

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137381, upper bound: 0.0138676
time: 1.51 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138676, upper bound: 0.0138676
time: 2.40 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.58 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 5.58
Output dim: 9, lower bound: -0.0136063, upper bound: 0.0138772
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 5.58
Output dim: 9, lower bound: -0.0136056, upper bound: 0.0138773
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 5.58
Output dim: 9, lower bound: -0.0137381, upper bound: 0.0138676
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 5.58
Output dim: 9, lower bound: -0.0138676, upper bound: 0.0138676

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -0.0053727, 0.0012404, -0.0059988, 0.0030505, -0.0084232, 0.0072393
1: -0.0029929, 0.0113409, -0.0036031, 0.0131742, -0.0161671, 0.0149440
2: 0.0050854, 0.0205099, 0.0049081, 0.0239698, -0.0182268, 0.0150836
3: -0.0070639, -0.0018734, -0.0085247, -0.0017981, -0.0052658, 0.0066513
4: 0.0033020, 0.0077088, 0.0025651, 0.0077340, -0.0043997, 0.0051437
5: -0.0055546, 0.0010362, -0.0062208, 0.0018778, -0.0074324, 0.0072571
6: -0.0068887, -0.0046656, -0.0074363, -0.0043264, -0.0025624, 0.0027707
7: -0.0055162, 0.0007463, -0.0063249, 0.0016107, -0.0071270, 0.0070711
8: -0.0081030, -0.0013371, -0.0103896, -0.0012233, -0.0068796, 0.0090524
9: 0.9938440, 1.0123242, 0.9913414, 1.0128148, -0.0189708, 0.0209828

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136063, upper bound: 0.0138049
time: 1.56 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136063, upper bound: 0.0138772
time: 2.64 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -0.0056013, 0.0020201, -0.0060376, 0.0031758, -0.0087771, 0.0080576
1: -0.0032511, 0.0122105, -0.0036183, 0.0133071, -0.0165582, 0.0158288
2: 0.0049314, 0.0220756, 0.0049076, 0.0242134, -0.0189650, 0.0159237
3: -0.0076040, -0.0018496, -0.0086135, -0.0017952, -0.0058089, 0.0067639
4: 0.0031385, 0.0077305, 0.0025387, 0.0077341, -0.0043816, 0.0051918
5: -0.0057961, 0.0012136, -0.0062727, 0.0019079, -0.0077040, 0.0074863
6: -0.0071121, -0.0046135, -0.0074725, -0.0043202, -0.0027919, 0.0028590
7: -0.0059255, 0.0010532, -0.0063836, 0.0016512, -0.0075767, 0.0074368
8: -0.0091426, -0.0013279, -0.0105528, -0.0012226, -0.0079200, 0.0092250
9: 0.9929135, 1.0125357, 0.9911945, 1.0128260, -0.0199125, 0.0213412

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136843, upper bound: 0.0137432
time: 1.62 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136843, upper bound: 0.0138772
time: 1.75 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.0053793, 0.0012629, -0.0059696, 0.0029833, -0.0083626, 0.0072325
1: -0.0033896, 0.0113412, -0.0035513, 0.0131153, -0.0165049, 0.0148925
2: 0.0048375, 0.0205157, 0.0049117, 0.0238497, -0.0182992, 0.0151847
3: -0.0070845, -0.0017411, -0.0084572, -0.0018048, -0.0052797, 0.0067161
4: 0.0032438, 0.0078796, 0.0026365, 0.0077335, -0.0044897, 0.0052430
5: -0.0058458, 0.0010971, -0.0061838, 0.0018013, -0.0076471, 0.0072809
6: -0.0068926, -0.0046262, -0.0074143, -0.0043730, -0.0025196, 0.0027881
7: -0.0055968, 0.0008190, -0.0062928, 0.0015465, -0.0071433, 0.0071118
8: -0.0081091, -0.0013180, -0.0103112, -0.0012398, -0.0068694, 0.0089932
9: 0.9937469, 1.0129554, 0.9914743, 1.0127728, -0.0190259, 0.0214811

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137381, upper bound: 0.0137381
time: 3.94 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137381, upper bound: 0.0138676
time: 1.85 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.0056385, 0.0020906, -0.0060084, 0.0031100, -0.0087485, 0.0080990
1: -0.0036719, 0.0122439, -0.0035661, 0.0132494, -0.0169213, 0.0158100
2: 0.0046625, 0.0221690, 0.0049111, 0.0240961, -0.0190775, 0.0161158
3: -0.0076904, -0.0017223, -0.0085470, -0.0018018, -0.0058886, 0.0068247
4: 0.0030661, 0.0079214, 0.0026095, 0.0077336, -0.0044969, 0.0053120
5: -0.0060600, 0.0012990, -0.0062360, 0.0018324, -0.0078924, 0.0075349
6: -0.0071367, -0.0045795, -0.0074509, -0.0043667, -0.0027699, 0.0028714
7: -0.0059889, 0.0011385, -0.0063519, 0.0015873, -0.0075763, 0.0074904
8: -0.0092090, -0.0013116, -0.0104760, -0.0012390, -0.0079700, 0.0091643
9: 0.9927459, 1.0132235, 0.9913258, 1.0127838, -0.0200379, 0.0218977

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138676, upper bound: 0.0137381
time: 2.01 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138676, upper bound: 0.0138676
time: 2.12 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.87 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 5.87
Output dim: 9, lower bound: -0.0136063, upper bound: 0.0138049
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 5.87
Output dim: 9, lower bound: -0.0136063, upper bound: 0.0138772
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.87
Output dim: 9, lower bound: -0.0136843, upper bound: 0.0137432
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 5.87
Output dim: 9, lower bound: -0.0136843, upper bound: 0.0138772
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 5.87
Output dim: 9, lower bound: -0.0137381, upper bound: 0.0137381
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 5.87
Output dim: 9, lower bound: -0.0137381, upper bound: 0.0138676
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.87
Output dim: 9, lower bound: -0.0138676, upper bound: 0.0137381
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 5.87
Output dim: 9, lower bound: -0.0138676, upper bound: 0.0138676

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0053727, 0.0012404, -0.0058183, 0.0026439, -0.0080166, 0.0070587
1: -0.0029929, 0.0113409, -0.0033205, 0.0128298, -0.0158227, 0.0146615
2: 0.0050854, 0.0205099, 0.0049239, 0.0232530, -0.0174257, 0.0150680
3: -0.0070639, -0.0018734, -0.0080996, -0.0018350, -0.0052289, 0.0062262
4: 0.0033020, 0.0077088, 0.0029707, 0.0077318, -0.0043942, 0.0046955
5: -0.0055546, 0.0010362, -0.0060046, 0.0014223, -0.0069769, 0.0070408
6: -0.0068887, -0.0046656, -0.0073003, -0.0045620, -0.0023268, 0.0026347
7: -0.0055162, 0.0007463, -0.0061686, 0.0012506, -0.0067668, 0.0069149
8: -0.0081030, -0.0013371, -0.0099276, -0.0013239, -0.0067791, 0.0085904
9: 0.9938440, 1.0123242, 0.9921345, 1.0125878, -0.0187438, 0.0201897

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136063, upper bound: 0.0136861
time: 1.60 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136063, upper bound: 0.0138049
time: 1.53 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0053727, 0.0012404, -0.0058413, 0.0026940, -0.0080667, 0.0070817
1: -0.0029929, 0.0113409, -0.0037383, 0.0128598, -0.0158527, 0.0150793
2: 0.0050854, 0.0205099, 0.0046546, 0.0233276, -0.0175616, 0.0153102
3: -0.0070639, -0.0018734, -0.0081612, -0.0017074, -0.0053565, 0.0062878
4: 0.0033020, 0.0077088, 0.0029118, 0.0079228, -0.0045801, 0.0047963
5: -0.0055546, 0.0010362, -0.0062460, 0.0014883, -0.0070429, 0.0072822
6: -0.0068887, -0.0046656, -0.0073179, -0.0045336, -0.0023551, 0.0026523
7: -0.0055162, 0.0007463, -0.0062339, 0.0013388, -0.0068550, 0.0069802
8: -0.0081030, -0.0013371, -0.0099755, -0.0013075, -0.0067955, 0.0086384
9: 0.9938440, 1.0123242, 0.9919965, 1.0132725, -0.0194286, 0.0203277

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136063, upper bound: 0.0137432
time: 2.21 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136063, upper bound: 0.0138772
time: 2.38 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0056013, 0.0020201, -0.0055506, 0.0016288, -0.0072301, 0.0075706
1: -0.0032511, 0.0122105, -0.0032378, 0.0116753, -0.0149265, 0.0154484
2: 0.0049314, 0.0220756, 0.0050699, 0.0212018, -0.0158276, 0.0164998
3: -0.0076040, -0.0018496, -0.0074756, -0.0018362, -0.0057678, 0.0056260
4: 0.0031385, 0.0077305, 0.0028859, 0.0077109, -0.0045724, 0.0048447
5: -0.0057961, 0.0012136, -0.0057254, 0.0014994, -0.0072954, 0.0069390
6: -0.0071121, -0.0046135, -0.0070200, -0.0044159, -0.0026962, 0.0024065
7: -0.0059255, 0.0010532, -0.0056909, 0.0010873, -0.0070128, 0.0067441
8: -0.0091426, -0.0013279, -0.0085474, -0.0012354, -0.0079072, 0.0072195
9: 0.9929135, 1.0125357, 0.9930466, 1.0125208, -0.0196073, 0.0194890

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136843, upper bound: 0.0136861
time: 2.62 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136843, upper bound: 0.0137432
time: 1.67 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0056013, 0.0020201, -0.0057894, 0.0024363, -0.0080376, 0.0078094
1: -0.0032511, 0.0122105, -0.0035176, 0.0125586, -0.0158098, 0.0157281
2: 0.0049314, 0.0220756, 0.0049159, 0.0228058, -0.0167110, 0.0159147
3: -0.0076040, -0.0018496, -0.0080411, -0.0018123, -0.0057918, 0.0061915
4: 0.0031385, 0.0077305, 0.0027203, 0.0077327, -0.0043799, 0.0048649
5: -0.0057961, 0.0012136, -0.0059986, 0.0016836, -0.0074797, 0.0072122
6: -0.0071121, -0.0046135, -0.0072517, -0.0043709, -0.0027412, 0.0026381
7: -0.0059255, 0.0010532, -0.0060874, 0.0014033, -0.0073288, 0.0071406
8: -0.0091426, -0.0013279, -0.0096148, -0.0012273, -0.0079152, 0.0082870
9: 0.9929135, 1.0125357, 0.9920993, 1.0127498, -0.0198363, 0.0204364

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136843, upper bound: 0.0136861
time: 2.48 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136843, upper bound: 0.0137432
time: 2.00 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0053793, 0.0012629, -0.0055192, 0.0015599, -0.0069393, 0.0067820
1: -0.0033896, 0.0113412, -0.0031936, 0.0116136, -0.0150032, 0.0145347
2: 0.0048375, 0.0205157, 0.0050733, 0.0210763, -0.0154164, 0.0147015
3: -0.0070845, -0.0017411, -0.0074047, -0.0018427, -0.0052418, 0.0056636
4: 0.0032438, 0.0078796, 0.0029611, 0.0077104, -0.0043674, 0.0048229
5: -0.0058458, 0.0010971, -0.0056945, 0.0014187, -0.0072645, 0.0067916
6: -0.0068926, -0.0046262, -0.0069969, -0.0044625, -0.0024300, 0.0023708
7: -0.0055968, 0.0008190, -0.0056558, 0.0010255, -0.0066223, 0.0064748
8: -0.0081091, -0.0013180, -0.0084662, -0.0012514, -0.0068577, 0.0071482
9: 0.9937469, 1.0129554, 0.9931909, 1.0124853, -0.0187383, 0.0197645

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137381, upper bound: 0.0136056
time: 2.66 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137381, upper bound: 0.0136056
time: 2.18 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0053793, 0.0012629, -0.0057629, 0.0023716, -0.0077509, 0.0070257
1: -0.0033896, 0.0113412, -0.0034691, 0.0125005, -0.0158901, 0.0148103
2: 0.0048375, 0.0205157, 0.0049194, 0.0226885, -0.0173755, 0.0151762
3: -0.0070845, -0.0017411, -0.0079767, -0.0018190, -0.0052656, 0.0062356
4: 0.0032438, 0.0078796, 0.0027913, 0.0077322, -0.0044884, 0.0050883
5: -0.0058458, 0.0010971, -0.0059646, 0.0016079, -0.0074537, 0.0070617
6: -0.0068926, -0.0046262, -0.0072306, -0.0044168, -0.0024758, 0.0026045
7: -0.0055968, 0.0008190, -0.0060544, 0.0013404, -0.0069373, 0.0068734
8: -0.0081091, -0.0013180, -0.0095390, -0.0012438, -0.0068653, 0.0082210
9: 0.9937469, 1.0129554, 0.9922287, 1.0127106, -0.0189636, 0.0207267

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0130808
time: 2.13 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129258, upper bound: 0.0130629
time: 1.75 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0056385, 0.0020906, -0.0055192, 0.0015599, -0.0071984, 0.0076098
1: -0.0036719, 0.0122439, -0.0031936, 0.0116136, -0.0152855, 0.0154374
2: 0.0046625, 0.0221690, 0.0050733, 0.0210763, -0.0159281, 0.0166847
3: -0.0076904, -0.0017223, -0.0074047, -0.0018427, -0.0058477, 0.0056824
4: 0.0030661, 0.0079214, 0.0029611, 0.0077104, -0.0046444, 0.0049603
5: -0.0060600, 0.0012990, -0.0056945, 0.0014187, -0.0074786, 0.0069935
6: -0.0071367, -0.0045795, -0.0069969, -0.0044625, -0.0026741, 0.0024175
7: -0.0059889, 0.0011385, -0.0056558, 0.0010255, -0.0070144, 0.0067943
8: -0.0092090, -0.0013116, -0.0084662, -0.0012514, -0.0079576, 0.0071545
9: 0.9927459, 1.0132235, 0.9931909, 1.0124853, -0.0197394, 0.0200326

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138676, upper bound: 0.0136056
time: 2.54 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138676, upper bound: 0.0136056
time: 2.13 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0056385, 0.0020906, -0.0057629, 0.0023716, -0.0080101, 0.0078535
1: -0.0036719, 0.0122439, -0.0034691, 0.0125005, -0.0161724, 0.0157130
2: 0.0046625, 0.0221690, 0.0049194, 0.0226885, -0.0167918, 0.0161069
3: -0.0076904, -0.0017223, -0.0079767, -0.0018190, -0.0058715, 0.0062544
4: 0.0030661, 0.0079214, 0.0027913, 0.0077322, -0.0044951, 0.0049594
5: -0.0060600, 0.0012990, -0.0059646, 0.0016079, -0.0076679, 0.0072636
6: -0.0071367, -0.0045795, -0.0072306, -0.0044168, -0.0027199, 0.0026512
7: -0.0059889, 0.0011385, -0.0060544, 0.0013404, -0.0073293, 0.0071929
8: -0.0092090, -0.0013116, -0.0095390, -0.0012438, -0.0079651, 0.0082274
9: 0.9927459, 1.0132235, 0.9922287, 1.0127106, -0.0199647, 0.0209948

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138676, upper bound: 0.0136056
time: 1.99 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138676, upper bound: 0.0136056
time: 2.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.91 seconds
NS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0136063, upper bound: 0.0136861
NS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0136063, upper bound: 0.0138049
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0136063, upper bound: 0.0137432
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0136063, upper bound: 0.0138772
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0136843, upper bound: 0.0136861
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0136843, upper bound: 0.0137432
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0136843, upper bound: 0.0136861
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0136843, upper bound: 0.0137432
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0137381, upper bound: 0.0136056
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0137381, upper bound: 0.0136056
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0130808
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0129258, upper bound: 0.0130629
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0138676, upper bound: 0.0136056
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0138676, upper bound: 0.0136056
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0138676, upper bound: 0.0136056
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 9, lower bound: -0.0138676, upper bound: 0.0136056

## BFS NS instance: NS_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0053727, 0.0012404, -0.0053727, 0.0012404, -0.0066131, 0.0066131
1: -0.0029929, 0.0113409, -0.0029929, 0.0113409, -0.0143338, 0.0143338
2: 0.0050854, 0.0205099, 0.0050854, 0.0205099, -0.0145855, 0.0145855
3: -0.0070639, -0.0018734, -0.0070639, -0.0018734, -0.0051905, 0.0051905
4: 0.0033020, 0.0077088, 0.0033020, 0.0077088, -0.0042366, 0.0042366
5: -0.0055546, 0.0010362, -0.0055546, 0.0010362, -0.0065908, 0.0065908
6: -0.0068887, -0.0046656, -0.0068887, -0.0046656, -0.0022231, 0.0022231
7: -0.0055162, 0.0007463, -0.0055162, 0.0007463, -0.0062625, 0.0062625
8: -0.0081030, -0.0013371, -0.0081030, -0.0013371, -0.0067659, 0.0067659
9: 0.9938440, 1.0123242, 0.9938440, 1.0123242, -0.0184802, 0.0184802

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_A1_B1_B1_A1

### Relational analysis result of NS_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132051, upper bound: 0.0132989
time: 1.38 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2

### Relational analysis result of NS_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132453, upper bound: 0.0132453
time: 1.49 seconds

## BFS NS instance: NS_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0053727, 0.0012404, -0.0056013, 0.0020201, -0.0073928, 0.0068417
1: -0.0029929, 0.0113409, -0.0032511, 0.0122105, -0.0152034, 0.0145921
2: 0.0050854, 0.0205099, 0.0049314, 0.0220756, -0.0164852, 0.0150596
3: -0.0070639, -0.0018734, -0.0076040, -0.0018496, -0.0052143, 0.0057306
4: 0.0033020, 0.0077088, 0.0031385, 0.0077305, -0.0043926, 0.0045702
5: -0.0055546, 0.0010362, -0.0057961, 0.0012136, -0.0067682, 0.0068323
6: -0.0068887, -0.0046656, -0.0071121, -0.0046135, -0.0022752, 0.0024465
7: -0.0055162, 0.0007463, -0.0059255, 0.0010532, -0.0065694, 0.0066718
8: -0.0081030, -0.0013371, -0.0091426, -0.0013279, -0.0067751, 0.0078054
9: 0.9938440, 1.0123242, 0.9929135, 1.0125357, -0.0186917, 0.0194107

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132989, upper bound: 0.0132994
time: 2.36 seconds

## Relational analysis of NS_A1_A1_B1_B2_B2

### Relational analysis result of NS_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132453, upper bound: 0.0133663
time: 1.52 seconds

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0053727, 0.0012404, -0.0053793, 0.0012629, -0.0066356, 0.0066198
1: -0.0029929, 0.0113409, -0.0033896, 0.0113412, -0.0143340, 0.0147305
2: 0.0050854, 0.0205099, 0.0048375, 0.0205157, -0.0146427, 0.0147896
3: -0.0070639, -0.0018734, -0.0070845, -0.0017411, -0.0053228, 0.0052111
4: 0.0033020, 0.0077088, 0.0032438, 0.0078796, -0.0044056, 0.0043586
5: -0.0055546, 0.0010362, -0.0058458, 0.0010971, -0.0066517, 0.0068820
6: -0.0068887, -0.0046656, -0.0068926, -0.0046262, -0.0022626, 0.0022270
7: -0.0055162, 0.0007463, -0.0055968, 0.0008190, -0.0063352, 0.0063431
8: -0.0081030, -0.0013371, -0.0081091, -0.0013180, -0.0067850, 0.0067720
9: 0.9938440, 1.0123242, 0.9937469, 1.0129554, -0.0191115, 0.0185773

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_A1_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128884, upper bound: 0.0132073
time: 1.95 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128779, upper bound: 0.0130919
time: 2.19 seconds

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0053727, 0.0012404, -0.0056385, 0.0020906, -0.0074633, 0.0068789
1: -0.0029929, 0.0113409, -0.0036719, 0.0122439, -0.0152367, 0.0150128
2: 0.0050854, 0.0205099, 0.0046625, 0.0221690, -0.0166465, 0.0153014
3: -0.0070639, -0.0018734, -0.0076904, -0.0017223, -0.0053416, 0.0058171
4: 0.0033020, 0.0077088, 0.0030661, 0.0079214, -0.0045785, 0.0046427
5: -0.0055546, 0.0010362, -0.0060600, 0.0012990, -0.0068536, 0.0070962
6: -0.0068887, -0.0046656, -0.0071367, -0.0045795, -0.0023093, 0.0024710
7: -0.0055162, 0.0007463, -0.0059889, 0.0011385, -0.0066547, 0.0067352
8: -0.0081030, -0.0013371, -0.0092090, -0.0013116, -0.0067913, 0.0078718
9: 0.9938440, 1.0123242, 0.9927459, 1.0132235, -0.0193796, 0.0195783

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131130, upper bound: 0.0132374
time: 2.24 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128779, upper bound: 0.0132612
time: 1.41 seconds

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0056013, 0.0020201, -0.0053727, 0.0012404, -0.0068417, 0.0073928
1: -0.0032511, 0.0122105, -0.0029929, 0.0113409, -0.0145921, 0.0152034
2: 0.0049314, 0.0220756, 0.0050854, 0.0205099, -0.0150596, 0.0164852
3: -0.0076040, -0.0018496, -0.0070639, -0.0018734, -0.0057306, 0.0052143
4: 0.0031385, 0.0077305, 0.0033020, 0.0077088, -0.0045702, 0.0043926
5: -0.0057961, 0.0012136, -0.0055546, 0.0010362, -0.0068323, 0.0067682
6: -0.0071121, -0.0046135, -0.0068887, -0.0046656, -0.0024465, 0.0022752
7: -0.0059255, 0.0010532, -0.0055162, 0.0007463, -0.0066718, 0.0065694
8: -0.0091426, -0.0013279, -0.0081030, -0.0013371, -0.0078054, 0.0067751
9: 0.9929135, 1.0125357, 0.9938440, 1.0123242, -0.0194107, 0.0186917

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_A2_B1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129546, upper bound: 0.0132035
time: 1.94 seconds

## Relational analysis of NS_A1_A2_B1_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129784, upper bound: 0.0130919
time: 1.47 seconds

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0056013, 0.0020201, -0.0053793, 0.0012629, -0.0068642, 0.0073994
1: -0.0032511, 0.0122105, -0.0033896, 0.0113412, -0.0145923, 0.0156001
2: 0.0049314, 0.0220756, 0.0048375, 0.0205157, -0.0151168, 0.0166894
3: -0.0076040, -0.0018496, -0.0070845, -0.0017411, -0.0058629, 0.0052349
4: 0.0031385, 0.0077305, 0.0032438, 0.0078796, -0.0047410, 0.0044867
5: -0.0057961, 0.0012136, -0.0058458, 0.0010971, -0.0068931, 0.0070594
6: -0.0071121, -0.0046135, -0.0068926, -0.0046262, -0.0024860, 0.0022791
7: -0.0059255, 0.0010532, -0.0055968, 0.0008190, -0.0067445, 0.0066500
8: -0.0091426, -0.0013279, -0.0081091, -0.0013180, -0.0078246, 0.0067813
9: 0.9929135, 1.0125357, 0.9937469, 1.0129554, -0.0200419, 0.0187888

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129546, upper bound: 0.0132073
time: 2.64 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129784, upper bound: 0.0130919
time: 1.49 seconds

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0056013, 0.0020201, -0.0056013, 0.0020201, -0.0076214, 0.0076214
1: -0.0032511, 0.0122105, -0.0032511, 0.0122105, -0.0154617, 0.0154617
2: 0.0049314, 0.0220756, 0.0049314, 0.0220756, -0.0159002, 0.0159002
3: -0.0076040, -0.0018496, -0.0076040, -0.0018496, -0.0057544, 0.0057544
4: 0.0031385, 0.0077305, 0.0031385, 0.0077305, -0.0043742, 0.0043742
5: -0.0057961, 0.0012136, -0.0057961, 0.0012136, -0.0070097, 0.0070097
6: -0.0071121, -0.0046135, -0.0071121, -0.0046135, -0.0024986, 0.0024986
7: -0.0059255, 0.0010532, -0.0059255, 0.0010532, -0.0069787, 0.0069787
8: -0.0091426, -0.0013279, -0.0091426, -0.0013279, -0.0078147, 0.0078147
9: 0.9929135, 1.0125357, 0.9929135, 1.0125357, -0.0196222, 0.0196222

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129561, upper bound: 0.0132035
time: 2.91 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129805, upper bound: 0.0130981
time: 1.55 seconds

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0056013, 0.0020201, -0.0056385, 0.0020906, -0.0076919, 0.0076586
1: -0.0032511, 0.0122105, -0.0036719, 0.0122439, -0.0154950, 0.0158824
2: 0.0049314, 0.0220756, 0.0046625, 0.0221690, -0.0160461, 0.0161109
3: -0.0076040, -0.0018496, -0.0076904, -0.0017223, -0.0058817, 0.0058409
4: 0.0031385, 0.0077305, 0.0030661, 0.0079214, -0.0045486, 0.0044708
5: -0.0057961, 0.0012136, -0.0060600, 0.0012990, -0.0070950, 0.0072736
6: -0.0071121, -0.0046135, -0.0071367, -0.0045795, -0.0025327, 0.0025231
7: -0.0059255, 0.0010532, -0.0059889, 0.0011385, -0.0070640, 0.0070421
8: -0.0091426, -0.0013279, -0.0092090, -0.0013116, -0.0078309, 0.0078811
9: 0.9929135, 1.0125357, 0.9927459, 1.0132235, -0.0203100, 0.0197898

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129561, upper bound: 0.0132073
time: 1.82 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129805, upper bound: 0.0130981
time: 1.83 seconds

## BFS NS instance: NS_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0053793, 0.0012629, -0.0053727, 0.0012404, -0.0066198, 0.0066356
1: -0.0033896, 0.0113412, -0.0029929, 0.0113409, -0.0147305, 0.0143340
2: 0.0048375, 0.0205157, 0.0050854, 0.0205099, -0.0147896, 0.0146427
3: -0.0070845, -0.0017411, -0.0070639, -0.0018734, -0.0052111, 0.0053228
4: 0.0032438, 0.0078796, 0.0033020, 0.0077088, -0.0043586, 0.0044056
5: -0.0058458, 0.0010971, -0.0055546, 0.0010362, -0.0068820, 0.0066517
6: -0.0068926, -0.0046262, -0.0068887, -0.0046656, -0.0022270, 0.0022626
7: -0.0055968, 0.0008190, -0.0055162, 0.0007463, -0.0063431, 0.0063352
8: -0.0081091, -0.0013180, -0.0081030, -0.0013371, -0.0067720, 0.0067850
9: 0.9937469, 1.0129554, 0.9938440, 1.0123242, -0.0185773, 0.0191115

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_A1_B1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0128884
time: 2.34 seconds

## Relational analysis of NS_A2_A1_B1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129258, upper bound: 0.0128770
time: 2.45 seconds

## BFS NS instance: NS_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0053793, 0.0012629, -0.0053793, 0.0012629, -0.0066422, 0.0066422
1: -0.0033896, 0.0113412, -0.0033896, 0.0113412, -0.0147307, 0.0147307
2: 0.0048375, 0.0205157, 0.0048375, 0.0205157, -0.0147451, 0.0147451
3: -0.0070845, -0.0017411, -0.0070845, -0.0017411, -0.0053319, 0.0053319
4: 0.0032438, 0.0078796, 0.0032438, 0.0078796, -0.0043662, 0.0043662
5: -0.0058458, 0.0010971, -0.0058458, 0.0010971, -0.0069429, 0.0069429
6: -0.0068926, -0.0046262, -0.0068926, -0.0046262, -0.0022664, 0.0022664
7: -0.0055968, 0.0008190, -0.0055968, 0.0008190, -0.0064158, 0.0064158
8: -0.0081091, -0.0013180, -0.0081091, -0.0013180, -0.0067911, 0.0067911
9: 0.9937469, 1.0129554, 0.9937469, 1.0129554, -0.0192085, 0.0192085

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_A1_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129789, upper bound: 0.0131125
time: 2.25 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129258, upper bound: 0.0128770
time: 1.74 seconds

## BFS NS instance: NS_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0053793, 0.0012629, -0.0055482, 0.0017760, -0.0071553, 0.0068111
1: -0.0033896, 0.0113412, -0.0033779, 0.0119042, -0.0152938, 0.0147191
2: 0.0048375, 0.0205157, 0.0049326, 0.0215528, -0.0162413, 0.0151586
3: -0.0070845, -0.0017411, -0.0074818, -0.0018331, -0.0052514, 0.0057407
4: 0.0032438, 0.0078796, 0.0029542, 0.0077294, -0.0044856, 0.0049253
5: -0.0058458, 0.0010971, -0.0057938, 0.0013822, -0.0072280, 0.0068909
6: -0.0068926, -0.0046262, -0.0070478, -0.0044634, -0.0024292, 0.0024217
7: -0.0055968, 0.0008190, -0.0058800, 0.0011068, -0.0067037, 0.0066989
8: -0.0081091, -0.0013180, -0.0087928, -0.0012884, -0.0068208, 0.0074748
9: 0.9937469, 1.0129554, 0.9929647, 1.0126379, -0.0188909, 0.0199907

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_A1_B2_B1_B1

### Relational analysis result of NS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0129546
time: 1.83 seconds

## Relational analysis of NS_A2_A1_B2_B1_B2

### Relational analysis result of NS_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0129546
time: 2.01 seconds

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0053183, 0.0011056, -0.0054835, 0.0016535, -0.0069718, 0.0065892
1: -0.0033679, 0.0111823, -0.0046749, 0.0117767, -0.0151446, 0.0158572
2: 0.0048411, 0.0202081, 0.0040539, 0.0212892, -0.0162092, 0.0158428
3: -0.0069432, -0.0017449, -0.0073451, -0.0017273, -0.0052159, 0.0056002
4: 0.0032943, 0.0078789, 0.0029132, 0.0080831, -0.0047888, 0.0049656
5: -0.0058163, 0.0010269, -0.0061863, 0.0013711, -0.0071875, 0.0072132
6: -0.0068415, -0.0046418, -0.0069983, -0.0043969, -0.0024446, 0.0023566
7: -0.0055513, 0.0007531, -0.0059260, 0.0012585, -0.0068099, 0.0066791
8: -0.0079094, -0.0013337, -0.0086259, -0.0013141, -0.0065954, 0.0072922
9: 0.9939555, 1.0129389, 0.9930835, 1.0142895, -0.0203340, 0.0198554

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_A1_B2_B2_B1

### Relational analysis result of NS_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129258, upper bound: 0.0129784
time: 1.70 seconds

## Relational analysis of NS_A2_A1_B2_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129258, upper bound: 0.0129784
time: 1.80 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0056385, 0.0020906, -0.0053727, 0.0012404, -0.0068789, 0.0074633
1: -0.0036719, 0.0122439, -0.0029929, 0.0113409, -0.0150128, 0.0152367
2: 0.0046625, 0.0221690, 0.0050854, 0.0205099, -0.0153014, 0.0166465
3: -0.0076904, -0.0017223, -0.0070639, -0.0018734, -0.0058171, 0.0053416
4: 0.0030661, 0.0079214, 0.0033020, 0.0077088, -0.0046427, 0.0045785
5: -0.0060600, 0.0012990, -0.0055546, 0.0010362, -0.0070962, 0.0068536
6: -0.0071367, -0.0045795, -0.0068887, -0.0046656, -0.0024710, 0.0023093
7: -0.0059889, 0.0011385, -0.0055162, 0.0007463, -0.0067352, 0.0066547
8: -0.0092090, -0.0013116, -0.0081030, -0.0013371, -0.0078718, 0.0067913
9: 0.9927459, 1.0132235, 0.9938440, 1.0123242, -0.0195783, 0.0193796

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130808, upper bound: 0.0131125
time: 2.22 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130629, upper bound: 0.0128770
time: 2.20 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0056385, 0.0020906, -0.0053793, 0.0012629, -0.0069014, 0.0074700
1: -0.0036719, 0.0122439, -0.0033896, 0.0113412, -0.0150130, 0.0156334
2: 0.0046625, 0.0221690, 0.0048375, 0.0205157, -0.0152249, 0.0167282
3: -0.0076904, -0.0017223, -0.0070845, -0.0017411, -0.0059493, 0.0053489
4: 0.0030661, 0.0079214, 0.0032438, 0.0078796, -0.0047134, 0.0045217
5: -0.0060600, 0.0012990, -0.0058458, 0.0010971, -0.0071570, 0.0071448
6: -0.0071367, -0.0045795, -0.0068926, -0.0046262, -0.0025105, 0.0023131
7: -0.0059889, 0.0011385, -0.0055968, 0.0008190, -0.0068079, 0.0067353
8: -0.0092090, -0.0013116, -0.0081091, -0.0013180, -0.0078910, 0.0067975
9: 0.9927459, 1.0132235, 0.9937469, 1.0129554, -0.0202096, 0.0194766

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130808, upper bound: 0.0131125
time: 1.79 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130629, upper bound: 0.0128770
time: 1.98 seconds

## BFS NS instance: NS_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0056385, 0.0020906, -0.0056013, 0.0020201, -0.0076586, 0.0076919
1: -0.0036719, 0.0122439, -0.0032511, 0.0122105, -0.0158824, 0.0154950
2: 0.0046625, 0.0221690, 0.0049314, 0.0220756, -0.0161109, 0.0160461
3: -0.0076904, -0.0017223, -0.0076040, -0.0018496, -0.0058409, 0.0058817
4: 0.0030661, 0.0079214, 0.0031385, 0.0077305, -0.0044708, 0.0045486
5: -0.0060600, 0.0012990, -0.0057961, 0.0012136, -0.0072736, 0.0070950
6: -0.0071367, -0.0045795, -0.0071121, -0.0046135, -0.0025231, 0.0025327
7: -0.0059889, 0.0011385, -0.0059255, 0.0010532, -0.0070421, 0.0070640
8: -0.0092090, -0.0013116, -0.0091426, -0.0013279, -0.0078811, 0.0078309
9: 0.9927459, 1.0132235, 0.9929135, 1.0125357, -0.0197898, 0.0203100

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_A2_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133681, upper bound: 0.0129041
time: 1.76 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130676, upper bound: 0.0129091
time: 1.73 seconds

## BFS NS instance: NS_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0056385, 0.0020906, -0.0056385, 0.0020906, -0.0077291, 0.0077291
1: -0.0036719, 0.0122439, -0.0036719, 0.0122439, -0.0159157, 0.0159157
2: 0.0046625, 0.0221690, 0.0046625, 0.0221690, -0.0161503, 0.0161503
3: -0.0076904, -0.0017223, -0.0076904, -0.0017223, -0.0059681, 0.0059681
4: 0.0030661, 0.0079214, 0.0030661, 0.0079214, -0.0044939, 0.0044939
5: -0.0060600, 0.0012990, -0.0060600, 0.0012990, -0.0073589, 0.0073589
6: -0.0071367, -0.0045795, -0.0071367, -0.0045795, -0.0025572, 0.0025572
7: -0.0059889, 0.0011385, -0.0059889, 0.0011385, -0.0071274, 0.0071274
8: -0.0092090, -0.0013116, -0.0092090, -0.0013116, -0.0078973, 0.0078973
9: 0.9927459, 1.0132235, 0.9927459, 1.0132235, -0.0204777, 0.0204777

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130859, upper bound: 0.0131130
time: 2.06 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130676, upper bound: 0.0129092
time: 2.19 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.02 seconds
NS_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0132051, upper bound: 0.0132989
NS_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0132453, upper bound: 0.0132453
NS_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0132989, upper bound: 0.0132994
NS_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0132453, upper bound: 0.0133663
NS_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0128884, upper bound: 0.0132073
NS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0128779, upper bound: 0.0130919
NS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0131130, upper bound: 0.0132374
NS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0128779, upper bound: 0.0132612
NS_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0129546, upper bound: 0.0132035
NS_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0129784, upper bound: 0.0130919
NS_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0129546, upper bound: 0.0132073
NS_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0129784, upper bound: 0.0130919
NS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0129561, upper bound: 0.0132035
NS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0129805, upper bound: 0.0130981
NS_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0129561, upper bound: 0.0132073
NS_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0129805, upper bound: 0.0130981
NS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0128884
NS_A2_A1_B1_B1_B2, status: Status.VERIFIED, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0129258, upper bound: 0.0128770
NS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0129789, upper bound: 0.0131125
NS_A2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0129258, upper bound: 0.0128770
NS_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0129546
NS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0129546
NS_A2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0129258, upper bound: 0.0129784
NS_A2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0129258, upper bound: 0.0129784
NS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0130808, upper bound: 0.0131125
NS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0130629, upper bound: 0.0128770
NS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0130808, upper bound: 0.0131125
NS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0130629, upper bound: 0.0128770
NS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0133681, upper bound: 0.0129041
NS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0130676, upper bound: 0.0129091
NS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0130859, upper bound: 0.0131130
NS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 9, lower bound: -0.0130676, upper bound: 0.0129092

## BFS NS instance: NS_A1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0051745, 0.0007101, -0.0053727, 0.0012404, -0.0064150, 0.0060828
1: -0.0029166, 0.0107924, -0.0029929, 0.0113409, -0.0142575, 0.0137852
2: 0.0051000, 0.0194649, 0.0050854, 0.0205099, -0.0145667, 0.0135260
3: -0.0066025, -0.0018866, -0.0070639, -0.0018734, -0.0047291, 0.0051773
4: 0.0034578, 0.0077059, 0.0033020, 0.0077088, -0.0040525, 0.0042321
5: -0.0054525, 0.0008183, -0.0055546, 0.0010362, -0.0064887, 0.0063729
6: -0.0067205, -0.0047118, -0.0068887, -0.0046656, -0.0020549, 0.0021770
7: -0.0053562, 0.0005414, -0.0055162, 0.0007463, -0.0061025, 0.0060577
8: -0.0074181, -0.0013818, -0.0081030, -0.0013371, -0.0060810, 0.0067212
9: 0.9945275, 1.0122643, 0.9938440, 1.0123242, -0.0177968, 0.0184203

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132038, upper bound: 0.0132038
time: 2.30 seconds

## Relational analysis of NS_A1_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132038, upper bound: 0.0132453
time: 2.49 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0050759, 0.0006935, -0.0053146, 0.0010854, -0.0061613, 0.0060081
1: -0.0041007, 0.0105854, -0.0029719, 0.0111889, -0.0152896, 0.0135573
2: 0.0042883, 0.0190407, 0.0050889, 0.0202149, -0.0152414, 0.0134010
3: -0.0063938, -0.0017833, -0.0069290, -0.0018771, -0.0045167, 0.0051457
4: 0.0034435, 0.0080327, 0.0033489, 0.0077081, -0.0042459, 0.0046101
5: -0.0058576, 0.0007851, -0.0055238, 0.0009717, -0.0068293, 0.0063089
6: -0.0066460, -0.0046634, -0.0068399, -0.0046797, -0.0019662, 0.0021765
7: -0.0053652, 0.0006751, -0.0054716, 0.0006833, -0.0060485, 0.0061466
8: -0.0071422, -0.0014029, -0.0079097, -0.0013518, -0.0057904, 0.0065068
9: 0.9947570, 1.0137668, 0.9940423, 1.0123080, -0.0175510, 0.0197245

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129952, upper bound: 0.0128392
time: 2.15 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_A1_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129986, upper bound: 0.0129986
time: 2.10 seconds

## BFS NS instance: NS_A1_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0053727, 0.0012404, -0.0053825, 0.0014202, -0.0067929, 0.0066229
1: -0.0029929, 0.0113409, -0.0031722, 0.0116048, -0.0145976, 0.0145131
2: 0.0050854, 0.0205099, 0.0049452, 0.0209254, -0.0153386, 0.0150414
3: -0.0070639, -0.0018734, -0.0071040, -0.0018634, -0.0052005, 0.0052306
4: 0.0033020, 0.0077088, 0.0033040, 0.0077276, -0.0043881, 0.0044048
5: -0.0055546, 0.0010362, -0.0056359, 0.0009839, -0.0065385, 0.0066722
6: -0.0068887, -0.0046656, -0.0069272, -0.0046625, -0.0022263, 0.0022616
7: -0.0055162, 0.0007463, -0.0057477, 0.0008325, -0.0063488, 0.0064940
8: -0.0081030, -0.0013371, -0.0083890, -0.0013721, -0.0067309, 0.0070518
9: 0.9938440, 1.0123242, 0.9936654, 1.0124726, -0.0186287, 0.0186588

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_A1_B1_B2_B1_A1

### Relational analysis result of NS_A1_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132038, upper bound: 0.0132994
time: 2.10 seconds

## Relational analysis of NS_A1_A1_B1_B2_B1_A2

### Relational analysis result of NS_A1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132038, upper bound: 0.0132994
time: 2.03 seconds

## BFS NS instance: NS_A1_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0053146, 0.0010854, -0.0053300, 0.0013300, -0.0066447, 0.0064154
1: -0.0029719, 0.0111889, -0.0044856, 0.0114998, -0.0144717, 0.0156745
2: 0.0050889, 0.0202149, 0.0040666, 0.0207120, -0.0153530, 0.0157378
3: -0.0069290, -0.0018771, -0.0069993, -0.0017582, -0.0051708, 0.0051222
4: 0.0033489, 0.0077081, 0.0032570, 0.0080813, -0.0047324, 0.0044511
5: -0.0055238, 0.0009717, -0.0060553, 0.0009895, -0.0065133, 0.0070270
6: -0.0068399, -0.0046797, -0.0068886, -0.0046063, -0.0022335, 0.0022089
7: -0.0054716, 0.0006833, -0.0057891, 0.0010180, -0.0064895, 0.0064724
8: -0.0079097, -0.0013518, -0.0082554, -0.0013952, -0.0065145, 0.0069035
9: 0.9940423, 1.0123080, 0.9937360, 1.0141370, -0.0200948, 0.0185720

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B1_B2_B2_A1

### Relational analysis result of NS_A1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128392, upper bound: 0.0131226
time: 2.11 seconds

## Relational analysis of NS_A1_A1_B1_B2_B2_A2

### Relational analysis result of NS_A1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129986, upper bound: 0.0131238
time: 2.01 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0051745, 0.0007101, -0.0053793, 0.0012629, -0.0064374, 0.0060894
1: -0.0029166, 0.0107924, -0.0033896, 0.0113412, -0.0142577, 0.0141820
2: 0.0051000, 0.0194649, 0.0048375, 0.0205157, -0.0146239, 0.0137301
3: -0.0066025, -0.0018866, -0.0070845, -0.0017411, -0.0048614, 0.0051979
4: 0.0034578, 0.0077059, 0.0032438, 0.0078796, -0.0042215, 0.0043541
5: -0.0054525, 0.0008183, -0.0058458, 0.0010971, -0.0065495, 0.0066641
6: -0.0067205, -0.0047118, -0.0068926, -0.0046262, -0.0020943, 0.0021808
7: -0.0053562, 0.0005414, -0.0055968, 0.0008190, -0.0061752, 0.0061383
8: -0.0074181, -0.0013818, -0.0081091, -0.0013180, -0.0061001, 0.0067274
9: 0.9945275, 1.0122643, 0.9937469, 1.0129554, -0.0184280, 0.0185173

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128723, upper bound: 0.0130869
time: 1.66 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128723, upper bound: 0.0130919
time: 1.81 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0050759, 0.0006935, -0.0053183, 0.0011056, -0.0061815, 0.0060118
1: -0.0041007, 0.0105854, -0.0033679, 0.0111823, -0.0152830, 0.0139533
2: 0.0042883, 0.0190407, 0.0048411, 0.0202081, -0.0152765, 0.0136050
3: -0.0063938, -0.0017833, -0.0069432, -0.0017449, -0.0046489, 0.0051599
4: 0.0034435, 0.0080327, 0.0032943, 0.0078789, -0.0044150, 0.0047231
5: -0.0058576, 0.0007851, -0.0058163, 0.0010269, -0.0068844, 0.0066015
6: -0.0066460, -0.0046634, -0.0068415, -0.0046418, -0.0020042, 0.0021781
7: -0.0053652, 0.0006751, -0.0055513, 0.0007531, -0.0061183, 0.0062264
8: -0.0071422, -0.0014029, -0.0079094, -0.0013337, -0.0058085, 0.0065065
9: 0.9947570, 1.0137668, 0.9939555, 1.0129389, -0.0181819, 0.0198113

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126269, upper bound: 0.0126916
time: 2.01 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126400, upper bound: 0.0128438
time: 1.74 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0053727, 0.0012404, -0.0054251, 0.0015051, -0.0068778, 0.0066655
1: -0.0029929, 0.0113409, -0.0035944, 0.0116515, -0.0146443, 0.0149354
2: 0.0050854, 0.0205099, 0.0046760, 0.0210417, -0.0155325, 0.0152836
3: -0.0070639, -0.0018734, -0.0072004, -0.0017358, -0.0053281, 0.0053271
4: 0.0033020, 0.0077088, 0.0032311, 0.0079185, -0.0045743, 0.0044777
5: -0.0055546, 0.0010362, -0.0059257, 0.0010752, -0.0066298, 0.0069619
6: -0.0068887, -0.0046656, -0.0069546, -0.0046292, -0.0022595, 0.0022890
7: -0.0055162, 0.0007463, -0.0058100, 0.0009171, -0.0064334, 0.0065563
8: -0.0081030, -0.0013371, -0.0084685, -0.0013540, -0.0067490, 0.0071314
9: 0.9938440, 1.0123242, 0.9934891, 1.0131640, -0.0193201, 0.0188351

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128723, upper bound: 0.0132354
time: 1.94 seconds

## Relational analysis of NS_A1_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128723, upper bound: 0.0132354
time: 1.88 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0053146, 0.0010854, -0.0053490, 0.0013783, -0.0066930, 0.0064344
1: -0.0029719, 0.0111889, -0.0048444, 0.0114943, -0.0144663, 0.0160333
2: 0.0050889, 0.0202149, 0.0038267, 0.0207236, -0.0154497, 0.0159474
3: -0.0069290, -0.0018771, -0.0070322, -0.0016424, -0.0052866, 0.0051551
4: 0.0033489, 0.0077081, 0.0031972, 0.0082527, -0.0049038, 0.0045109
5: -0.0055238, 0.0009717, -0.0063181, 0.0010545, -0.0065783, 0.0072898
6: -0.0068399, -0.0046797, -0.0068964, -0.0045752, -0.0022647, 0.0022167
7: -0.0054716, 0.0006833, -0.0058517, 0.0010632, -0.0065348, 0.0065350
8: -0.0079097, -0.0013518, -0.0082703, -0.0013857, -0.0065240, 0.0069185
9: 0.9940423, 1.0123080, 0.9936355, 1.0147262, -0.0206839, 0.0186725

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124713, upper bound: 0.0130064
time: 1.49 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126400, upper bound: 0.0130096
time: 1.94 seconds

## BFS NS instance: NS_A1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0053825, 0.0014202, -0.0053727, 0.0012404, -0.0066229, 0.0067929
1: -0.0031722, 0.0116048, -0.0029929, 0.0113409, -0.0145131, 0.0145976
2: 0.0049452, 0.0209254, 0.0050854, 0.0205099, -0.0150414, 0.0153386
3: -0.0071040, -0.0018634, -0.0070639, -0.0018734, -0.0052306, 0.0052005
4: 0.0033040, 0.0077276, 0.0033020, 0.0077088, -0.0044048, 0.0043881
5: -0.0056359, 0.0009839, -0.0055546, 0.0010362, -0.0066722, 0.0065385
6: -0.0069272, -0.0046625, -0.0068887, -0.0046656, -0.0022616, 0.0022263
7: -0.0057477, 0.0008325, -0.0055162, 0.0007463, -0.0064940, 0.0063488
8: -0.0083890, -0.0013721, -0.0081030, -0.0013371, -0.0070518, 0.0067309
9: 0.9936654, 1.0124726, 0.9938440, 1.0123242, -0.0186588, 0.0186287

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_A2_B1_B1_A1_B1

### Relational analysis result of NS_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132994, upper bound: 0.0132038
time: 2.14 seconds

## Relational analysis of NS_A1_A2_B1_B1_A1_B2

### Relational analysis result of NS_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132994, upper bound: 0.0132453
time: 1.90 seconds

## BFS NS instance: NS_A1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0053300, 0.0013300, -0.0053146, 0.0010854, -0.0064154, 0.0066447
1: -0.0044856, 0.0114998, -0.0029719, 0.0111889, -0.0156745, 0.0144717
2: 0.0040666, 0.0207120, 0.0050889, 0.0202149, -0.0157378, 0.0153530
3: -0.0069993, -0.0017582, -0.0069290, -0.0018771, -0.0051222, 0.0051708
4: 0.0032570, 0.0080813, 0.0033489, 0.0077081, -0.0044511, 0.0047324
5: -0.0060553, 0.0009895, -0.0055238, 0.0009717, -0.0070270, 0.0065133
6: -0.0068886, -0.0046063, -0.0068399, -0.0046797, -0.0022089, 0.0022335
7: -0.0057891, 0.0010180, -0.0054716, 0.0006833, -0.0064724, 0.0064895
8: -0.0082554, -0.0013952, -0.0079097, -0.0013518, -0.0069035, 0.0065145
9: 0.9937360, 1.0141370, 0.9940423, 1.0123080, -0.0185720, 0.0200948

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B1_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131226, upper bound: 0.0128392
time: 1.64 seconds

## Relational analysis of NS_A1_A2_B1_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131238, upper bound: 0.0129986
time: 2.35 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0053825, 0.0014202, -0.0053793, 0.0012629, -0.0066454, 0.0067995
1: -0.0031722, 0.0116048, -0.0033896, 0.0113412, -0.0145134, 0.0149943
2: 0.0049452, 0.0209254, 0.0048375, 0.0205157, -0.0150986, 0.0155427
3: -0.0071040, -0.0018634, -0.0070845, -0.0017411, -0.0053629, 0.0052211
4: 0.0033040, 0.0077276, 0.0032438, 0.0078796, -0.0045756, 0.0044838
5: -0.0056359, 0.0009839, -0.0058458, 0.0010971, -0.0067330, 0.0068297
6: -0.0069272, -0.0046625, -0.0068926, -0.0046262, -0.0023010, 0.0022301
7: -0.0057477, 0.0008325, -0.0055968, 0.0008190, -0.0065667, 0.0064293
8: -0.0083890, -0.0013721, -0.0081091, -0.0013180, -0.0070710, 0.0067370
9: 0.9936654, 1.0124726, 0.9937469, 1.0129554, -0.0192900, 0.0187257

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_A2_B1_B2_A1_B1

### Relational analysis result of NS_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129538, upper bound: 0.0130869
time: 1.98 seconds

## Relational analysis of NS_A1_A2_B1_B2_A1_B2

### Relational analysis result of NS_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129538, upper bound: 0.0130919
time: 1.94 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0053300, 0.0013300, -0.0053183, 0.0011056, -0.0064357, 0.0066483
1: -0.0044856, 0.0114998, -0.0033679, 0.0111823, -0.0156679, 0.0148677
2: 0.0040666, 0.0207120, 0.0048411, 0.0202081, -0.0157728, 0.0155570
3: -0.0069993, -0.0017582, -0.0069432, -0.0017449, -0.0052545, 0.0051850
4: 0.0032570, 0.0080813, 0.0032943, 0.0078789, -0.0046218, 0.0047871
5: -0.0060553, 0.0009895, -0.0058163, 0.0010269, -0.0070822, 0.0068058
6: -0.0068886, -0.0046063, -0.0068415, -0.0046418, -0.0022468, 0.0022352
7: -0.0057891, 0.0010180, -0.0055513, 0.0007531, -0.0065422, 0.0065693
8: -0.0082554, -0.0013952, -0.0079094, -0.0013337, -0.0069217, 0.0065143
9: 0.9937360, 1.0141370, 0.9939555, 1.0129389, -0.0192028, 0.0201815

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B1_B2_A2_B1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127544, upper bound: 0.0126916
time: 1.76 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127608, upper bound: 0.0128437
time: 2.49 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0053825, 0.0014202, -0.0056013, 0.0020201, -0.0074026, 0.0070215
1: -0.0031722, 0.0116048, -0.0032511, 0.0122105, -0.0153827, 0.0148559
2: 0.0049452, 0.0209254, 0.0049314, 0.0220756, -0.0158814, 0.0147596
3: -0.0071040, -0.0018634, -0.0076040, -0.0018496, -0.0052544, 0.0057406
4: 0.0033040, 0.0077276, 0.0031385, 0.0077305, -0.0041870, 0.0043697
5: -0.0056359, 0.0009839, -0.0057961, 0.0012136, -0.0068495, 0.0067800
6: -0.0069272, -0.0046625, -0.0071121, -0.0046135, -0.0023137, 0.0024497
7: -0.0057477, 0.0008325, -0.0059255, 0.0010532, -0.0068009, 0.0067580
8: -0.0083890, -0.0013721, -0.0091426, -0.0013279, -0.0070611, 0.0077705
9: 0.9936654, 1.0124726, 0.9929135, 1.0125357, -0.0188703, 0.0195591

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_A2_B2_B1_A1_B1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132994, upper bound: 0.0132038
time: 2.20 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132994, upper bound: 0.0132453
time: 2.43 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0053300, 0.0013300, -0.0055487, 0.0018739, -0.0072040, 0.0068787
1: -0.0044856, 0.0114998, -0.0032315, 0.0120676, -0.0165532, 0.0147313
2: 0.0040666, 0.0207120, 0.0049348, 0.0218003, -0.0165825, 0.0148231
3: -0.0069993, -0.0017582, -0.0074802, -0.0018532, -0.0051461, 0.0057220
4: 0.0032570, 0.0080813, 0.0031815, 0.0077299, -0.0044115, 0.0047529
5: -0.0060553, 0.0009895, -0.0057542, 0.0011527, -0.0072081, 0.0067437
6: -0.0068886, -0.0046063, -0.0070671, -0.0046273, -0.0022613, 0.0024607
7: -0.0057891, 0.0010180, -0.0058839, 0.0009911, -0.0067801, 0.0069019
8: -0.0082554, -0.0013952, -0.0089621, -0.0013431, -0.0069123, 0.0075669
9: 0.9937360, 1.0141370, 0.9930980, 1.0125202, -0.0187842, 0.0210390

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131226, upper bound: 0.0128432
time: 1.61 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131238, upper bound: 0.0129986
time: 2.29 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0053825, 0.0014202, -0.0056385, 0.0020906, -0.0074731, 0.0070587
1: -0.0031722, 0.0116048, -0.0036719, 0.0122439, -0.0154161, 0.0152766
2: 0.0049452, 0.0209254, 0.0046625, 0.0221690, -0.0160273, 0.0149704
3: -0.0071040, -0.0018634, -0.0076904, -0.0017223, -0.0053817, 0.0058270
4: 0.0033040, 0.0077276, 0.0030661, 0.0079214, -0.0043614, 0.0044663
5: -0.0056359, 0.0009839, -0.0060600, 0.0012990, -0.0069349, 0.0070439
6: -0.0069272, -0.0046625, -0.0071367, -0.0045795, -0.0023477, 0.0024742
7: -0.0057477, 0.0008325, -0.0059889, 0.0011385, -0.0068862, 0.0068214
8: -0.0083890, -0.0013721, -0.0092090, -0.0013116, -0.0070773, 0.0078369
9: 0.9936654, 1.0124726, 0.9927459, 1.0132235, -0.0195581, 0.0197268

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_A2_B2_B2_A1_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129555, upper bound: 0.0130913
time: 1.94 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129555, upper bound: 0.0130981
time: 1.83 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0053300, 0.0013300, -0.0055828, 0.0019398, -0.0072698, 0.0069129
1: -0.0044856, 0.0114998, -0.0036516, 0.0120949, -0.0165805, 0.0151514
2: 0.0040666, 0.0207120, 0.0046660, 0.0218834, -0.0167079, 0.0150336
3: -0.0069993, -0.0017582, -0.0075603, -0.0017259, -0.0052735, 0.0058021
4: 0.0032570, 0.0080813, 0.0031124, 0.0079207, -0.0045859, 0.0048413
5: -0.0060553, 0.0009895, -0.0060233, 0.0012338, -0.0072891, 0.0070128
6: -0.0068886, -0.0046063, -0.0070896, -0.0045943, -0.0022944, 0.0024833
7: -0.0057891, 0.0010180, -0.0059468, 0.0010737, -0.0068628, 0.0069647
8: -0.0082554, -0.0013952, -0.0090217, -0.0013278, -0.0069275, 0.0076265
9: 0.9937360, 1.0141370, 0.9929397, 1.0132080, -0.0194720, 0.0211974

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127575, upper bound: 0.0127111
time: 1.79 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127639, upper bound: 0.0128519
time: 1.89 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0053793, 0.0012629, -0.0051745, 0.0007101, -0.0060894, 0.0064374
1: -0.0033896, 0.0113412, -0.0029166, 0.0107924, -0.0141820, 0.0142577
2: 0.0048375, 0.0205157, 0.0051000, 0.0194649, -0.0137301, 0.0146239
3: -0.0070845, -0.0017411, -0.0066025, -0.0018866, -0.0051979, 0.0048614
4: 0.0032438, 0.0078796, 0.0034578, 0.0077059, -0.0043541, 0.0042215
5: -0.0058458, 0.0010971, -0.0054525, 0.0008183, -0.0066641, 0.0065495
6: -0.0068926, -0.0046262, -0.0067205, -0.0047118, -0.0021808, 0.0020943
7: -0.0055968, 0.0008190, -0.0053562, 0.0005414, -0.0061383, 0.0061752
8: -0.0081091, -0.0013180, -0.0074181, -0.0013818, -0.0067274, 0.0061001
9: 0.9937469, 1.0129554, 0.9945275, 1.0122643, -0.0185173, 0.0184280

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_A1_B1_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130869, upper bound: 0.0128723
time: 2.21 seconds

## Relational analysis of NS_A2_A1_B1_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130869, upper bound: 0.0128723
time: 2.14 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0051840, 0.0007616, -0.0053793, 0.0012629, -0.0064469, 0.0061409
1: -0.0033155, 0.0108001, -0.0033896, 0.0113412, -0.0146567, 0.0141897
2: 0.0048515, 0.0194852, 0.0048375, 0.0205157, -0.0147259, 0.0137070
3: -0.0066333, -0.0017540, -0.0070845, -0.0017411, -0.0048511, 0.0053154
4: 0.0033963, 0.0078766, 0.0032438, 0.0078796, -0.0041822, 0.0043616
5: -0.0057507, 0.0008870, -0.0058458, 0.0010971, -0.0068478, 0.0067328
6: -0.0067269, -0.0046726, -0.0068926, -0.0046262, -0.0021007, 0.0022199
7: -0.0054356, 0.0006124, -0.0055968, 0.0008190, -0.0062546, 0.0062093
8: -0.0074341, -0.0013610, -0.0081091, -0.0013180, -0.0061161, 0.0067482
9: 0.9944275, 1.0128963, 0.9937469, 1.0129554, -0.0185279, 0.0191494

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_A1_B1_B2_A1_B1

### Relational analysis result of NS_A2_A1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129258, upper bound: 0.0128720
time: 2.16 seconds

## Relational analysis of NS_A2_A1_B1_B2_A1_B2

### Relational analysis result of NS_A2_A1_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129258, upper bound: 0.0128770
time: 1.62 seconds

## BFS NS instance: NS_A2_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0053793, 0.0012629, -0.0053825, 0.0014202, -0.0067995, 0.0066454
1: -0.0033896, 0.0113412, -0.0031722, 0.0116048, -0.0149943, 0.0145134
2: 0.0048375, 0.0205157, 0.0049452, 0.0209254, -0.0155427, 0.0150986
3: -0.0070845, -0.0017411, -0.0071040, -0.0018634, -0.0052211, 0.0053629
4: 0.0032438, 0.0078796, 0.0033040, 0.0077276, -0.0044838, 0.0045756
5: -0.0058458, 0.0010971, -0.0056359, 0.0009839, -0.0068297, 0.0067330
6: -0.0068926, -0.0046262, -0.0069272, -0.0046625, -0.0022301, 0.0023010
7: -0.0055968, 0.0008190, -0.0057477, 0.0008325, -0.0064293, 0.0065667
8: -0.0081091, -0.0013180, -0.0083890, -0.0013721, -0.0067370, 0.0070710
9: 0.9937469, 1.0129554, 0.9936654, 1.0124726, -0.0187257, 0.0192900

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_A1_B2_B1_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0129546
time: 2.26 seconds

## Relational analysis of NS_A2_A1_B2_B1_B1_A2

### Relational analysis result of NS_A2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0129546
time: 2.01 seconds

## BFS NS instance: NS_A2_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0053793, 0.0012629, -0.0054251, 0.0015051, -0.0068845, 0.0066880
1: -0.0033896, 0.0113412, -0.0035944, 0.0116515, -0.0150411, 0.0149356
2: 0.0048375, 0.0205157, 0.0046760, 0.0210417, -0.0155972, 0.0152064
3: -0.0070845, -0.0017411, -0.0072004, -0.0017358, -0.0053320, 0.0054593
4: 0.0032438, 0.0078796, 0.0032311, 0.0079185, -0.0045172, 0.0045265
5: -0.0058458, 0.0010971, -0.0059257, 0.0010752, -0.0069210, 0.0070227
6: -0.0068926, -0.0046262, -0.0069546, -0.0046292, -0.0022633, 0.0023285
7: -0.0055968, 0.0008190, -0.0058100, 0.0009171, -0.0065140, 0.0066290
8: -0.0081091, -0.0013180, -0.0084685, -0.0013540, -0.0067551, 0.0071505
9: 0.9937469, 1.0129554, 0.9934891, 1.0131640, -0.0194171, 0.0194663

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_A1_B2_B1_B2_A1

### Relational analysis result of NS_A2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0129546
time: 2.26 seconds

## Relational analysis of NS_A2_A1_B2_B1_B2_A2

### Relational analysis result of NS_A2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0129546
time: 2.36 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0054251, 0.0015051, -0.0053727, 0.0012404, -0.0066655, 0.0068778
1: -0.0035944, 0.0116515, -0.0029929, 0.0113409, -0.0149354, 0.0146443
2: 0.0046760, 0.0210417, 0.0050854, 0.0205099, -0.0152836, 0.0155325
3: -0.0072004, -0.0017358, -0.0070639, -0.0018734, -0.0053271, 0.0053281
4: 0.0032311, 0.0079185, 0.0033020, 0.0077088, -0.0044777, 0.0045743
5: -0.0059257, 0.0010752, -0.0055546, 0.0010362, -0.0069619, 0.0066298
6: -0.0069546, -0.0046292, -0.0068887, -0.0046656, -0.0022890, 0.0022595
7: -0.0058100, 0.0009171, -0.0055162, 0.0007463, -0.0065563, 0.0064334
8: -0.0084685, -0.0013540, -0.0081030, -0.0013371, -0.0071314, 0.0067490
9: 0.9934891, 1.0131640, 0.9938440, 1.0123242, -0.0188351, 0.0193201

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132354, upper bound: 0.0128723
time: 1.78 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132354, upper bound: 0.0128779
time: 1.57 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0053490, 0.0013783, -0.0053146, 0.0010854, -0.0064344, 0.0066930
1: -0.0048444, 0.0114943, -0.0029719, 0.0111889, -0.0160333, 0.0144663
2: 0.0038267, 0.0207236, 0.0050889, 0.0202149, -0.0159474, 0.0154497
3: -0.0070322, -0.0016424, -0.0069290, -0.0018771, -0.0051551, 0.0052866
4: 0.0031972, 0.0082527, 0.0033489, 0.0077081, -0.0045109, 0.0049038
5: -0.0063181, 0.0010545, -0.0055238, 0.0009717, -0.0072898, 0.0065783
6: -0.0068964, -0.0045752, -0.0068399, -0.0046797, -0.0022167, 0.0022647
7: -0.0058517, 0.0010632, -0.0054716, 0.0006833, -0.0065350, 0.0065348
8: -0.0082703, -0.0013857, -0.0079097, -0.0013518, -0.0069185, 0.0065240
9: 0.9936355, 1.0147262, 0.9940423, 1.0123080, -0.0186725, 0.0206839

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130064, upper bound: 0.0124713
time: 1.99 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128342, upper bound: 0.0126400
time: 2.59 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054251, 0.0015051, -0.0053793, 0.0012629, -0.0066880, 0.0068845
1: -0.0035944, 0.0116515, -0.0033896, 0.0113412, -0.0149356, 0.0150411
2: 0.0046760, 0.0210417, 0.0048375, 0.0205157, -0.0152064, 0.0155972
3: -0.0072004, -0.0017358, -0.0070845, -0.0017411, -0.0054593, 0.0053320
4: 0.0032311, 0.0079185, 0.0032438, 0.0078796, -0.0045265, 0.0045172
5: -0.0059257, 0.0010752, -0.0058458, 0.0010971, -0.0070227, 0.0069210
6: -0.0069546, -0.0046292, -0.0068926, -0.0046262, -0.0023285, 0.0022633
7: -0.0058100, 0.0009171, -0.0055968, 0.0008190, -0.0066290, 0.0065140
8: -0.0084685, -0.0013540, -0.0081091, -0.0013180, -0.0071505, 0.0067551
9: 0.9934891, 1.0131640, 0.9937469, 1.0129554, -0.0194663, 0.0194171

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130625, upper bound: 0.0128720
time: 2.22 seconds

## Relational analysis of NS_A2_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130625, upper bound: 0.0128770
time: 2.73 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0053490, 0.0013783, -0.0053183, 0.0011056, -0.0064547, 0.0066966
1: -0.0048444, 0.0114943, -0.0033679, 0.0111823, -0.0160267, 0.0148623
2: 0.0038267, 0.0207236, 0.0048411, 0.0202081, -0.0158877, 0.0155177
3: -0.0070322, -0.0016424, -0.0069432, -0.0017449, -0.0052873, 0.0053008
4: 0.0031972, 0.0082527, 0.0032943, 0.0078789, -0.0046816, 0.0048955
5: -0.0063181, 0.0010545, -0.0058163, 0.0010269, -0.0073450, 0.0068709
6: -0.0068964, -0.0045752, -0.0068415, -0.0046418, -0.0022546, 0.0022664
7: -0.0058517, 0.0010632, -0.0055513, 0.0007531, -0.0066047, 0.0066146
8: -0.0082703, -0.0013857, -0.0079094, -0.0013337, -0.0069366, 0.0065237
9: 0.9936355, 1.0147262, 0.9939555, 1.0129389, -0.0193033, 0.0207707

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128324, upper bound: 0.0124703
time: 2.89 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128428, upper bound: 0.0126387
time: 1.68 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0056385, 0.0020906, -0.0053825, 0.0014202, -0.0070587, 0.0074731
1: -0.0036719, 0.0122439, -0.0031722, 0.0116048, -0.0152766, 0.0154161
2: 0.0046625, 0.0221690, 0.0049452, 0.0209254, -0.0149704, 0.0160273
3: -0.0076904, -0.0017223, -0.0071040, -0.0018634, -0.0058270, 0.0053817
4: 0.0030661, 0.0079214, 0.0033040, 0.0077276, -0.0044663, 0.0043614
5: -0.0060600, 0.0012990, -0.0056359, 0.0009839, -0.0070439, 0.0069349
6: -0.0071367, -0.0045795, -0.0069272, -0.0046625, -0.0024742, 0.0023477
7: -0.0059889, 0.0011385, -0.0057477, 0.0008325, -0.0068214, 0.0068862
8: -0.0092090, -0.0013116, -0.0083890, -0.0013721, -0.0078369, 0.0070773
9: 0.9927459, 1.0132235, 0.9936654, 1.0124726, -0.0197268, 0.0195581

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132256, upper bound: 0.0128960
time: 2.12 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132256, upper bound: 0.0128960
time: 2.85 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0055828, 0.0019398, -0.0053300, 0.0013300, -0.0069129, 0.0072698
1: -0.0036516, 0.0120949, -0.0044856, 0.0114998, -0.0151514, 0.0165805
2: 0.0046660, 0.0218834, 0.0040666, 0.0207120, -0.0150336, 0.0167079
3: -0.0075603, -0.0017259, -0.0069993, -0.0017582, -0.0058021, 0.0052735
4: 0.0031124, 0.0079207, 0.0032570, 0.0080813, -0.0048413, 0.0045859
5: -0.0060233, 0.0012338, -0.0060553, 0.0009895, -0.0070128, 0.0072891
6: -0.0070896, -0.0045943, -0.0068886, -0.0046063, -0.0024833, 0.0022944
7: -0.0059468, 0.0010737, -0.0057891, 0.0010180, -0.0069647, 0.0068628
8: -0.0090217, -0.0013278, -0.0082554, -0.0013952, -0.0076265, 0.0069275
9: 0.9929397, 1.0132080, 0.9937360, 1.0141370, -0.0211974, 0.0194720

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128993, upper bound: 0.0126629
time: 2.37 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130096, upper bound: 0.0126761
time: 2.42 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054251, 0.0015051, -0.0056385, 0.0020906, -0.0075157, 0.0071436
1: -0.0035944, 0.0116515, -0.0036719, 0.0122439, -0.0158383, 0.0153234
2: 0.0046760, 0.0210417, 0.0046625, 0.0221690, -0.0161310, 0.0150219
3: -0.0072004, -0.0017358, -0.0076904, -0.0017223, -0.0054781, 0.0059546
4: 0.0032311, 0.0079185, 0.0030661, 0.0079214, -0.0043079, 0.0044893
5: -0.0059257, 0.0010752, -0.0060600, 0.0012990, -0.0072246, 0.0071352
6: -0.0069546, -0.0046292, -0.0071367, -0.0045795, -0.0023752, 0.0025074
7: -0.0058100, 0.0009171, -0.0059889, 0.0011385, -0.0069485, 0.0069060
8: -0.0084685, -0.0013540, -0.0092090, -0.0013116, -0.0071569, 0.0078550
9: 0.9934891, 1.0131640, 0.9927459, 1.0132235, -0.0197344, 0.0204182

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130675, upper bound: 0.0128960
time: 2.37 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130675, upper bound: 0.0129091
time: 2.10 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0053490, 0.0013783, -0.0055828, 0.0019398, -0.0072888, 0.0069612
1: -0.0048444, 0.0114943, -0.0036516, 0.0120949, -0.0169394, 0.0151460
2: 0.0038267, 0.0207236, 0.0046660, 0.0218834, -0.0168153, 0.0149923
3: -0.0070322, -0.0016424, -0.0075603, -0.0017259, -0.0053063, 0.0059179
4: 0.0031972, 0.0082527, 0.0031124, 0.0079207, -0.0045093, 0.0048686
5: -0.0063181, 0.0010545, -0.0060233, 0.0012338, -0.0075519, 0.0070779
6: -0.0068964, -0.0045752, -0.0070896, -0.0045943, -0.0023022, 0.0025144
7: -0.0058517, 0.0010632, -0.0059468, 0.0010737, -0.0069253, 0.0070100
8: -0.0082703, -0.0013857, -0.0090217, -0.0013278, -0.0069425, 0.0076360
9: 0.9936355, 1.0147262, 0.9929397, 1.0132080, -0.0195725, 0.0217865

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128381, upper bound: 0.0125199
time: 1.72 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128489, upper bound: 0.0126759
time: 2.04 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.63 seconds
NS_A1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0132038, upper bound: 0.0132038
NS_A1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0132038, upper bound: 0.0132453
NS_A1_A1_B1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0129952, upper bound: 0.0128392
NS_A1_A1_B1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0129986, upper bound: 0.0129986
NS_A1_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0132038, upper bound: 0.0132994
NS_A1_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0132038, upper bound: 0.0132994
NS_A1_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0128392, upper bound: 0.0131226
NS_A1_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0129986, upper bound: 0.0131238
NS_A1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0128723, upper bound: 0.0130869
NS_A1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0128723, upper bound: 0.0130919
NS_A1_A1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0126269, upper bound: 0.0126916
NS_A1_A1_B2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0126400, upper bound: 0.0128438
NS_A1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0128723, upper bound: 0.0132354
NS_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0128723, upper bound: 0.0132354
NS_A1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0124713, upper bound: 0.0130064
NS_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0126400, upper bound: 0.0130096
NS_A1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0132994, upper bound: 0.0132038
NS_A1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0132994, upper bound: 0.0132453
NS_A1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0131226, upper bound: 0.0128392
NS_A1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0131238, upper bound: 0.0129986
NS_A1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0129538, upper bound: 0.0130869
NS_A1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0129538, upper bound: 0.0130919
NS_A1_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0127544, upper bound: 0.0126916
NS_A1_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0127608, upper bound: 0.0128437
NS_A1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0132994, upper bound: 0.0132038
NS_A1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0132994, upper bound: 0.0132453
NS_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0131226, upper bound: 0.0128432
NS_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0131238, upper bound: 0.0129986
NS_A1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0129555, upper bound: 0.0130913
NS_A1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0129555, upper bound: 0.0130981
NS_A1_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0127575, upper bound: 0.0127111
NS_A1_A2_B2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0127639, upper bound: 0.0128519
NS_A2_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0130869, upper bound: 0.0128723
NS_A2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0130869, upper bound: 0.0128723
NS_A2_A1_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0129258, upper bound: 0.0128720
NS_A2_A1_B1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0129258, upper bound: 0.0128770
NS_A2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0129546
NS_A2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0129546
NS_A2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0129546
NS_A2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0131607, upper bound: 0.0129546
NS_A2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0132354, upper bound: 0.0128723
NS_A2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0132354, upper bound: 0.0128779
NS_A2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0130064, upper bound: 0.0124713
NS_A2_A2_B1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0128342, upper bound: 0.0126400
NS_A2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0130625, upper bound: 0.0128720
NS_A2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0130625, upper bound: 0.0128770
NS_A2_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0128324, upper bound: 0.0124703
NS_A2_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0128428, upper bound: 0.0126387
NS_A2_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0132256, upper bound: 0.0128960
NS_A2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0132256, upper bound: 0.0128960
NS_A2_A2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0128993, upper bound: 0.0126629
NS_A2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0130096, upper bound: 0.0126761
NS_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0130675, upper bound: 0.0128960
NS_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0130675, upper bound: 0.0129091
NS_A2_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0128381, upper bound: 0.0125199
NS_A2_A2_B2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 9, lower bound: -0.0128489, upper bound: 0.0126759

## BFS NS instance: NS_A1_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0051745, 0.0007101, -0.0051745, 0.0007101, -0.0058847, 0.0058847
1: -0.0029166, 0.0107924, -0.0029166, 0.0107924, -0.0137089, 0.0137089
2: 0.0051000, 0.0194649, 0.0051000, 0.0194649, -0.0135072, 0.0135072
3: -0.0066025, -0.0018866, -0.0066025, -0.0018866, -0.0047159, 0.0047159
4: 0.0034578, 0.0077059, 0.0034578, 0.0077059, -0.0040480, 0.0040480
5: -0.0054525, 0.0008183, -0.0054525, 0.0008183, -0.0062708, 0.0062708
6: -0.0067205, -0.0047118, -0.0067205, -0.0047118, -0.0020087, 0.0020087
7: -0.0053562, 0.0005414, -0.0053562, 0.0005414, -0.0058977, 0.0058977
8: -0.0074181, -0.0013818, -0.0074181, -0.0013818, -0.0060363, 0.0060363
9: 0.9945275, 1.0122643, 0.9945275, 1.0122643, -0.0177368, 0.0177368

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B1_B1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129404, upper bound: 0.0128677
time: 2.45 seconds

## Relational analysis of NS_A1_A1_B1_B1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129457, upper bound: 0.0130287
time: 1.65 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0051745, 0.0007101, -0.0050759, 0.0006935, -0.0058680, 0.0057860
1: -0.0029166, 0.0107924, -0.0041007, 0.0105854, -0.0135019, 0.0148930
2: 0.0051000, 0.0194649, 0.0042883, 0.0190407, -0.0132749, 0.0144549
3: -0.0066025, -0.0018866, -0.0063938, -0.0017833, -0.0048192, 0.0045072
4: 0.0034578, 0.0077059, 0.0034435, 0.0080327, -0.0044680, 0.0041690
5: -0.0054525, 0.0008183, -0.0058576, 0.0007851, -0.0062376, 0.0066759
6: -0.0067205, -0.0047118, -0.0066460, -0.0046634, -0.0020571, 0.0019342
7: -0.0053562, 0.0005414, -0.0053652, 0.0006751, -0.0060313, 0.0059066
8: -0.0074181, -0.0013818, -0.0071422, -0.0014029, -0.0060152, 0.0057604
9: 0.9945275, 1.0122643, 0.9947570, 1.0137668, -0.0192393, 0.0175073

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B1_B1_A1_B2_A1

### Relational analysis result of NS_A1_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127813, upper bound: 0.0130289
time: 3.04 seconds

## Relational analysis of NS_A1_A1_B1_B1_A1_B2_A2

### Relational analysis result of NS_A1_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127813, upper bound: 0.0130330
time: 2.33 seconds

## BFS NS instance: NS_A1_A1_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0051745, 0.0007101, -0.0053825, 0.0014202, -0.0065947, 0.0060926
1: -0.0029166, 0.0107924, -0.0031722, 0.0116048, -0.0145213, 0.0139646
2: 0.0051000, 0.0194649, 0.0049452, 0.0209254, -0.0153198, 0.0139818
3: -0.0066025, -0.0018866, -0.0071040, -0.0018634, -0.0047390, 0.0052174
4: 0.0034578, 0.0077059, 0.0033040, 0.0077276, -0.0042041, 0.0044019
5: -0.0054525, 0.0008183, -0.0056359, 0.0009839, -0.0064364, 0.0064542
6: -0.0067205, -0.0047118, -0.0069272, -0.0046625, -0.0020580, 0.0022154
7: -0.0053562, 0.0005414, -0.0057477, 0.0008325, -0.0061887, 0.0062892
8: -0.0074181, -0.0013818, -0.0083890, -0.0013721, -0.0060460, 0.0070072
9: 0.9945275, 1.0122643, 0.9936654, 1.0124726, -0.0179452, 0.0185989

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B1_B2_B1_A1_B1

### Relational analysis result of NS_A1_A1_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130247, upper bound: 0.0129135
time: 1.65 seconds

## Relational analysis of NS_A1_A1_B1_B2_B1_A1_B2

### Relational analysis result of NS_A1_A1_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130247, upper bound: 0.0130414
time: 2.21 seconds

## BFS NS instance: NS_A1_A1_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0050759, 0.0006935, -0.0053825, 0.0014202, -0.0064961, 0.0060760
1: -0.0041007, 0.0105854, -0.0031722, 0.0116048, -0.0157054, 0.0137576
2: 0.0042883, 0.0190407, 0.0049452, 0.0209254, -0.0162675, 0.0137495
3: -0.0063938, -0.0017833, -0.0071040, -0.0018634, -0.0045304, 0.0053208
4: 0.0034435, 0.0080327, 0.0033040, 0.0077276, -0.0042841, 0.0047287
5: -0.0058576, 0.0007851, -0.0056359, 0.0009839, -0.0068415, 0.0064211
6: -0.0066460, -0.0046634, -0.0069272, -0.0046625, -0.0019835, 0.0022638
7: -0.0053652, 0.0006751, -0.0057477, 0.0008325, -0.0061977, 0.0064228
8: -0.0071422, -0.0014029, -0.0083890, -0.0013721, -0.0057701, 0.0069861
9: 0.9947570, 1.0137668, 0.9936654, 1.0124726, -0.0177156, 0.0201014

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B1_B2_B1_A2_B1

### Relational analysis result of NS_A1_A1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130247, upper bound: 0.0129135
time: 1.48 seconds

## Relational analysis of NS_A1_A1_B1_B2_B1_A2_B2

### Relational analysis result of NS_A1_A1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130247, upper bound: 0.0130414
time: 2.07 seconds

## BFS NS instance: NS_A1_A1_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0049249, 0.0002318, -0.0052703, 0.0011872, -0.0061121, 0.0055021
1: -0.0029030, 0.0102336, -0.0044603, 0.0113534, -0.0142564, 0.0146939
2: 0.0050393, 0.0183484, 0.0040703, 0.0204258, -0.0148935, 0.0138374
3: -0.0060307, -0.0019027, -0.0068615, -0.0017616, -0.0042691, 0.0049588
4: 0.0037200, 0.0077219, 0.0033157, 0.0080807, -0.0043607, 0.0044062
5: -0.0053537, 0.0004760, -0.0060250, 0.0009106, -0.0062643, 0.0065010
6: -0.0065249, -0.0048131, -0.0068405, -0.0046282, -0.0018967, 0.0020274
7: -0.0052333, 0.0003302, -0.0057492, 0.0009673, -0.0062007, 0.0060794
8: -0.0066987, -0.0014507, -0.0080672, -0.0014113, -0.0052874, 0.0066164
9: 0.9953943, 1.0122818, 0.9939442, 1.0141168, -0.0187225, 0.0183375

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_A1_B1_B2_B2_A1_A1

### Relational analysis result of NS_A1_A1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127808, upper bound: 0.0131226
time: 2.13 seconds

## Relational analysis of NS_A1_A1_B1_B2_B2_A1_A2

### Relational analysis result of NS_A1_A1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127808, upper bound: 0.0130400
time: 2.50 seconds

## BFS NS instance: NS_A1_A1_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0051842, 0.0007455, -0.0053300, 0.0013300, -0.0065142, 0.0060755
1: -0.0029059, 0.0108433, -0.0044856, 0.0114998, -0.0144057, 0.0153289
2: 0.0050979, 0.0195519, 0.0040666, 0.0207120, -0.0153099, 0.0147550
3: -0.0066237, -0.0018859, -0.0069993, -0.0017582, -0.0048655, 0.0051134
4: 0.0034946, 0.0077066, 0.0032570, 0.0080813, -0.0045150, 0.0044495
5: -0.0054547, 0.0007899, -0.0060553, 0.0009895, -0.0064442, 0.0068452
6: -0.0067312, -0.0047397, -0.0068886, -0.0046063, -0.0021248, 0.0021489
7: -0.0053494, 0.0005430, -0.0057891, 0.0010180, -0.0063673, 0.0063321
8: -0.0074742, -0.0013840, -0.0082554, -0.0013952, -0.0060790, 0.0068714
9: 0.9945230, 1.0122563, 0.9937360, 1.0141370, -0.0196140, 0.0185202

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B1_B2_B2_A2_B1

### Relational analysis result of NS_A1_A1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129952, upper bound: 0.0130013
time: 1.64 seconds

## Relational analysis of NS_A1_A1_B1_B2_B2_A2_B2

### Relational analysis result of NS_A1_A1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129952, upper bound: 0.0131238
time: 2.20 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0051745, 0.0007101, -0.0051840, 0.0007616, -0.0059361, 0.0058941
1: -0.0029166, 0.0107924, -0.0033155, 0.0108001, -0.0137167, 0.0141079
2: 0.0051000, 0.0194649, 0.0048515, 0.0194852, -0.0135948, 0.0137115
3: -0.0066025, -0.0018866, -0.0066333, -0.0017540, -0.0048484, 0.0047467
4: 0.0034578, 0.0077059, 0.0033963, 0.0078766, -0.0042175, 0.0041785
5: -0.0054525, 0.0008183, -0.0057507, 0.0008870, -0.0063395, 0.0065691
6: -0.0067205, -0.0047118, -0.0067269, -0.0046726, -0.0020478, 0.0020151
7: -0.0053562, 0.0005414, -0.0054356, 0.0006124, -0.0059687, 0.0059770
8: -0.0074181, -0.0013818, -0.0074341, -0.0013610, -0.0060571, 0.0060523
9: 0.9945275, 1.0122643, 0.9944275, 1.0128963, -0.0183688, 0.0178367

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124702, upper bound: 0.0129371
time: 1.84 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126440, upper bound: 0.0129455
time: 2.13 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0051745, 0.0007101, -0.0050623, 0.0008295, -0.0060041, 0.0057725
1: -0.0029166, 0.0107924, -0.0044343, 0.0105351, -0.0134517, 0.0152267
2: 0.0051000, 0.0194649, 0.0040739, 0.0189515, -0.0132217, 0.0146341
3: -0.0066025, -0.0018866, -0.0063708, -0.0016628, -0.0049396, 0.0044842
4: 0.0034578, 0.0077059, 0.0034033, 0.0081798, -0.0045954, 0.0042391
5: -0.0054525, 0.0008183, -0.0061150, 0.0008212, -0.0062737, 0.0069333
6: -0.0067205, -0.0047118, -0.0066357, -0.0046325, -0.0020880, 0.0019239
7: -0.0053562, 0.0005414, -0.0054401, 0.0007161, -0.0060723, 0.0059815
8: -0.0074181, -0.0013818, -0.0071006, -0.0013904, -0.0060277, 0.0057188
9: 0.9945275, 1.0122643, 0.9947289, 1.0142944, -0.0197669, 0.0175353

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124702, upper bound: 0.0129371
time: 2.35 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126440, upper bound: 0.0129455
time: 2.34 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0051745, 0.0007101, -0.0054251, 0.0015051, -0.0066797, 0.0061352
1: -0.0029166, 0.0107924, -0.0035944, 0.0116515, -0.0145680, 0.0143868
2: 0.0051000, 0.0194649, 0.0046760, 0.0210417, -0.0155137, 0.0142241
3: -0.0066025, -0.0018866, -0.0072004, -0.0017358, -0.0048667, 0.0053139
4: 0.0034578, 0.0077059, 0.0032311, 0.0079185, -0.0043902, 0.0044748
5: -0.0054525, 0.0008183, -0.0059257, 0.0010752, -0.0065277, 0.0067440
6: -0.0067205, -0.0047118, -0.0069546, -0.0046292, -0.0020912, 0.0022429
7: -0.0053562, 0.0005414, -0.0058100, 0.0009171, -0.0062734, 0.0063515
8: -0.0074181, -0.0013818, -0.0084685, -0.0013540, -0.0060641, 0.0070868
9: 0.9945275, 1.0122643, 0.9934891, 1.0131640, -0.0186366, 0.0187752

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B2_B2_B1_A1_A1

### Relational analysis result of NS_A1_A1_B2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126773, upper bound: 0.0129739
time: 2.44 seconds

## Relational analysis of NS_A1_A1_B2_B2_B1_A1_A2

### Relational analysis result of NS_A1_A1_B2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128611, upper bound: 0.0129771
time: 2.13 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0050759, 0.0006935, -0.0054251, 0.0015051, -0.0065810, 0.0061186
1: -0.0041007, 0.0105854, -0.0035944, 0.0116515, -0.0157521, 0.0141798
2: 0.0042883, 0.0190407, 0.0046760, 0.0210417, -0.0164614, 0.0139918
3: -0.0063938, -0.0017833, -0.0072004, -0.0017358, -0.0046580, 0.0054172
4: 0.0034435, 0.0080327, 0.0032311, 0.0079185, -0.0044750, 0.0048016
5: -0.0058576, 0.0007851, -0.0059257, 0.0010752, -0.0069327, 0.0067108
6: -0.0066460, -0.0046634, -0.0069546, -0.0046292, -0.0020167, 0.0022912
7: -0.0053652, 0.0006751, -0.0058100, 0.0009171, -0.0062823, 0.0064851
8: -0.0071422, -0.0014029, -0.0084685, -0.0013540, -0.0057882, 0.0070656
9: 0.9947570, 1.0137668, 0.9934891, 1.0131640, -0.0184070, 0.0202777

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B2_B2_B1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128509, upper bound: 0.0128613
time: 1.78 seconds

## Relational analysis of NS_A1_A1_B2_B2_B1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128611, upper bound: 0.0129771
time: 1.96 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0049249, 0.0002318, -0.0052901, 0.0012540, -0.0061790, 0.0055220
1: -0.0029030, 0.0102336, -0.0048203, 0.0113482, -0.0142512, 0.0150538
2: 0.0050393, 0.0183484, 0.0038302, 0.0204381, -0.0149923, 0.0140470
3: -0.0060307, -0.0019027, -0.0068969, -0.0016459, -0.0043849, 0.0049942
4: 0.0037200, 0.0077219, 0.0032559, 0.0082521, -0.0045176, 0.0044660
5: -0.0053537, 0.0004760, -0.0062905, 0.0009776, -0.0063313, 0.0067665
6: -0.0065249, -0.0048131, -0.0068486, -0.0045969, -0.0019280, 0.0020355
7: -0.0052333, 0.0003302, -0.0058112, 0.0010118, -0.0062451, 0.0061414
8: -0.0066987, -0.0014507, -0.0080842, -0.0014006, -0.0052981, 0.0066334
9: 0.9953943, 1.0122818, 0.9938437, 1.0147070, -0.0193127, 0.0184381

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_A1_B2_B2_B2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124604, upper bound: 0.0130064
time: 1.62 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124604, upper bound: 0.0129738
time: 1.99 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0051842, 0.0007455, -0.0053490, 0.0013783, -0.0065625, 0.0060945
1: -0.0029059, 0.0108433, -0.0048444, 0.0114943, -0.0144003, 0.0156877
2: 0.0050979, 0.0195519, 0.0038267, 0.0207236, -0.0154066, 0.0149696
3: -0.0066237, -0.0018859, -0.0070322, -0.0016424, -0.0049814, 0.0051463
4: 0.0034946, 0.0077066, 0.0031972, 0.0082527, -0.0046687, 0.0045093
5: -0.0054547, 0.0007899, -0.0063181, 0.0010545, -0.0065092, 0.0071080
6: -0.0067312, -0.0047397, -0.0068964, -0.0045752, -0.0021560, 0.0021568
7: -0.0053494, 0.0005430, -0.0058517, 0.0010632, -0.0064126, 0.0063947
8: -0.0074742, -0.0013840, -0.0082703, -0.0013857, -0.0060885, 0.0068863
9: 0.9945230, 1.0122563, 0.9936355, 1.0147262, -0.0202031, 0.0186207

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B2_B2_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126269, upper bound: 0.0128993
time: 1.45 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126269, upper bound: 0.0130096
time: 3.05 seconds

## BFS NS instance: NS_A1_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0053825, 0.0014202, -0.0051745, 0.0007101, -0.0060926, 0.0065947
1: -0.0031722, 0.0116048, -0.0029166, 0.0107924, -0.0139646, 0.0145213
2: 0.0049452, 0.0209254, 0.0051000, 0.0194649, -0.0139818, 0.0153198
3: -0.0071040, -0.0018634, -0.0066025, -0.0018866, -0.0052174, 0.0047390
4: 0.0033040, 0.0077276, 0.0034578, 0.0077059, -0.0044019, 0.0042041
5: -0.0056359, 0.0009839, -0.0054525, 0.0008183, -0.0064542, 0.0064364
6: -0.0069272, -0.0046625, -0.0067205, -0.0047118, -0.0022154, 0.0020580
7: -0.0057477, 0.0008325, -0.0053562, 0.0005414, -0.0062892, 0.0061887
8: -0.0083890, -0.0013721, -0.0074181, -0.0013818, -0.0070072, 0.0060460
9: 0.9936654, 1.0124726, 0.9945275, 1.0122643, -0.0185989, 0.0179452

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129135, upper bound: 0.0130247
time: 2.32 seconds

## Relational analysis of NS_A1_A2_B1_B1_A1_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130414, upper bound: 0.0130287
time: 3.23 seconds

## BFS NS instance: NS_A1_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0053825, 0.0014202, -0.0050759, 0.0006935, -0.0060760, 0.0064961
1: -0.0031722, 0.0116048, -0.0041007, 0.0105854, -0.0137576, 0.0157054
2: 0.0049452, 0.0209254, 0.0042883, 0.0190407, -0.0137495, 0.0162675
3: -0.0071040, -0.0018634, -0.0063938, -0.0017833, -0.0053208, 0.0045304
4: 0.0033040, 0.0077276, 0.0034435, 0.0080327, -0.0047287, 0.0042841
5: -0.0056359, 0.0009839, -0.0058576, 0.0007851, -0.0064211, 0.0068415
6: -0.0069272, -0.0046625, -0.0066460, -0.0046634, -0.0022638, 0.0019835
7: -0.0057477, 0.0008325, -0.0053652, 0.0006751, -0.0064228, 0.0061977
8: -0.0083890, -0.0013721, -0.0071422, -0.0014029, -0.0069861, 0.0057701
9: 0.9936654, 1.0124726, 0.9947570, 1.0137668, -0.0201014, 0.0177156

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129135, upper bound: 0.0130289
time: 1.74 seconds

## Relational analysis of NS_A1_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130414, upper bound: 0.0130330
time: 1.60 seconds

## BFS NS instance: NS_A1_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0052703, 0.0011872, -0.0049249, 0.0002318, -0.0055021, 0.0061121
1: -0.0044603, 0.0113534, -0.0029030, 0.0102336, -0.0146939, 0.0142564
2: 0.0040703, 0.0204258, 0.0050393, 0.0183484, -0.0138374, 0.0148935
3: -0.0068615, -0.0017616, -0.0060307, -0.0019027, -0.0049588, 0.0042691
4: 0.0033157, 0.0080807, 0.0037200, 0.0077219, -0.0044062, 0.0043607
5: -0.0060250, 0.0009106, -0.0053537, 0.0004760, -0.0065010, 0.0062643
6: -0.0068405, -0.0046282, -0.0065249, -0.0048131, -0.0020274, 0.0018967
7: -0.0057492, 0.0009673, -0.0052333, 0.0003302, -0.0060794, 0.0062007
8: -0.0080672, -0.0014113, -0.0066987, -0.0014507, -0.0066164, 0.0052874
9: 0.9939442, 1.0141168, 0.9953943, 1.0122818, -0.0183375, 0.0187225

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_A2_B1_B1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131226, upper bound: 0.0127808
time: 2.06 seconds

## Relational analysis of NS_A1_A2_B1_B1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131226, upper bound: 0.0127808
time: 1.67 seconds

## BFS NS instance: NS_A1_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0053300, 0.0013300, -0.0051842, 0.0007455, -0.0060755, 0.0065142
1: -0.0044856, 0.0114998, -0.0029059, 0.0108433, -0.0153289, 0.0144057
2: 0.0040666, 0.0207120, 0.0050979, 0.0195519, -0.0147550, 0.0153099
3: -0.0069993, -0.0017582, -0.0066237, -0.0018859, -0.0051134, 0.0048655
4: 0.0032570, 0.0080813, 0.0034946, 0.0077066, -0.0044495, 0.0045150
5: -0.0060553, 0.0009895, -0.0054547, 0.0007899, -0.0068452, 0.0064442
6: -0.0068886, -0.0046063, -0.0067312, -0.0047397, -0.0021489, 0.0021248
7: -0.0057891, 0.0010180, -0.0053494, 0.0005430, -0.0063321, 0.0063673
8: -0.0082554, -0.0013952, -0.0074742, -0.0013840, -0.0068714, 0.0060790
9: 0.9937360, 1.0141370, 0.9945230, 1.0122563, -0.0185202, 0.0196140

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130013, upper bound: 0.0129952
time: 2.28 seconds

## Relational analysis of NS_A1_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130013, upper bound: 0.0129986
time: 2.21 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0053825, 0.0014202, -0.0051840, 0.0007616, -0.0061441, 0.0066042
1: -0.0031722, 0.0116048, -0.0033155, 0.0108001, -0.0139723, 0.0149203
2: 0.0049452, 0.0209254, 0.0048515, 0.0194852, -0.0140694, 0.0155241
3: -0.0071040, -0.0018634, -0.0066333, -0.0017540, -0.0053500, 0.0047699
4: 0.0033040, 0.0077276, 0.0033963, 0.0078766, -0.0045727, 0.0043313
5: -0.0056359, 0.0009839, -0.0057507, 0.0008870, -0.0065230, 0.0067346
6: -0.0069272, -0.0046625, -0.0067269, -0.0046726, -0.0022546, 0.0020644
7: -0.0057477, 0.0008325, -0.0054356, 0.0006124, -0.0063602, 0.0062681
8: -0.0083890, -0.0013721, -0.0074341, -0.0013610, -0.0070280, 0.0060620
9: 0.9936654, 1.0124726, 0.9944275, 1.0128963, -0.0192309, 0.0180451

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B1_B2_A1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0125678, upper bound: 0.0129371
time: 2.01 seconds

## Relational analysis of NS_A1_A2_B1_B2_A1_B1_A2

### Relational analysis result of NS_A1_A2_B1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127234, upper bound: 0.0129455
time: 2.84 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0053825, 0.0014202, -0.0050623, 0.0008295, -0.0062120, 0.0064825
1: -0.0031722, 0.0116048, -0.0044343, 0.0105351, -0.0137073, 0.0160391
2: 0.0049452, 0.0209254, 0.0040739, 0.0189515, -0.0136964, 0.0164467
3: -0.0071040, -0.0018634, -0.0063708, -0.0016628, -0.0054412, 0.0045073
4: 0.0033040, 0.0077276, 0.0034033, 0.0081798, -0.0048758, 0.0043243
5: -0.0056359, 0.0009839, -0.0061150, 0.0008212, -0.0064571, 0.0070989
6: -0.0069272, -0.0046625, -0.0066357, -0.0046325, -0.0022947, 0.0019732
7: -0.0057477, 0.0008325, -0.0054401, 0.0007161, -0.0064639, 0.0062726
8: -0.0083890, -0.0013721, -0.0071006, -0.0013904, -0.0069986, 0.0057285
9: 0.9936654, 1.0124726, 0.9947289, 1.0142944, -0.0206290, 0.0177437

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B1_B2_A1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0125678, upper bound: 0.0129371
time: 1.85 seconds

## Relational analysis of NS_A1_A2_B1_B2_A1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127234, upper bound: 0.0129455
time: 2.05 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0053825, 0.0014202, -0.0053825, 0.0014202, -0.0068027, 0.0068027
1: -0.0031722, 0.0116048, -0.0031722, 0.0116048, -0.0147769, 0.0147769
2: 0.0049452, 0.0209254, 0.0049452, 0.0209254, -0.0147408, 0.0147408
3: -0.0071040, -0.0018634, -0.0071040, -0.0018634, -0.0052406, 0.0052406
4: 0.0033040, 0.0077276, 0.0033040, 0.0077276, -0.0041826, 0.0041826
5: -0.0056359, 0.0009839, -0.0056359, 0.0009839, -0.0066198, 0.0066198
6: -0.0069272, -0.0046625, -0.0069272, -0.0046625, -0.0022647, 0.0022647
7: -0.0057477, 0.0008325, -0.0057477, 0.0008325, -0.0065802, 0.0065802
8: -0.0083890, -0.0013721, -0.0083890, -0.0013721, -0.0070169, 0.0070169
9: 0.9936654, 1.0124726, 0.9936654, 1.0124726, -0.0188072, 0.0188072

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B2_B1_A1_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129135, upper bound: 0.0130245
time: 1.61 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130414, upper bound: 0.0130287
time: 2.76 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0053825, 0.0014202, -0.0053300, 0.0013300, -0.0067125, 0.0067502
1: -0.0031722, 0.0116048, -0.0044856, 0.0114998, -0.0146720, 0.0160904
2: 0.0049452, 0.0209254, 0.0040666, 0.0207120, -0.0146595, 0.0157028
3: -0.0071040, -0.0018634, -0.0069993, -0.0017582, -0.0053458, 0.0051359
4: 0.0033040, 0.0077276, 0.0032570, 0.0080813, -0.0046076, 0.0043166
5: -0.0056359, 0.0009839, -0.0060553, 0.0009895, -0.0066254, 0.0070392
6: -0.0069272, -0.0046625, -0.0068886, -0.0046063, -0.0023208, 0.0022262
7: -0.0057477, 0.0008325, -0.0057891, 0.0010180, -0.0067657, 0.0066216
8: -0.0083890, -0.0013721, -0.0082554, -0.0013952, -0.0069938, 0.0068833
9: 0.9936654, 1.0124726, 0.9937360, 1.0141370, -0.0204716, 0.0187366

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B2_B1_A1_B2_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129135, upper bound: 0.0130289
time: 2.06 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B2_A2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130414, upper bound: 0.0130330
time: 2.26 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0052703, 0.0011872, -0.0051899, 0.0009601, -0.0062304, 0.0063771
1: -0.0044603, 0.0113534, -0.0032120, 0.0111712, -0.0156315, 0.0145654
2: 0.0040703, 0.0204258, 0.0048593, 0.0200606, -0.0148022, 0.0143571
3: -0.0068615, -0.0017616, -0.0066482, -0.0018786, -0.0049829, 0.0048866
4: 0.0033157, 0.0080807, 0.0035332, 0.0077555, -0.0042505, 0.0043666
5: -0.0060250, 0.0009106, -0.0055465, 0.0006820, -0.0067070, 0.0064571
6: -0.0068405, -0.0046282, -0.0067744, -0.0047561, -0.0020844, 0.0021461
7: -0.0057492, 0.0009673, -0.0056619, 0.0006466, -0.0063958, 0.0066293
8: -0.0080672, -0.0014113, -0.0078332, -0.0014430, -0.0066242, 0.0064220
9: 0.9939442, 1.0141168, 0.9943468, 1.0125587, -0.0186145, 0.0197700

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131226, upper bound: 0.0127819
time: 2.12 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131226, upper bound: 0.0127819
time: 2.11 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0053300, 0.0013300, -0.0054097, 0.0015013, -0.0068313, 0.0067398
1: -0.0044856, 0.0114998, -0.0031624, 0.0116949, -0.0161805, 0.0146622
2: 0.0040666, 0.0207120, 0.0049439, 0.0210881, -0.0155004, 0.0147799
3: -0.0069993, -0.0017582, -0.0071601, -0.0018621, -0.0051372, 0.0054019
4: 0.0032570, 0.0080813, 0.0033236, 0.0077283, -0.0043934, 0.0044480
5: -0.0060553, 0.0009895, -0.0056515, 0.0009747, -0.0070300, 0.0066410
6: -0.0068886, -0.0046063, -0.0069510, -0.0046843, -0.0022043, 0.0023446
7: -0.0057891, 0.0010180, -0.0057661, 0.0008403, -0.0066294, 0.0067841
8: -0.0082554, -0.0013952, -0.0085006, -0.0013751, -0.0068803, 0.0071054
9: 0.9937360, 1.0141370, 0.9936050, 1.0124654, -0.0187293, 0.0205320

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131238, upper bound: 0.0129451
time: 2.49 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131238, upper bound: 0.0129451
time: 2.59 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0053825, 0.0014202, -0.0054251, 0.0015051, -0.0068876, 0.0068453
1: -0.0031722, 0.0116048, -0.0035944, 0.0116515, -0.0148237, 0.0151992
2: 0.0049452, 0.0209254, 0.0046760, 0.0210417, -0.0149105, 0.0149516
3: -0.0071040, -0.0018634, -0.0072004, -0.0017358, -0.0053682, 0.0053370
4: 0.0033040, 0.0077276, 0.0032311, 0.0079185, -0.0043573, 0.0042871
5: -0.0056359, 0.0009839, -0.0059257, 0.0010752, -0.0067111, 0.0069096
6: -0.0069272, -0.0046625, -0.0069546, -0.0046292, -0.0022980, 0.0022922
7: -0.0057477, 0.0008325, -0.0058100, 0.0009171, -0.0066649, 0.0066426
8: -0.0083890, -0.0013721, -0.0084685, -0.0013540, -0.0070349, 0.0070964
9: 0.9936654, 1.0124726, 0.9934891, 1.0131640, -0.0194986, 0.0189835

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B2_B2_A1_B1_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0125705, upper bound: 0.0129384
time: 2.41 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B1_A2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127246, upper bound: 0.0129470
time: 1.93 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0053825, 0.0014202, -0.0053490, 0.0013783, -0.0067608, 0.0067692
1: -0.0031722, 0.0116048, -0.0048444, 0.0114943, -0.0146665, 0.0164492
2: 0.0049452, 0.0209254, 0.0038267, 0.0207236, -0.0146832, 0.0158846
3: -0.0071040, -0.0018634, -0.0070322, -0.0016424, -0.0054617, 0.0051687
4: 0.0033040, 0.0077276, 0.0031972, 0.0082527, -0.0047381, 0.0043649
5: -0.0056359, 0.0009839, -0.0063181, 0.0010545, -0.0066905, 0.0073020
6: -0.0069272, -0.0046625, -0.0068964, -0.0045752, -0.0023520, 0.0022340
7: -0.0057477, 0.0008325, -0.0058517, 0.0010632, -0.0068110, 0.0066842
8: -0.0083890, -0.0013721, -0.0082703, -0.0013857, -0.0070033, 0.0068982
9: 0.9936654, 1.0124726, 0.9936355, 1.0147262, -0.0210608, 0.0188371

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0125705, upper bound: 0.0129384
time: 1.67 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127246, upper bound: 0.0129470
time: 2.22 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0051840, 0.0007616, -0.0051745, 0.0007101, -0.0058941, 0.0059361
1: -0.0033155, 0.0108001, -0.0029166, 0.0107924, -0.0141079, 0.0137167
2: 0.0048515, 0.0194852, 0.0051000, 0.0194649, -0.0137115, 0.0135948
3: -0.0066333, -0.0017540, -0.0066025, -0.0018866, -0.0047467, 0.0048484
4: 0.0033963, 0.0078766, 0.0034578, 0.0077059, -0.0041785, 0.0042175
5: -0.0057507, 0.0008870, -0.0054525, 0.0008183, -0.0065691, 0.0063395
6: -0.0067269, -0.0046726, -0.0067205, -0.0047118, -0.0020151, 0.0020478
7: -0.0054356, 0.0006124, -0.0053562, 0.0005414, -0.0059770, 0.0059687
8: -0.0074341, -0.0013610, -0.0074181, -0.0013818, -0.0060523, 0.0060571
9: 0.9944275, 1.0128963, 0.9945275, 1.0122643, -0.0178367, 0.0183688

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A1_B1_B1_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129371, upper bound: 0.0124702
time: 2.25 seconds

## Relational analysis of NS_A2_A1_B1_B1_B1_A1_B2

### Relational analysis result of NS_A2_A1_B1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129455, upper bound: 0.0126440
time: 2.22 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0050623, 0.0008295, -0.0051745, 0.0007101, -0.0057725, 0.0060041
1: -0.0044343, 0.0105351, -0.0029166, 0.0107924, -0.0152267, 0.0134517
2: 0.0040739, 0.0189515, 0.0051000, 0.0194649, -0.0146341, 0.0132217
3: -0.0063708, -0.0016628, -0.0066025, -0.0018866, -0.0044842, 0.0049396
4: 0.0034033, 0.0081798, 0.0034578, 0.0077059, -0.0042391, 0.0045954
5: -0.0061150, 0.0008212, -0.0054525, 0.0008183, -0.0069333, 0.0062737
6: -0.0066357, -0.0046325, -0.0067205, -0.0047118, -0.0019239, 0.0020880
7: -0.0054401, 0.0007161, -0.0053562, 0.0005414, -0.0059815, 0.0060723
8: -0.0071006, -0.0013904, -0.0074181, -0.0013818, -0.0057188, 0.0060277
9: 0.9947289, 1.0142944, 0.9945275, 1.0122643, -0.0175353, 0.0197669

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A1_B1_B1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129371, upper bound: 0.0124702
time: 2.30 seconds

## Relational analysis of NS_A2_A1_B1_B1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129455, upper bound: 0.0126440
time: 1.82 seconds

## BFS NS instance: NS_A2_A1_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0051840, 0.0007616, -0.0053825, 0.0014202, -0.0066042, 0.0061441
1: -0.0033155, 0.0108001, -0.0031722, 0.0116048, -0.0149203, 0.0139723
2: 0.0048515, 0.0194852, 0.0049452, 0.0209254, -0.0155241, 0.0140694
3: -0.0066333, -0.0017540, -0.0071040, -0.0018634, -0.0047699, 0.0053500
4: 0.0033963, 0.0078766, 0.0033040, 0.0077276, -0.0043313, 0.0045727
5: -0.0057507, 0.0008870, -0.0056359, 0.0009839, -0.0067346, 0.0065230
6: -0.0067269, -0.0046726, -0.0069272, -0.0046625, -0.0020644, 0.0022546
7: -0.0054356, 0.0006124, -0.0057477, 0.0008325, -0.0062681, 0.0063602
8: -0.0074341, -0.0013610, -0.0083890, -0.0013721, -0.0060620, 0.0070280
9: 0.9944275, 1.0128963, 0.9936654, 1.0124726, -0.0180451, 0.0192309

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A1_B2_B1_B1_A1_B1

### Relational analysis result of NS_A2_A1_B2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129310, upper bound: 0.0125678
time: 2.10 seconds

## Relational analysis of NS_A2_A1_B2_B1_B1_A1_B2

### Relational analysis result of NS_A2_A1_B2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129455, upper bound: 0.0127234
time: 2.07 seconds

## BFS NS instance: NS_A2_A1_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0050623, 0.0008295, -0.0053825, 0.0014202, -0.0064825, 0.0062120
1: -0.0044343, 0.0105351, -0.0031722, 0.0116048, -0.0160391, 0.0137073
2: 0.0040739, 0.0189515, 0.0049452, 0.0209254, -0.0164467, 0.0136964
3: -0.0063708, -0.0016628, -0.0071040, -0.0018634, -0.0045073, 0.0054412
4: 0.0034033, 0.0081798, 0.0033040, 0.0077276, -0.0043243, 0.0048758
5: -0.0061150, 0.0008212, -0.0056359, 0.0009839, -0.0070989, 0.0064571
6: -0.0066357, -0.0046325, -0.0069272, -0.0046625, -0.0019732, 0.0022947
7: -0.0054401, 0.0007161, -0.0057477, 0.0008325, -0.0062726, 0.0064639
8: -0.0071006, -0.0013904, -0.0083890, -0.0013721, -0.0057285, 0.0069986
9: 0.9947289, 1.0142944, 0.9936654, 1.0124726, -0.0177437, 0.0206290

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A1_B2_B1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129371, upper bound: 0.0125678
time: 2.71 seconds

## Relational analysis of NS_A2_A1_B2_B1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129455, upper bound: 0.0127234
time: 1.64 seconds

## BFS NS instance: NS_A2_A1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0051840, 0.0007616, -0.0054251, 0.0015051, -0.0066891, 0.0061867
1: -0.0033155, 0.0108001, -0.0035944, 0.0116515, -0.0149670, 0.0143945
2: 0.0048515, 0.0194852, 0.0046760, 0.0210417, -0.0155780, 0.0141683
3: -0.0066333, -0.0017540, -0.0072004, -0.0017358, -0.0048512, 0.0054464
4: 0.0033963, 0.0078766, 0.0032311, 0.0079185, -0.0043333, 0.0045220
5: -0.0057507, 0.0008870, -0.0059257, 0.0010752, -0.0068259, 0.0068127
6: -0.0067269, -0.0046726, -0.0069546, -0.0046292, -0.0020976, 0.0022820
7: -0.0054356, 0.0006124, -0.0058100, 0.0009171, -0.0063527, 0.0064225
8: -0.0074341, -0.0013610, -0.0084685, -0.0013540, -0.0060801, 0.0071076
9: 0.9944275, 1.0128963, 0.9934891, 1.0131640, -0.0187365, 0.0194072

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A1_B2_B1_B2_A1_B1

### Relational analysis result of NS_A2_A1_B2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129001, upper bound: 0.0125678
time: 2.13 seconds

## Relational analysis of NS_A2_A1_B2_B1_B2_A1_B2

### Relational analysis result of NS_A2_A1_B2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129117, upper bound: 0.0127234
time: 3.18 seconds

## BFS NS instance: NS_A2_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0050623, 0.0008295, -0.0054251, 0.0015051, -0.0065675, 0.0062546
1: -0.0044343, 0.0105351, -0.0035944, 0.0116515, -0.0160858, 0.0141296
2: 0.0040739, 0.0189515, 0.0046760, 0.0210417, -0.0165208, 0.0138494
3: -0.0063708, -0.0016628, -0.0072004, -0.0017358, -0.0046119, 0.0055376
4: 0.0034033, 0.0081798, 0.0032311, 0.0079185, -0.0044298, 0.0049424
5: -0.0061150, 0.0008212, -0.0059257, 0.0010752, -0.0071902, 0.0067469
6: -0.0066357, -0.0046325, -0.0069546, -0.0046292, -0.0020064, 0.0023222
7: -0.0054401, 0.0007161, -0.0058100, 0.0009171, -0.0063573, 0.0065262
8: -0.0071006, -0.0013904, -0.0084685, -0.0013540, -0.0057466, 0.0070781
9: 0.9947289, 1.0142944, 0.9934891, 1.0131640, -0.0184351, 0.0208053

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A1_B2_B1_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129001, upper bound: 0.0125678
time: 2.89 seconds

## Relational analysis of NS_A2_A1_B2_B1_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129117, upper bound: 0.0127234
time: 1.61 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0054251, 0.0015051, -0.0051745, 0.0007101, -0.0061352, 0.0066797
1: -0.0035944, 0.0116515, -0.0029166, 0.0107924, -0.0143868, 0.0145680
2: 0.0046760, 0.0210417, 0.0051000, 0.0194649, -0.0142241, 0.0155137
3: -0.0072004, -0.0017358, -0.0066025, -0.0018866, -0.0053139, 0.0048667
4: 0.0032311, 0.0079185, 0.0034578, 0.0077059, -0.0044748, 0.0043902
5: -0.0059257, 0.0010752, -0.0054525, 0.0008183, -0.0067440, 0.0065277
6: -0.0069546, -0.0046292, -0.0067205, -0.0047118, -0.0022429, 0.0020912
7: -0.0058100, 0.0009171, -0.0053562, 0.0005414, -0.0063515, 0.0062734
8: -0.0084685, -0.0013540, -0.0074181, -0.0013818, -0.0070868, 0.0060641
9: 0.9934891, 1.0131640, 0.9945275, 1.0122643, -0.0187752, 0.0186366

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B1_B1_A1_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129739, upper bound: 0.0126773
time: 1.66 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_B1_B2

### Relational analysis result of NS_A2_A2_B1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129771, upper bound: 0.0128611
time: 1.96 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0054251, 0.0015051, -0.0050759, 0.0006935, -0.0061186, 0.0065810
1: -0.0035944, 0.0116515, -0.0041007, 0.0105854, -0.0141798, 0.0157521
2: 0.0046760, 0.0210417, 0.0042883, 0.0190407, -0.0139918, 0.0164614
3: -0.0072004, -0.0017358, -0.0063938, -0.0017833, -0.0054172, 0.0046580
4: 0.0032311, 0.0079185, 0.0034435, 0.0080327, -0.0048016, 0.0044750
5: -0.0059257, 0.0010752, -0.0058576, 0.0007851, -0.0067108, 0.0069327
6: -0.0069546, -0.0046292, -0.0066460, -0.0046634, -0.0022912, 0.0020167
7: -0.0058100, 0.0009171, -0.0053652, 0.0006751, -0.0064851, 0.0062823
8: -0.0084685, -0.0013540, -0.0071422, -0.0014029, -0.0070656, 0.0057882
9: 0.9934891, 1.0131640, 0.9947570, 1.0137668, -0.0202777, 0.0184070

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128613, upper bound: 0.0128509
time: 2.17 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129771, upper bound: 0.0128611
time: 2.56 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0052901, 0.0012540, -0.0049249, 0.0002318, -0.0055220, 0.0061790
1: -0.0048203, 0.0113482, -0.0029030, 0.0102336, -0.0150538, 0.0142512
2: 0.0038302, 0.0204381, 0.0050393, 0.0183484, -0.0140470, 0.0149923
3: -0.0068969, -0.0016459, -0.0060307, -0.0019027, -0.0049942, 0.0043849
4: 0.0032559, 0.0082521, 0.0037200, 0.0077219, -0.0044660, 0.0045176
5: -0.0062905, 0.0009776, -0.0053537, 0.0004760, -0.0067665, 0.0063313
6: -0.0068486, -0.0045969, -0.0065249, -0.0048131, -0.0020355, 0.0019280
7: -0.0058112, 0.0010118, -0.0052333, 0.0003302, -0.0061414, 0.0062451
8: -0.0080842, -0.0014006, -0.0066987, -0.0014507, -0.0066334, 0.0052981
9: 0.9938437, 1.0147070, 0.9953943, 1.0122818, -0.0184381, 0.0193127

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_A2_B1_B1_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130064, upper bound: 0.0124604
time: 2.76 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130064, upper bound: 0.0124604
time: 2.16 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0054251, 0.0015051, -0.0051840, 0.0007616, -0.0061867, 0.0066891
1: -0.0035944, 0.0116515, -0.0033155, 0.0108001, -0.0143945, 0.0149670
2: 0.0046760, 0.0210417, 0.0048515, 0.0194852, -0.0141683, 0.0155780
3: -0.0072004, -0.0017358, -0.0066333, -0.0017540, -0.0054464, 0.0048512
4: 0.0032311, 0.0079185, 0.0033963, 0.0078766, -0.0045220, 0.0043333
5: -0.0059257, 0.0010752, -0.0057507, 0.0008870, -0.0068127, 0.0068259
6: -0.0069546, -0.0046292, -0.0067269, -0.0046726, -0.0022820, 0.0020976
7: -0.0058100, 0.0009171, -0.0054356, 0.0006124, -0.0064225, 0.0063527
8: -0.0084685, -0.0013540, -0.0074341, -0.0013610, -0.0071076, 0.0060801
9: 0.9934891, 1.0131640, 0.9944275, 1.0128963, -0.0194072, 0.0187365

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B1_B2_A1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127069, upper bound: 0.0128508
time: 2.07 seconds

## Relational analysis of NS_A2_A2_B1_B2_A1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128472, upper bound: 0.0128606
time: 1.71 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0054251, 0.0015051, -0.0050623, 0.0008295, -0.0062546, 0.0065675
1: -0.0035944, 0.0116515, -0.0044343, 0.0105351, -0.0141296, 0.0160858
2: 0.0046760, 0.0210417, 0.0040739, 0.0189515, -0.0138494, 0.0165208
3: -0.0072004, -0.0017358, -0.0063708, -0.0016628, -0.0055376, 0.0046119
4: 0.0032311, 0.0079185, 0.0034033, 0.0081798, -0.0049424, 0.0044298
5: -0.0059257, 0.0010752, -0.0061150, 0.0008212, -0.0067469, 0.0071902
6: -0.0069546, -0.0046292, -0.0066357, -0.0046325, -0.0023222, 0.0020064
7: -0.0058100, 0.0009171, -0.0054401, 0.0007161, -0.0065262, 0.0063573
8: -0.0084685, -0.0013540, -0.0071006, -0.0013904, -0.0070781, 0.0057466
9: 0.9934891, 1.0131640, 0.9947289, 1.0142944, -0.0208053, 0.0184351

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B1_B2_A1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127069, upper bound: 0.0128508
time: 1.98 seconds

## Relational analysis of NS_A2_A2_B1_B2_A1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128472, upper bound: 0.0128606
time: 2.33 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0054251, 0.0015051, -0.0053825, 0.0014202, -0.0068453, 0.0068876
1: -0.0035944, 0.0116515, -0.0031722, 0.0116048, -0.0151992, 0.0148237
2: 0.0046760, 0.0210417, 0.0049452, 0.0209254, -0.0149516, 0.0149105
3: -0.0072004, -0.0017358, -0.0071040, -0.0018634, -0.0053370, 0.0053682
4: 0.0032311, 0.0079185, 0.0033040, 0.0077276, -0.0042871, 0.0043573
5: -0.0059257, 0.0010752, -0.0056359, 0.0009839, -0.0069096, 0.0067111
6: -0.0069546, -0.0046292, -0.0069272, -0.0046625, -0.0022922, 0.0022980
7: -0.0058100, 0.0009171, -0.0057477, 0.0008325, -0.0066426, 0.0066649
8: -0.0084685, -0.0013540, -0.0083890, -0.0013721, -0.0070964, 0.0070349
9: 0.9934891, 1.0131640, 0.9936654, 1.0124726, -0.0189835, 0.0194986

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_B1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131434, upper bound: 0.0125026
time: 2.78 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_B2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131481, upper bound: 0.0126620
time: 2.78 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0053490, 0.0013783, -0.0053825, 0.0014202, -0.0067692, 0.0067608
1: -0.0048444, 0.0114943, -0.0031722, 0.0116048, -0.0164492, 0.0146665
2: 0.0038267, 0.0207236, 0.0049452, 0.0209254, -0.0158846, 0.0146832
3: -0.0070322, -0.0016424, -0.0071040, -0.0018634, -0.0051687, 0.0054617
4: 0.0031972, 0.0082527, 0.0033040, 0.0077276, -0.0043649, 0.0047381
5: -0.0063181, 0.0010545, -0.0056359, 0.0009839, -0.0073020, 0.0066905
6: -0.0068964, -0.0045752, -0.0069272, -0.0046625, -0.0022340, 0.0023520
7: -0.0058517, 0.0010632, -0.0057477, 0.0008325, -0.0066842, 0.0068110
8: -0.0082703, -0.0013857, -0.0083890, -0.0013721, -0.0068982, 0.0070033
9: 0.9936355, 1.0147262, 0.9936654, 1.0124726, -0.0188371, 0.0210608

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131434, upper bound: 0.0125026
time: 2.05 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131481, upper bound: 0.0126620
time: 3.17 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0054383, 0.0015609, -0.0053300, 0.0013300, -0.0067683, 0.0068909
1: -0.0035855, 0.0117203, -0.0044856, 0.0114998, -0.0150853, 0.0162059
2: 0.0046748, 0.0211602, 0.0040666, 0.0207120, -0.0149919, 0.0156326
3: -0.0072268, -0.0017345, -0.0069993, -0.0017582, -0.0054686, 0.0052649
4: 0.0032633, 0.0079192, 0.0032570, 0.0080813, -0.0045509, 0.0045689
5: -0.0059331, 0.0010410, -0.0060553, 0.0009895, -0.0069226, 0.0070963
6: -0.0069696, -0.0046552, -0.0068886, -0.0046063, -0.0023633, 0.0022334
7: -0.0058279, 0.0009175, -0.0057891, 0.0010180, -0.0068458, 0.0067066
8: -0.0085510, -0.0013634, -0.0082554, -0.0013952, -0.0071558, 0.0068919
9: 0.9934667, 1.0131572, 0.9937360, 1.0141370, -0.0206703, 0.0194212

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B2_B1_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130064, upper bound: 0.0125202
time: 2.06 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130064, upper bound: 0.0126761
time: 2.33 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0054251, 0.0015051, -0.0054251, 0.0015051, -0.0069302, 0.0069302
1: -0.0035944, 0.0116515, -0.0035944, 0.0116515, -0.0152459, 0.0152459
2: 0.0046760, 0.0210417, 0.0046760, 0.0210417, -0.0150026, 0.0150026
3: -0.0072004, -0.0017358, -0.0072004, -0.0017358, -0.0054646, 0.0054646
4: 0.0032311, 0.0079185, 0.0032311, 0.0079185, -0.0043034, 0.0043033
5: -0.0059257, 0.0010752, -0.0059257, 0.0010752, -0.0070008, 0.0070008
6: -0.0069546, -0.0046292, -0.0069546, -0.0046292, -0.0023254, 0.0023254
7: -0.0058100, 0.0009171, -0.0058100, 0.0009171, -0.0067272, 0.0067272
8: -0.0084685, -0.0013540, -0.0084685, -0.0013540, -0.0071145, 0.0071145
9: 0.9934891, 1.0131640, 0.9934891, 1.0131640, -0.0196750, 0.0196750

Time for backsubstitution: 2.32 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.47 + 595.64 = 600.10 seconds
