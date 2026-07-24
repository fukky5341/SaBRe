## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0134062


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010673, 0.0010673)
1: (-0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0059097, 0.0059097)
2: (0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0132030, 0.0132030)
3: (-0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0055638, 0.0055638)
4: (0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0215853, 0.0215853)
5: (0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0041992, 0.0041992)
6: (-0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0054646, 0.0054646)
7: (-0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006971, 0.0006971)
8: (-0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0037756, 0.0037756)
9: (-0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0189017, 0.0189017)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.09 + 2.58 = 4.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0157720, upper bound: 0.0157720

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0153024, upper bound: 0.0142492
time: 1.54 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0153024, upper bound: 0.0153024
time: 1.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.35 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.35
Output dim: 4, lower bound: -0.0153024, upper bound: 0.0142492
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.35
Output dim: 4, lower bound: -0.0153024, upper bound: 0.0153024

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0043403, -0.0033256, -0.0044193, -0.0033169, -0.0009793, 0.0010590
1: -0.0018383, 0.0037804, -0.0018863, 0.0042175, -0.0058638, 0.0054226
2: 0.0065204, 0.0190731, 0.0055439, 0.0191803, -0.0121146, 0.0131004
3: -0.0007031, 0.0045867, -0.0007483, 0.0049982, -0.0055205, 0.0051051
4: 0.9940226, 1.0145447, 0.9938473, 1.0161412, -0.0214176, 0.0198059
5: 0.0010786, 0.0050709, 0.0010445, 0.0053815, -0.0041666, 0.0038530
6: -0.0123420, -0.0071465, -0.0127462, -0.0071022, -0.0050142, 0.0054222
7: -0.0103777, -0.0097150, -0.0104293, -0.0097093, -0.0006396, 0.0006917
8: -0.0059164, -0.0023268, -0.0059471, -0.0020475, -0.0037463, 0.0034644
9: -0.0065227, 0.0114481, -0.0079206, 0.0116016, -0.0173435, 0.0187549

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0142492, upper bound: 0.0142492
time: 1.47 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0142492, upper bound: 0.0142492
time: 1.68 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0043832, -0.0032419, -0.0044138, -0.0033179, -0.0010247, 0.0011464
1: -0.0023018, 0.0040176, -0.0018810, 0.0041870, -0.0063474, 0.0056737
2: 0.0059903, 0.0201087, 0.0056119, 0.0191685, -0.0126758, 0.0141808
3: -0.0011395, 0.0048100, -0.0007433, 0.0049695, -0.0059758, 0.0053416
4: 0.9923295, 1.0154114, 0.9938666, 1.0160300, -0.0231840, 0.0207234
5: 0.0007492, 0.0052395, 0.0010482, 0.0053599, -0.0045102, 0.0040315
6: -0.0125614, -0.0067179, -0.0127180, -0.0071071, -0.0052464, 0.0058694
7: -0.0104057, -0.0096603, -0.0104257, -0.0097099, -0.0006692, 0.0007487
8: -0.0062126, -0.0021752, -0.0059437, -0.0020670, -0.0040552, 0.0036249
9: -0.0072815, 0.0129307, -0.0078232, 0.0115847, -0.0181470, 0.0203016

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0145181, upper bound: 0.0144168
time: 1.53 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0145181, upper bound: 0.0145181
time: 1.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.05 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.05
Output dim: 4, lower bound: -0.0142492, upper bound: 0.0142492
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.05
Output dim: 4, lower bound: -0.0142492, upper bound: 0.0142492
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.05
Output dim: 4, lower bound: -0.0145181, upper bound: 0.0144168
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.05
Output dim: 4, lower bound: -0.0145181, upper bound: 0.0145181

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043403, -0.0033256, -0.0043403, -0.0033256, -0.0009710, 0.0009710
1: -0.0018383, 0.0037804, -0.0018383, 0.0037804, -0.0053766, 0.0053766
2: 0.0065204, 0.0190731, 0.0065204, 0.0190731, -0.0120120, 0.0120120
3: -0.0007031, 0.0045867, -0.0007031, 0.0045867, -0.0050619, 0.0050619
4: 0.9940226, 1.0145447, 0.9940226, 1.0145447, -0.0196382, 0.0196382
5: 0.0010786, 0.0050709, 0.0010786, 0.0050709, -0.0038204, 0.0038204
6: -0.0123420, -0.0071465, -0.0123420, -0.0071465, -0.0049717, 0.0049717
7: -0.0103777, -0.0097150, -0.0103777, -0.0097150, -0.0006342, 0.0006342
8: -0.0059164, -0.0023268, -0.0059164, -0.0023268, -0.0034350, 0.0034350
9: -0.0065227, 0.0114481, -0.0065227, 0.0114481, -0.0171967, 0.0171967

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0136295, upper bound: 0.0135010
time: 1.55 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0137449, upper bound: 0.0135010
time: 1.60 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0043403, -0.0033256, -0.0043832, -0.0032419, -0.0010621, 0.0010362
1: -0.0018383, 0.0037804, -0.0023018, 0.0040176, -0.0057374, 0.0058809
2: 0.0065204, 0.0190731, 0.0059903, 0.0201087, -0.0131386, 0.0128181
3: -0.0007031, 0.0045867, -0.0011395, 0.0048100, -0.0054016, 0.0055366
4: 0.9940226, 1.0145447, 0.9923295, 1.0154114, -0.0209561, 0.0214800
5: 0.0010786, 0.0050709, 0.0007492, 0.0052395, -0.0040768, 0.0041787
6: -0.0123420, -0.0071465, -0.0125614, -0.0067179, -0.0054380, 0.0053053
7: -0.0103777, -0.0097150, -0.0104057, -0.0096603, -0.0006937, 0.0006767
8: -0.0059164, -0.0023268, -0.0062126, -0.0021752, -0.0036656, 0.0037572
9: -0.0065227, 0.0114481, -0.0072815, 0.0129307, -0.0188095, 0.0183507

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0136295, upper bound: 0.0135010
time: 1.86 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0137449, upper bound: 0.0135010
time: 1.81 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043831, -0.0032443, -0.0044102, -0.0033761, -0.0009520, 0.0011320
1: -0.0022883, 0.0040169, -0.0015588, 0.0041673, -0.0062676, 0.0052714
2: 0.0059920, 0.0200784, 0.0056558, 0.0184488, -0.0117768, 0.0140025
3: -0.0011268, 0.0048093, -0.0004400, 0.0049510, -0.0059007, 0.0049628
4: 0.9923789, 1.0154086, 0.9950433, 1.0159581, -0.0228924, 0.0192537
5: 0.0007588, 0.0052390, 0.0012771, 0.0053459, -0.0044535, 0.0037456
6: -0.0125607, -0.0067304, -0.0126999, -0.0074049, -0.0048744, 0.0057955
7: -0.0104056, -0.0096619, -0.0104233, -0.0097479, -0.0006218, 0.0007393
8: -0.0062039, -0.0021757, -0.0057379, -0.0020795, -0.0040042, 0.0033678
9: -0.0072791, 0.0128874, -0.0077603, 0.0105543, -0.0168600, 0.0200463

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0143811, upper bound: 0.0143811
time: 1.54 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0143811, upper bound: 0.0143811
time: 1.68 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043829, -0.0032489, -0.0044299, -0.0033696, -0.0009827, 0.0011656
1: -0.0022628, 0.0040159, -0.0015948, 0.0042761, -0.0064539, 0.0054409
2: 0.0059941, 0.0200214, 0.0054129, 0.0185292, -0.0121556, 0.0144188
3: -0.0011027, 0.0048084, -0.0004739, 0.0050533, -0.0060761, 0.0051224
4: 0.9924721, 1.0154051, 0.9949117, 1.0163554, -0.0235729, 0.0198730
5: 0.0007769, 0.0052383, 0.0012515, 0.0054232, -0.0045859, 0.0038661
6: -0.0125598, -0.0067540, -0.0128004, -0.0073717, -0.0050311, 0.0059678
7: -0.0104055, -0.0096649, -0.0104362, -0.0097437, -0.0006418, 0.0007613
8: -0.0061876, -0.0021763, -0.0057609, -0.0020101, -0.0041233, 0.0034761
9: -0.0072760, 0.0128058, -0.0081081, 0.0106695, -0.0174023, 0.0206422

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0143811, upper bound: 0.0145181
time: 1.53 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0143811, upper bound: 0.0145181
time: 1.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.30 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.30
Output dim: 4, lower bound: -0.0136295, upper bound: 0.0135010
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.30
Output dim: 4, lower bound: -0.0137449, upper bound: 0.0135010
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.30
Output dim: 4, lower bound: -0.0136295, upper bound: 0.0135010
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.30
Output dim: 4, lower bound: -0.0137449, upper bound: 0.0135010
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.30
Output dim: 4, lower bound: -0.0143811, upper bound: 0.0143811
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.30
Output dim: 4, lower bound: -0.0143811, upper bound: 0.0143811
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.30
Output dim: 4, lower bound: -0.0143811, upper bound: 0.0145181
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.30
Output dim: 4, lower bound: -0.0143811, upper bound: 0.0145181

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0043368, -0.0033838, -0.0043402, -0.0033281, -0.0009556, 0.0008987
1: -0.0015159, 0.0037610, -0.0018245, 0.0037797, -0.0049758, 0.0052913
2: 0.0065638, 0.0183527, 0.0065219, 0.0190423, -0.0118213, 0.0111166
3: -0.0003995, 0.0045684, -0.0006901, 0.0045860, -0.0046846, 0.0049815
4: 0.9952002, 1.0144738, 0.9940728, 1.0145421, -0.0181743, 0.0193263
5: 0.0013077, 0.0050571, 0.0010883, 0.0050704, -0.0035356, 0.0037597
6: -0.0123241, -0.0074447, -0.0123414, -0.0071593, -0.0048927, 0.0046011
7: -0.0103754, -0.0097530, -0.0103776, -0.0097166, -0.0006241, 0.0005869
8: -0.0057104, -0.0023392, -0.0059076, -0.0023272, -0.0031790, 0.0033805
9: -0.0064605, 0.0104169, -0.0065204, 0.0114041, -0.0169236, 0.0159148

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0135854, upper bound: 0.0135854
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0135854, upper bound: 0.0137459
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0043563, -0.0033774, -0.0043400, -0.0033327, -0.0009956, 0.0009293
1: -0.0015516, 0.0038687, -0.0017991, 0.0037787, -0.0051455, 0.0055124
2: 0.0063230, 0.0184327, 0.0065242, 0.0189855, -0.0123154, 0.0114957
3: -0.0004332, 0.0046698, -0.0006662, 0.0045850, -0.0048443, 0.0051897
4: 0.9950695, 1.0148674, 0.9941658, 1.0145385, -0.0187940, 0.0201341
5: 0.0012822, 0.0051337, 0.0011064, 0.0050697, -0.0036562, 0.0039169
6: -0.0124237, -0.0074116, -0.0123405, -0.0071828, -0.0050973, 0.0047580
7: -0.0103881, -0.0097488, -0.0103775, -0.0097196, -0.0006502, 0.0006069
8: -0.0057333, -0.0022703, -0.0058914, -0.0023278, -0.0032874, 0.0035218
9: -0.0068051, 0.0105313, -0.0065172, 0.0113227, -0.0176310, 0.0164575

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0137459, upper bound: 0.0135854
time: 1.45 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0137459, upper bound: 0.0137459
time: 1.44 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0043368, -0.0033838, -0.0043831, -0.0032443, -0.0010470, 0.0009638
1: -0.0015159, 0.0037610, -0.0022883, 0.0040169, -0.0053368, 0.0057972
2: 0.0065638, 0.0183527, 0.0059920, 0.0200784, -0.0129515, 0.0119230
3: -0.0003995, 0.0045684, -0.0011268, 0.0048093, -0.0050244, 0.0054578
4: 0.9952002, 1.0144738, 0.9923789, 1.0154086, -0.0194927, 0.0211742
5: 0.0013077, 0.0050571, 0.0007588, 0.0052390, -0.0037921, 0.0041192
6: -0.0123241, -0.0074447, -0.0125607, -0.0067304, -0.0053606, 0.0049349
7: -0.0103754, -0.0097530, -0.0104056, -0.0096619, -0.0006838, 0.0006295
8: -0.0057104, -0.0023392, -0.0062039, -0.0021757, -0.0034096, 0.0037037
9: -0.0064605, 0.0104169, -0.0072791, 0.0128874, -0.0185417, 0.0170693

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0143811, upper bound: 0.0133397
time: 1.54 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0143811, upper bound: 0.0135010
time: 1.51 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0043563, -0.0033774, -0.0043829, -0.0032489, -0.0010858, 0.0009944
1: -0.0015516, 0.0038687, -0.0022628, 0.0040159, -0.0055058, 0.0060121
2: 0.0063230, 0.0184327, 0.0059941, 0.0200214, -0.0134318, 0.0123006
3: -0.0004332, 0.0046698, -0.0011027, 0.0048084, -0.0051835, 0.0056602
4: 0.9950695, 1.0148674, 0.9924721, 1.0154051, -0.0201100, 0.0219593
5: 0.0012822, 0.0051337, 0.0007769, 0.0052383, -0.0039122, 0.0042720
6: -0.0124237, -0.0074116, -0.0125598, -0.0067540, -0.0055593, 0.0050911
7: -0.0103881, -0.0097488, -0.0104055, -0.0096649, -0.0007091, 0.0006494
8: -0.0057333, -0.0022703, -0.0061876, -0.0021763, -0.0035176, 0.0038410
9: -0.0068051, 0.0105313, -0.0072760, 0.0128058, -0.0192292, 0.0176098

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0145181, upper bound: 0.0133397
time: 1.46 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0145181, upper bound: 0.0135010
time: 1.49 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0043796, -0.0033013, -0.0044102, -0.0033761, -0.0009408, 0.0010656
1: -0.0019725, 0.0039978, -0.0015588, 0.0041673, -0.0059003, 0.0052092
2: 0.0060345, 0.0193730, 0.0056558, 0.0184488, -0.0116379, 0.0131819
3: -0.0008295, 0.0047914, -0.0004400, 0.0049510, -0.0055549, 0.0049043
4: 0.9935323, 1.0153390, 0.9950433, 1.0159581, -0.0215509, 0.0190266
5: 0.0009832, 0.0052255, 0.0012771, 0.0053459, -0.0041925, 0.0037014
6: -0.0125431, -0.0070224, -0.0126999, -0.0074049, -0.0048169, 0.0054559
7: -0.0104033, -0.0096991, -0.0104233, -0.0097479, -0.0006144, 0.0006960
8: -0.0060022, -0.0021878, -0.0057379, -0.0020795, -0.0037696, 0.0033281
9: -0.0072182, 0.0118775, -0.0077603, 0.0105543, -0.0166612, 0.0188715

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0144168
time: 1.55 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0144168
time: 1.90 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0043991, -0.0032925, -0.0044102, -0.0033761, -0.0009781, 0.0010852
1: -0.0020212, 0.0041057, -0.0015588, 0.0041673, -0.0060086, 0.0054155
2: 0.0057937, 0.0194817, 0.0056558, 0.0184488, -0.0120988, 0.0134239
3: -0.0008753, 0.0048929, -0.0004400, 0.0049510, -0.0056569, 0.0050985
4: 0.9933544, 1.0157328, 0.9950433, 1.0159581, -0.0219465, 0.0197801
5: 0.0009486, 0.0053021, 0.0012771, 0.0053459, -0.0042695, 0.0038480
6: -0.0126428, -0.0069774, -0.0126999, -0.0074049, -0.0050076, 0.0055561
7: -0.0104161, -0.0096934, -0.0104233, -0.0097479, -0.0006388, 0.0007087
8: -0.0060333, -0.0021189, -0.0057379, -0.0020795, -0.0038388, 0.0034599
9: -0.0075630, 0.0120332, -0.0077603, 0.0105543, -0.0173209, 0.0192180

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0144168
time: 1.58 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0144168
time: 1.62 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0043796, -0.0033013, -0.0044299, -0.0033696, -0.0009679, 0.0011031
1: -0.0019725, 0.0039978, -0.0015948, 0.0042761, -0.0061077, 0.0053591
2: 0.0060345, 0.0193730, 0.0054129, 0.0185292, -0.0119728, 0.0136452
3: -0.0008295, 0.0047914, -0.0004739, 0.0050533, -0.0057501, 0.0050453
4: 0.9935323, 1.0153390, 0.9949117, 1.0163554, -0.0223082, 0.0195740
5: 0.0009832, 0.0052255, 0.0012515, 0.0054232, -0.0043398, 0.0038079
6: -0.0125431, -0.0070224, -0.0128004, -0.0073717, -0.0049555, 0.0056477
7: -0.0104033, -0.0096991, -0.0104362, -0.0097437, -0.0006321, 0.0007204
8: -0.0060022, -0.0021878, -0.0057609, -0.0020101, -0.0039021, 0.0034238
9: -0.0072182, 0.0118775, -0.0081081, 0.0106695, -0.0171405, 0.0195348

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0145181
time: 1.62 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0145181
time: 1.81 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0043991, -0.0032925, -0.0044299, -0.0033696, -0.0009789, 0.0011030
1: -0.0020212, 0.0041057, -0.0015948, 0.0042761, -0.0061073, 0.0054201
2: 0.0057937, 0.0194817, 0.0054129, 0.0185292, -0.0121091, 0.0136443
3: -0.0008753, 0.0048929, -0.0004739, 0.0050533, -0.0057498, 0.0051028
4: 0.9933544, 1.0157328, 0.9949117, 1.0163554, -0.0223069, 0.0197969
5: 0.0009486, 0.0053021, 0.0012515, 0.0054232, -0.0043396, 0.0038513
6: -0.0126428, -0.0069774, -0.0128004, -0.0073717, -0.0050119, 0.0056473
7: -0.0104161, -0.0096934, -0.0104362, -0.0097437, -0.0006393, 0.0007204
8: -0.0060333, -0.0021189, -0.0057609, -0.0020101, -0.0039018, 0.0034628
9: -0.0075630, 0.0120332, -0.0081081, 0.0106695, -0.0173356, 0.0195336

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0143869
time: 1.65 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0143870
time: 1.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.10 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0135854, upper bound: 0.0135854
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0135854, upper bound: 0.0137459
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0137459, upper bound: 0.0135854
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0137459, upper bound: 0.0137459
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0143811, upper bound: 0.0133397
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0143811, upper bound: 0.0135010
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0145181, upper bound: 0.0133397
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0145181, upper bound: 0.0135010
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0144168
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0144168
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0144168
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0144168
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0145181
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0145181
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0143869
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 4, lower bound: -0.0133397, upper bound: 0.0143870

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043368, -0.0033838, -0.0043368, -0.0033838, -0.0008869, 0.0008869
1: -0.0015159, 0.0037610, -0.0015159, 0.0037610, -0.0049106, 0.0049106
2: 0.0065638, 0.0183527, 0.0065638, 0.0183527, -0.0109708, 0.0109708
3: -0.0003995, 0.0045684, -0.0003995, 0.0045684, -0.0046231, 0.0046231
4: 0.9952002, 1.0144738, 0.9952002, 1.0144738, -0.0179360, 0.0179360
5: 0.0013077, 0.0050571, 0.0013077, 0.0050571, -0.0034893, 0.0034893
6: -0.0123241, -0.0074447, -0.0123241, -0.0074447, -0.0045408, 0.0045408
7: -0.0103754, -0.0097530, -0.0103754, -0.0097530, -0.0005792, 0.0005792
8: -0.0057104, -0.0023392, -0.0057104, -0.0023392, -0.0031373, 0.0031373
9: -0.0064605, 0.0104169, -0.0064605, 0.0104169, -0.0157061, 0.0157061

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0132021, upper bound: 0.0129549
time: 1.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0132021, upper bound: 0.0131544
time: 1.47 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0043368, -0.0033838, -0.0043563, -0.0033774, -0.0009138, 0.0009295
1: -0.0015159, 0.0037610, -0.0015516, 0.0038687, -0.0051466, 0.0050597
2: 0.0065638, 0.0183527, 0.0063230, 0.0184327, -0.0113040, 0.0114980
3: -0.0003995, 0.0045684, -0.0004332, 0.0046698, -0.0048453, 0.0047635
4: 0.9952002, 1.0144738, 0.9950695, 1.0148674, -0.0187979, 0.0184807
5: 0.0013077, 0.0050571, 0.0012822, 0.0051337, -0.0036569, 0.0035952
6: -0.0123241, -0.0074447, -0.0124237, -0.0074116, -0.0046787, 0.0047590
7: -0.0103754, -0.0097530, -0.0103881, -0.0097488, -0.0005968, 0.0006071
8: -0.0057104, -0.0023392, -0.0057333, -0.0022703, -0.0032881, 0.0032326
9: -0.0064605, 0.0104169, -0.0068051, 0.0105313, -0.0161831, 0.0164609

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0132021, upper bound: 0.0131426
time: 1.45 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0132021, upper bound: 0.0133036
time: 1.45 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043563, -0.0033774, -0.0043368, -0.0033838, -0.0009295, 0.0009138
1: -0.0015516, 0.0038687, -0.0015159, 0.0037610, -0.0050597, 0.0051466
2: 0.0063230, 0.0184327, 0.0065638, 0.0183527, -0.0114980, 0.0113040
3: -0.0004332, 0.0046698, -0.0003995, 0.0045684, -0.0047635, 0.0048453
4: 0.9950695, 1.0148674, 0.9952002, 1.0144738, -0.0184807, 0.0187979
5: 0.0012822, 0.0051337, 0.0013077, 0.0050571, -0.0035952, 0.0036569
6: -0.0124237, -0.0074116, -0.0123241, -0.0074447, -0.0047590, 0.0046787
7: -0.0103881, -0.0097488, -0.0103754, -0.0097530, -0.0006071, 0.0005968
8: -0.0057333, -0.0022703, -0.0057104, -0.0023392, -0.0032326, 0.0032881
9: -0.0068051, 0.0105313, -0.0064605, 0.0104169, -0.0164609, 0.0161831

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0133036, upper bound: 0.0129549
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0133036, upper bound: 0.0131544
time: 1.54 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043563, -0.0033774, -0.0043563, -0.0033774, -0.0009260, 0.0009260
1: -0.0015516, 0.0038687, -0.0015516, 0.0038687, -0.0051274, 0.0051274
2: 0.0063230, 0.0184327, 0.0063230, 0.0184327, -0.0114552, 0.0114552
3: -0.0004332, 0.0046698, -0.0004332, 0.0046698, -0.0048273, 0.0048273
4: 0.9950695, 1.0148674, 0.9950695, 1.0148674, -0.0187279, 0.0187279
5: 0.0012822, 0.0051337, 0.0012822, 0.0051337, -0.0036433, 0.0036433
6: -0.0124237, -0.0074116, -0.0124237, -0.0074116, -0.0047412, 0.0047412
7: -0.0103881, -0.0097488, -0.0103881, -0.0097488, -0.0006048, 0.0006048
8: -0.0057333, -0.0022703, -0.0057333, -0.0022703, -0.0032758, 0.0032758
9: -0.0068051, 0.0105313, -0.0068051, 0.0105313, -0.0163995, 0.0163995

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0133036, upper bound: 0.0129606
time: 1.52 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0133036, upper bound: 0.0131595
time: 1.50 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043368, -0.0033838, -0.0043796, -0.0033013, -0.0009807, 0.0009528
1: -0.0015159, 0.0037610, -0.0019725, 0.0039978, -0.0052754, 0.0054299
2: 0.0065638, 0.0183527, 0.0060345, 0.0193730, -0.0121310, 0.0117858
3: -0.0003995, 0.0045684, -0.0008295, 0.0047914, -0.0049665, 0.0051120
4: 0.9952002, 1.0144738, 0.9935323, 1.0153390, -0.0192683, 0.0198327
5: 0.0013077, 0.0050571, 0.0009832, 0.0052255, -0.0037485, 0.0038582
6: -0.0123241, -0.0074447, -0.0125431, -0.0070224, -0.0050209, 0.0048781
7: -0.0103754, -0.0097530, -0.0104033, -0.0096991, -0.0006405, 0.0006222
8: -0.0057104, -0.0023392, -0.0060022, -0.0021878, -0.0033703, 0.0034691
9: -0.0064605, 0.0104169, -0.0072182, 0.0118775, -0.0173670, 0.0168728

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139248, upper bound: 0.0126811
time: 1.38 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139248, upper bound: 0.0129250
time: 1.47 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0043368, -0.0033838, -0.0043991, -0.0032925, -0.0010002, 0.0009851
1: -0.0015159, 0.0037610, -0.0020212, 0.0041057, -0.0054545, 0.0055382
2: 0.0065638, 0.0183527, 0.0057937, 0.0194817, -0.0123730, 0.0121859
3: -0.0003995, 0.0045684, -0.0008753, 0.0048929, -0.0051352, 0.0052140
4: 0.9952002, 1.0144738, 0.9933544, 1.0157328, -0.0199226, 0.0202283
5: 0.0013077, 0.0050571, 0.0009486, 0.0053021, -0.0038757, 0.0039352
6: -0.0123241, -0.0074447, -0.0126428, -0.0069774, -0.0051211, 0.0050437
7: -0.0103754, -0.0097530, -0.0104161, -0.0096934, -0.0006532, 0.0006434
8: -0.0057104, -0.0023392, -0.0060333, -0.0021189, -0.0034848, 0.0035383
9: -0.0064605, 0.0104169, -0.0075630, 0.0120332, -0.0177134, 0.0174457

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139248, upper bound: 0.0128705
time: 1.57 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139248, upper bound: 0.0130772
time: 1.48 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043563, -0.0033774, -0.0043796, -0.0033013, -0.0010233, 0.0009797
1: -0.0015516, 0.0038687, -0.0019725, 0.0039978, -0.0054245, 0.0056659
2: 0.0063230, 0.0184327, 0.0060345, 0.0193730, -0.0126582, 0.0121190
3: -0.0004332, 0.0046698, -0.0008295, 0.0047914, -0.0051070, 0.0053342
4: 0.9950695, 1.0148674, 0.9935323, 1.0153390, -0.0198131, 0.0206946
5: 0.0012822, 0.0051337, 0.0009832, 0.0052255, -0.0038544, 0.0040259
6: -0.0124237, -0.0074116, -0.0125431, -0.0070224, -0.0052391, 0.0050160
7: -0.0103881, -0.0097488, -0.0104033, -0.0096991, -0.0006683, 0.0006398
8: -0.0057333, -0.0022703, -0.0060022, -0.0021878, -0.0034656, 0.0036198
9: -0.0068051, 0.0105313, -0.0072182, 0.0118775, -0.0181218, 0.0173498

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0140090, upper bound: 0.0126811
time: 1.54 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0140090, upper bound: 0.0129250
time: 1.52 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043563, -0.0033774, -0.0043991, -0.0032925, -0.0010191, 0.0009904
1: -0.0015516, 0.0038687, -0.0020212, 0.0041057, -0.0054840, 0.0056427
2: 0.0063230, 0.0184327, 0.0057937, 0.0194817, -0.0126064, 0.0122520
3: -0.0004332, 0.0046698, -0.0008753, 0.0048929, -0.0051630, 0.0053124
4: 0.9950695, 1.0148674, 0.9933544, 1.0157328, -0.0200305, 0.0206100
5: 0.0012822, 0.0051337, 0.0009486, 0.0053021, -0.0038967, 0.0040095
6: -0.0124237, -0.0074116, -0.0126428, -0.0069774, -0.0052177, 0.0050710
7: -0.0103881, -0.0097488, -0.0104161, -0.0096934, -0.0006656, 0.0006469
8: -0.0057333, -0.0022703, -0.0060333, -0.0021189, -0.0035037, 0.0036050
9: -0.0068051, 0.0105313, -0.0075630, 0.0120332, -0.0180476, 0.0175402

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0140090, upper bound: 0.0126860
time: 1.70 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0140090, upper bound: 0.0129264
time: 1.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043796, -0.0033013, -0.0043368, -0.0033838, -0.0009528, 0.0009807
1: -0.0019725, 0.0039978, -0.0015159, 0.0037610, -0.0054299, 0.0052754
2: 0.0060345, 0.0193730, 0.0065638, 0.0183527, -0.0117858, 0.0121310
3: -0.0008295, 0.0047914, -0.0003995, 0.0045684, -0.0051120, 0.0049665
4: 0.9935323, 1.0153390, 0.9952002, 1.0144738, -0.0198327, 0.0192683
5: 0.0009832, 0.0052255, 0.0013077, 0.0050571, -0.0038582, 0.0037485
6: -0.0125431, -0.0070224, -0.0123241, -0.0074447, -0.0048781, 0.0050209
7: -0.0104033, -0.0096991, -0.0103754, -0.0097530, -0.0006222, 0.0006405
8: -0.0060022, -0.0021878, -0.0057104, -0.0023392, -0.0034691, 0.0033703
9: -0.0072182, 0.0118775, -0.0064605, 0.0104169, -0.0168728, 0.0173670

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0129905, upper bound: 0.0137652
time: 1.47 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0129905, upper bound: 0.0139278
time: 1.41 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0043796, -0.0033013, -0.0043796, -0.0033013, -0.0009797, 0.0009797
1: -0.0019725, 0.0039978, -0.0019725, 0.0039978, -0.0054246, 0.0054246
2: 0.0060345, 0.0193730, 0.0060345, 0.0193730, -0.0121191, 0.0121191
3: -0.0008295, 0.0047914, -0.0008295, 0.0047914, -0.0051070, 0.0051070
4: 0.9935323, 1.0153390, 0.9935323, 1.0153390, -0.0198132, 0.0198132
5: 0.0009832, 0.0052255, 0.0009832, 0.0052255, -0.0038545, 0.0038545
6: -0.0125431, -0.0070224, -0.0125431, -0.0070224, -0.0050160, 0.0050160
7: -0.0104033, -0.0096991, -0.0104033, -0.0096991, -0.0006398, 0.0006398
8: -0.0060022, -0.0021878, -0.0060022, -0.0021878, -0.0034657, 0.0034657
9: -0.0072182, 0.0118775, -0.0072182, 0.0118775, -0.0173500, 0.0173500

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0129905, upper bound: 0.0137652
time: 1.55 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0129905, upper bound: 0.0139278
time: 1.63 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043991, -0.0032925, -0.0043368, -0.0033838, -0.0009851, 0.0010002
1: -0.0020212, 0.0041057, -0.0015159, 0.0037610, -0.0055382, 0.0054545
2: 0.0057937, 0.0194817, 0.0065638, 0.0183527, -0.0121859, 0.0123730
3: -0.0008753, 0.0048929, -0.0003995, 0.0045684, -0.0052140, 0.0051352
4: 0.9933544, 1.0157328, 0.9952002, 1.0144738, -0.0202283, 0.0199226
5: 0.0009486, 0.0053021, 0.0013077, 0.0050571, -0.0039352, 0.0038757
6: -0.0126428, -0.0069774, -0.0123241, -0.0074447, -0.0050437, 0.0051211
7: -0.0104161, -0.0096934, -0.0103754, -0.0097530, -0.0006434, 0.0006532
8: -0.0060333, -0.0021189, -0.0057104, -0.0023392, -0.0035383, 0.0034848
9: -0.0075630, 0.0120332, -0.0064605, 0.0104169, -0.0174457, 0.0177134

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0137587
time: 1.40 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0139248
time: 1.50 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043991, -0.0032925, -0.0043796, -0.0033013, -0.0010170, 0.0010026
1: -0.0020212, 0.0041057, -0.0019725, 0.0039978, -0.0055512, 0.0056308
2: 0.0057937, 0.0194817, 0.0060345, 0.0193730, -0.0125799, 0.0124019
3: -0.0008753, 0.0048929, -0.0008295, 0.0047914, -0.0052262, 0.0053012
4: 0.9933544, 1.0157328, 0.9935323, 1.0153390, -0.0202756, 0.0205667
5: 0.0009486, 0.0053021, 0.0009832, 0.0052255, -0.0039444, 0.0040010
6: -0.0126428, -0.0069774, -0.0125431, -0.0070224, -0.0052068, 0.0051331
7: -0.0104161, -0.0096934, -0.0104033, -0.0096991, -0.0006642, 0.0006548
8: -0.0060333, -0.0021189, -0.0060022, -0.0021878, -0.0035465, 0.0035974
9: -0.0075630, 0.0120332, -0.0072182, 0.0118775, -0.0180097, 0.0177549

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0137587
time: 1.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0139248
time: 1.75 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043796, -0.0033013, -0.0043563, -0.0033774, -0.0009797, 0.0010233
1: -0.0019725, 0.0039978, -0.0015516, 0.0038687, -0.0056659, 0.0054245
2: 0.0060345, 0.0193730, 0.0063230, 0.0184327, -0.0121190, 0.0126582
3: -0.0008295, 0.0047914, -0.0004332, 0.0046698, -0.0053342, 0.0051070
4: 0.9935323, 1.0153390, 0.9950695, 1.0148674, -0.0206946, 0.0198131
5: 0.0009832, 0.0052255, 0.0012822, 0.0051337, -0.0040259, 0.0038544
6: -0.0125431, -0.0070224, -0.0124237, -0.0074116, -0.0050160, 0.0052391
7: -0.0104033, -0.0096991, -0.0103881, -0.0097488, -0.0006398, 0.0006683
8: -0.0060022, -0.0021878, -0.0057333, -0.0022703, -0.0036198, 0.0034656
9: -0.0072182, 0.0118775, -0.0068051, 0.0105313, -0.0173498, 0.0181218

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0129819, upper bound: 0.0139003
time: 1.46 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0129819, upper bound: 0.0140090
time: 1.55 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0043796, -0.0033013, -0.0043991, -0.0032925, -0.0010026, 0.0010170
1: -0.0019725, 0.0039978, -0.0020212, 0.0041057, -0.0056308, 0.0055512
2: 0.0060345, 0.0193730, 0.0057937, 0.0194817, -0.0124019, 0.0125799
3: -0.0008295, 0.0047914, -0.0008753, 0.0048929, -0.0053012, 0.0052262
4: 0.9935323, 1.0153390, 0.9933544, 1.0157328, -0.0205667, 0.0202756
5: 0.0009832, 0.0052255, 0.0009486, 0.0053021, -0.0040010, 0.0039444
6: -0.0125431, -0.0070224, -0.0126428, -0.0069774, -0.0051331, 0.0052068
7: -0.0104033, -0.0096991, -0.0104161, -0.0096934, -0.0006548, 0.0006642
8: -0.0060022, -0.0021878, -0.0060333, -0.0021189, -0.0035974, 0.0035465
9: -0.0072182, 0.0118775, -0.0075630, 0.0120332, -0.0177549, 0.0180097

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0129819, upper bound: 0.0139003
time: 1.79 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0129819, upper bound: 0.0140090
time: 1.61 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043991, -0.0032925, -0.0043563, -0.0033774, -0.0009904, 0.0010191
1: -0.0020212, 0.0041057, -0.0015516, 0.0038687, -0.0056427, 0.0054840
2: 0.0057937, 0.0194817, 0.0063230, 0.0184327, -0.0122520, 0.0126064
3: -0.0008753, 0.0048929, -0.0004332, 0.0046698, -0.0053124, 0.0051630
4: 0.9933544, 1.0157328, 0.9950695, 1.0148674, -0.0206100, 0.0200305
5: 0.0009486, 0.0053021, 0.0012822, 0.0051337, -0.0040095, 0.0038967
6: -0.0126428, -0.0069774, -0.0124237, -0.0074116, -0.0050710, 0.0052177
7: -0.0104161, -0.0096934, -0.0103881, -0.0097488, -0.0006469, 0.0006656
8: -0.0060333, -0.0021189, -0.0057333, -0.0022703, -0.0036050, 0.0035037
9: -0.0075630, 0.0120332, -0.0068051, 0.0105313, -0.0175402, 0.0180476

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0137460
time: 1.52 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0138907
time: 1.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043991, -0.0032925, -0.0043991, -0.0032925, -0.0010176, 0.0010176
1: -0.0020212, 0.0041057, -0.0020212, 0.0041057, -0.0056344, 0.0056344
2: 0.0057937, 0.0194817, 0.0057937, 0.0194817, -0.0125878, 0.0125878
3: -0.0008753, 0.0048929, -0.0008753, 0.0048929, -0.0053045, 0.0053045
4: 0.9933544, 1.0157328, 0.9933544, 1.0157328, -0.0205796, 0.0205796
5: 0.0009486, 0.0053021, 0.0009486, 0.0053021, -0.0040035, 0.0040035
6: -0.0126428, -0.0069774, -0.0126428, -0.0069774, -0.0052100, 0.0052100
7: -0.0104161, -0.0096934, -0.0104161, -0.0096934, -0.0006646, 0.0006646
8: -0.0060333, -0.0021189, -0.0060333, -0.0021189, -0.0035997, 0.0035997
9: -0.0075630, 0.0120332, -0.0075630, 0.0120332, -0.0180210, 0.0180210

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0137460
time: 1.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0138907
time: 1.56 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.25 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0132021, upper bound: 0.0129549
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0132021, upper bound: 0.0131544
NS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0132021, upper bound: 0.0131426
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0132021, upper bound: 0.0133036
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0133036, upper bound: 0.0129549
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0133036, upper bound: 0.0131544
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0133036, upper bound: 0.0129606
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0133036, upper bound: 0.0131595
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0139248, upper bound: 0.0126811
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0139248, upper bound: 0.0129250
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0139248, upper bound: 0.0128705
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0139248, upper bound: 0.0130772
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0140090, upper bound: 0.0126811
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0140090, upper bound: 0.0129250
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0140090, upper bound: 0.0126860
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0140090, upper bound: 0.0129264
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0129905, upper bound: 0.0137652
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0129905, upper bound: 0.0139278
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0129905, upper bound: 0.0137652
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0129905, upper bound: 0.0139278
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0137587
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0139248
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0137587
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0139248
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0129819, upper bound: 0.0139003
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0129819, upper bound: 0.0140090
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0129819, upper bound: 0.0139003
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0129819, upper bound: 0.0140090
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0137460
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0138907
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0137460
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 4, lower bound: -0.0130772, upper bound: 0.0138907

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042948, -0.0034131, -0.0043764, -0.0033020, -0.0009258, 0.0009046
1: -0.0013538, 0.0035283, -0.0019688, 0.0039799, -0.0050085, 0.0051261
2: 0.0070836, 0.0179907, 0.0060746, 0.0193647, -0.0114523, 0.0111895
3: -0.0002470, 0.0043493, -0.0008260, 0.0047745, -0.0047153, 0.0048260
4: 0.9957922, 1.0136240, 0.9935458, 1.0152735, -0.0182935, 0.0187232
5: 0.0014228, 0.0048918, 0.0009858, 0.0052127, -0.0035588, 0.0036424
6: -0.0121089, -0.0075945, -0.0125265, -0.0070258, -0.0047401, 0.0046313
7: -0.0103480, -0.0097721, -0.0104012, -0.0096996, -0.0006046, 0.0005908
8: -0.0056069, -0.0024878, -0.0059998, -0.0021993, -0.0031998, 0.0032750
9: -0.0057163, 0.0098986, -0.0071608, 0.0118656, -0.0163954, 0.0160192

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0137652, upper bound: 0.0127283
time: 1.66 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0137652, upper bound: 0.0127283
time: 1.71 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0043104, -0.0033899, -0.0043778, -0.0033017, -0.0009212, 0.0009461
1: -0.0014823, 0.0036144, -0.0019704, 0.0039879, -0.0052385, 0.0051008
2: 0.0068911, 0.0182778, 0.0060568, 0.0193682, -0.0113958, 0.0117035
3: -0.0003680, 0.0044304, -0.0008274, 0.0047820, -0.0049319, 0.0048022
4: 0.9953227, 1.0139387, 0.9935401, 1.0153028, -0.0191338, 0.0186308
5: 0.0013315, 0.0049530, 0.0009847, 0.0052184, -0.0037223, 0.0036244
6: -0.0121886, -0.0074757, -0.0125339, -0.0070244, -0.0047167, 0.0048440
7: -0.0103581, -0.0097570, -0.0104022, -0.0096994, -0.0006017, 0.0006179
8: -0.0056890, -0.0024328, -0.0060008, -0.0021942, -0.0033468, 0.0032588
9: -0.0059918, 0.0103096, -0.0071863, 0.0118706, -0.0163145, 0.0167550

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0137652, upper bound: 0.0129905
time: 1.58 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0137652, upper bound: 0.0129905
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042948, -0.0034131, -0.0043959, -0.0032932, -0.0009472, 0.0009374
1: -0.0013538, 0.0035283, -0.0020173, 0.0040878, -0.0051902, 0.0052445
2: 0.0070836, 0.0179907, 0.0058335, 0.0194731, -0.0117168, 0.0115955
3: -0.0002470, 0.0043493, -0.0008717, 0.0048761, -0.0048864, 0.0049375
4: 0.9957922, 1.0136240, 0.9933686, 1.0156677, -0.0189572, 0.0191555
5: 0.0014228, 0.0048918, 0.0009513, 0.0052894, -0.0036879, 0.0037265
6: -0.0121089, -0.0075945, -0.0126263, -0.0069810, -0.0048495, 0.0047993
7: -0.0103480, -0.0097721, -0.0104140, -0.0096938, -0.0006186, 0.0006122
8: -0.0056069, -0.0024878, -0.0060308, -0.0021303, -0.0033159, 0.0033506
9: -0.0057163, 0.0098986, -0.0075059, 0.0120208, -0.0167740, 0.0166004

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0137587, upper bound: 0.0128704
time: 1.52 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0137587, upper bound: 0.0128705
time: 1.84 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0043104, -0.0033899, -0.0043973, -0.0032930, -0.0009463, 0.0009783
1: -0.0014823, 0.0036144, -0.0020190, 0.0040956, -0.0054169, 0.0052398
2: 0.0068911, 0.0182778, 0.0058162, 0.0194767, -0.0117062, 0.0121020
3: -0.0003680, 0.0044304, -0.0008732, 0.0048834, -0.0050998, 0.0049330
4: 0.9953227, 1.0139387, 0.9933627, 1.0156960, -0.0197854, 0.0191382
5: 0.0013315, 0.0049530, 0.0009502, 0.0052949, -0.0038490, 0.0037231
6: -0.0121886, -0.0074757, -0.0126335, -0.0069795, -0.0048451, 0.0050090
7: -0.0103581, -0.0097570, -0.0104149, -0.0096937, -0.0006180, 0.0006389
8: -0.0056890, -0.0024328, -0.0060318, -0.0021254, -0.0034608, 0.0033476
9: -0.0059918, 0.0103096, -0.0075307, 0.0120260, -0.0167589, 0.0173255

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0137587, upper bound: 0.0130772
time: 1.45 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0137587, upper bound: 0.0130772
time: 1.91 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0043140, -0.0034045, -0.0043764, -0.0033020, -0.0009780, 0.0009402
1: -0.0014012, 0.0036348, -0.0019688, 0.0039799, -0.0052060, 0.0054150
2: 0.0068456, 0.0180967, 0.0060746, 0.0193647, -0.0120977, 0.0116308
3: -0.0002916, 0.0044496, -0.0008260, 0.0047745, -0.0049013, 0.0050980
4: 0.9956189, 1.0140129, 0.9935458, 1.0152735, -0.0190150, 0.0197782
5: 0.0013891, 0.0049675, 0.0009858, 0.0052127, -0.0036992, 0.0038477
6: -0.0122074, -0.0075507, -0.0125265, -0.0070258, -0.0050072, 0.0048139
7: -0.0103605, -0.0097665, -0.0104012, -0.0096996, -0.0006387, 0.0006141
8: -0.0056372, -0.0024198, -0.0059998, -0.0021993, -0.0033260, 0.0034595
9: -0.0060570, 0.0100503, -0.0071608, 0.0118656, -0.0173193, 0.0166509

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0127173
time: 1.62 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0127173
time: 1.78 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0043295, -0.0033837, -0.0043778, -0.0033017, -0.0009738, 0.0009726
1: -0.0015163, 0.0037204, -0.0019704, 0.0039879, -0.0053851, 0.0053918
2: 0.0066544, 0.0183536, 0.0060568, 0.0193682, -0.0120458, 0.0120308
3: -0.0003999, 0.0045302, -0.0008274, 0.0047820, -0.0050698, 0.0050761
4: 0.9951988, 1.0143256, 0.9935401, 1.0153028, -0.0196689, 0.0196934
5: 0.0013074, 0.0050283, 0.0009847, 0.0052184, -0.0038264, 0.0038312
6: -0.0122866, -0.0074443, -0.0125339, -0.0070244, -0.0049857, 0.0049795
7: -0.0103706, -0.0097530, -0.0104022, -0.0096994, -0.0006360, 0.0006352
8: -0.0057107, -0.0023651, -0.0060008, -0.0021942, -0.0034404, 0.0034447
9: -0.0063307, 0.0104181, -0.0071863, 0.0118706, -0.0172450, 0.0172236

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0129819
time: 1.53 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0129819
time: 1.77 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0043140, -0.0034045, -0.0043959, -0.0032932, -0.0009722, 0.0009414
1: -0.0014012, 0.0036348, -0.0020173, 0.0040878, -0.0052123, 0.0053829
2: 0.0068456, 0.0180967, 0.0058335, 0.0194731, -0.0120260, 0.0116447
3: -0.0002916, 0.0044496, -0.0008717, 0.0048761, -0.0049071, 0.0050678
4: 0.9956189, 1.0140129, 0.9933686, 1.0156677, -0.0190378, 0.0196611
5: 0.0013891, 0.0049675, 0.0009513, 0.0052894, -0.0037036, 0.0038249
6: -0.0122074, -0.0075507, -0.0126263, -0.0069810, -0.0049775, 0.0048197
7: -0.0103605, -0.0097665, -0.0104140, -0.0096938, -0.0006349, 0.0006148
8: -0.0056372, -0.0024198, -0.0060308, -0.0021303, -0.0033300, 0.0034390
9: -0.0060570, 0.0100503, -0.0075059, 0.0120208, -0.0172168, 0.0166709

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0126860
time: 1.46 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0126860
time: 1.84 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0043295, -0.0033837, -0.0043973, -0.0032930, -0.0009681, 0.0009836
1: -0.0015163, 0.0037204, -0.0020190, 0.0040956, -0.0054463, 0.0053604
2: 0.0066544, 0.0183536, 0.0058162, 0.0194767, -0.0119758, 0.0121676
3: -0.0003999, 0.0045302, -0.0008732, 0.0048834, -0.0051275, 0.0050466
4: 0.9951988, 1.0143256, 0.9933627, 1.0156960, -0.0198926, 0.0195790
5: 0.0013074, 0.0050283, 0.0009502, 0.0052949, -0.0038699, 0.0038089
6: -0.0122866, -0.0074443, -0.0126335, -0.0069795, -0.0049567, 0.0050361
7: -0.0103706, -0.0097530, -0.0104149, -0.0096937, -0.0006323, 0.0006424
8: -0.0057107, -0.0023651, -0.0060318, -0.0021254, -0.0034795, 0.0034247
9: -0.0063307, 0.0104181, -0.0075307, 0.0120260, -0.0171449, 0.0174194

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0129264
time: 1.53 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0129264
time: 1.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0043374, -0.0033275, -0.0043335, -0.0033845, -0.0009094, 0.0009443
1: -0.0018278, 0.0037643, -0.0015122, 0.0037426, -0.0052285, 0.0050355
2: 0.0065563, 0.0190496, 0.0066048, 0.0183446, -0.0112500, 0.0116810
3: -0.0006932, 0.0045715, -0.0003961, 0.0045511, -0.0049224, 0.0047408
4: 0.9940610, 1.0144860, 0.9952136, 1.0144067, -0.0190970, 0.0183923
5: 0.0010860, 0.0050595, 0.0013103, 0.0050441, -0.0037151, 0.0035780
6: -0.0123272, -0.0071562, -0.0123071, -0.0074481, -0.0046563, 0.0048347
7: -0.0103758, -0.0097162, -0.0103732, -0.0097534, -0.0005940, 0.0006167
8: -0.0059097, -0.0023370, -0.0057081, -0.0023509, -0.0033404, 0.0032171
9: -0.0064712, 0.0114145, -0.0064018, 0.0104052, -0.0161057, 0.0167228

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127283, upper bound: 0.0137652
time: 1.63 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127283, upper bound: 0.0137652
time: 1.95 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0043511, -0.0033076, -0.0043351, -0.0033842, -0.0008962, 0.0009738
1: -0.0019376, 0.0038402, -0.0015138, 0.0037514, -0.0053917, 0.0049625
2: 0.0063867, 0.0192950, 0.0065850, 0.0183481, -0.0110867, 0.0120457
3: -0.0007966, 0.0046430, -0.0003976, 0.0045594, -0.0050761, 0.0046720
4: 0.9936598, 1.0147634, 0.9952078, 1.0144390, -0.0196932, 0.0181254
5: 0.0010080, 0.0051135, 0.0013091, 0.0050504, -0.0038311, 0.0035261
6: -0.0123974, -0.0070547, -0.0123153, -0.0074466, -0.0045887, 0.0049856
7: -0.0103848, -0.0097033, -0.0103743, -0.0097532, -0.0005853, 0.0006360
8: -0.0059799, -0.0022885, -0.0057091, -0.0023452, -0.0034447, 0.0031704
9: -0.0067141, 0.0117658, -0.0064301, 0.0104102, -0.0158720, 0.0172449

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127283, upper bound: 0.0139278
time: 1.47 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127283, upper bound: 0.0139278
time: 1.57 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0043374, -0.0033275, -0.0043764, -0.0033020, -0.0009352, 0.0009388
1: -0.0018278, 0.0037643, -0.0019688, 0.0039799, -0.0051982, 0.0051780
2: 0.0065563, 0.0190496, 0.0060746, 0.0193647, -0.0115682, 0.0116133
3: -0.0006932, 0.0045715, -0.0008260, 0.0047745, -0.0048939, 0.0048749
4: 0.9940610, 1.0144860, 0.9935458, 1.0152735, -0.0189864, 0.0189126
5: 0.0010860, 0.0050595, 0.0009858, 0.0052127, -0.0036936, 0.0036793
6: -0.0123272, -0.0071562, -0.0125265, -0.0070258, -0.0047880, 0.0048067
7: -0.0103758, -0.0097162, -0.0104012, -0.0096996, -0.0006108, 0.0006131
8: -0.0059097, -0.0023370, -0.0059998, -0.0021993, -0.0033210, 0.0033081
9: -0.0064712, 0.0114145, -0.0071608, 0.0118656, -0.0165613, 0.0166259

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127383, upper bound: 0.0137652
time: 1.54 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127383, upper bound: 0.0137652
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0043511, -0.0033076, -0.0043778, -0.0033017, -0.0009267, 0.0009728
1: -0.0019376, 0.0038402, -0.0019704, 0.0039879, -0.0053866, 0.0051313
2: 0.0063867, 0.0192950, 0.0060568, 0.0193682, -0.0114640, 0.0120343
3: -0.0007966, 0.0046430, -0.0008274, 0.0047820, -0.0050713, 0.0048309
4: 0.9936598, 1.0147634, 0.9935401, 1.0153028, -0.0196747, 0.0187422
5: 0.0010080, 0.0051135, 0.0009847, 0.0052184, -0.0038275, 0.0036461
6: -0.0123974, -0.0070547, -0.0125339, -0.0070244, -0.0047449, 0.0049809
7: -0.0103848, -0.0097033, -0.0104022, -0.0096994, -0.0006053, 0.0006354
8: -0.0059799, -0.0022885, -0.0060008, -0.0021942, -0.0034414, 0.0032783
9: -0.0067141, 0.0117658, -0.0071863, 0.0118706, -0.0164121, 0.0172287

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127383, upper bound: 0.0139278
time: 1.44 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127383, upper bound: 0.0139278
time: 1.60 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0043570, -0.0033173, -0.0043335, -0.0033845, -0.0009422, 0.0009687
1: -0.0018839, 0.0038728, -0.0015122, 0.0037426, -0.0053639, 0.0052171
2: 0.0063139, 0.0191750, 0.0066048, 0.0183446, -0.0116555, 0.0119835
3: -0.0007460, 0.0046736, -0.0003961, 0.0045511, -0.0050499, 0.0049117
4: 0.9938560, 1.0148822, 0.9952136, 1.0144067, -0.0195916, 0.0190554
5: 0.0010461, 0.0051366, 0.0013103, 0.0050441, -0.0038113, 0.0037070
6: -0.0124275, -0.0071044, -0.0123071, -0.0074481, -0.0048242, 0.0049599
7: -0.0103886, -0.0097096, -0.0103732, -0.0097534, -0.0006154, 0.0006327
8: -0.0059456, -0.0022677, -0.0057081, -0.0023509, -0.0034269, 0.0033331
9: -0.0068182, 0.0115940, -0.0064018, 0.0104052, -0.0166863, 0.0171559

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128704, upper bound: 0.0137587
time: 1.63 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128704, upper bound: 0.0137587
time: 1.97 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0043706, -0.0032991, -0.0043351, -0.0033842, -0.0009292, 0.0009930
1: -0.0019849, 0.0039482, -0.0015138, 0.0037514, -0.0054980, 0.0051452
2: 0.0061455, 0.0194005, 0.0065850, 0.0183481, -0.0114949, 0.0122832
3: -0.0008411, 0.0047446, -0.0003976, 0.0045594, -0.0051762, 0.0048440
4: 0.9934872, 1.0151576, 0.9952078, 1.0144390, -0.0200816, 0.0187927
5: 0.0009744, 0.0051901, 0.0013091, 0.0050504, -0.0039067, 0.0036559
6: -0.0124972, -0.0070110, -0.0123153, -0.0074466, -0.0047577, 0.0050840
7: -0.0103975, -0.0096977, -0.0103743, -0.0097532, -0.0006069, 0.0006485
8: -0.0060101, -0.0022196, -0.0057091, -0.0023452, -0.0035126, 0.0032871
9: -0.0070593, 0.0119169, -0.0064301, 0.0104102, -0.0164563, 0.0175850

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128705, upper bound: 0.0139248
time: 1.53 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128705, upper bound: 0.0139248
time: 1.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0043570, -0.0033173, -0.0043764, -0.0033020, -0.0009730, 0.0009687
1: -0.0018839, 0.0038728, -0.0019688, 0.0039799, -0.0053636, 0.0053877
2: 0.0063139, 0.0191750, 0.0060746, 0.0193647, -0.0120367, 0.0119828
3: -0.0007460, 0.0046736, -0.0008260, 0.0047745, -0.0050496, 0.0050723
4: 0.9938560, 1.0148822, 0.9935458, 1.0152735, -0.0195905, 0.0196786
5: 0.0010461, 0.0051366, 0.0009858, 0.0052127, -0.0038111, 0.0038283
6: -0.0124275, -0.0071044, -0.0125265, -0.0070258, -0.0049819, 0.0049596
7: -0.0103886, -0.0097096, -0.0104012, -0.0096996, -0.0006355, 0.0006326
8: -0.0059456, -0.0022677, -0.0059998, -0.0021993, -0.0034267, 0.0034421
9: -0.0068182, 0.0115940, -0.0071608, 0.0118656, -0.0172321, 0.0171549

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0137587
time: 1.84 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0137587
time: 1.54 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0043706, -0.0032991, -0.0043778, -0.0033017, -0.0009647, 0.0009952
1: -0.0019849, 0.0039482, -0.0019704, 0.0039879, -0.0055105, 0.0053414
2: 0.0061455, 0.0194005, 0.0060568, 0.0193682, -0.0119333, 0.0123111
3: -0.0008411, 0.0047446, -0.0008274, 0.0047820, -0.0051879, 0.0050287
4: 0.9934872, 1.0151576, 0.9935401, 1.0153028, -0.0201271, 0.0195095
5: 0.0009744, 0.0051901, 0.0009847, 0.0052184, -0.0039155, 0.0037954
6: -0.0124972, -0.0070110, -0.0125339, -0.0070244, -0.0049391, 0.0050955
7: -0.0103975, -0.0096977, -0.0104022, -0.0096994, -0.0006300, 0.0006500
8: -0.0060101, -0.0022196, -0.0060008, -0.0021942, -0.0035206, 0.0034125
9: -0.0070593, 0.0119169, -0.0071863, 0.0118706, -0.0170840, 0.0176248

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0139248
time: 1.62 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0139248
time: 1.78 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0043374, -0.0033275, -0.0043530, -0.0033781, -0.0009363, 0.0009870
1: -0.0018278, 0.0037643, -0.0015478, 0.0038503, -0.0054648, 0.0051843
2: 0.0065563, 0.0190496, 0.0063642, 0.0184240, -0.0115822, 0.0122090
3: -0.0006932, 0.0045715, -0.0004296, 0.0046525, -0.0051449, 0.0048808
4: 0.9940610, 1.0144860, 0.9950837, 1.0148002, -0.0199602, 0.0189356
5: 0.0010860, 0.0050595, 0.0012850, 0.0051206, -0.0038830, 0.0036837
6: -0.0123272, -0.0071562, -0.0124067, -0.0074152, -0.0047938, 0.0050532
7: -0.0103758, -0.0097162, -0.0103859, -0.0097492, -0.0006115, 0.0006446
8: -0.0059097, -0.0023370, -0.0057308, -0.0022821, -0.0034914, 0.0033121
9: -0.0064712, 0.0114145, -0.0067463, 0.0105189, -0.0165814, 0.0174786

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127173, upper bound: 0.0139003
time: 1.60 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127173, upper bound: 0.0139003
time: 1.81 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0043511, -0.0033076, -0.0043546, -0.0033778, -0.0009286, 0.0010162
1: -0.0019376, 0.0038402, -0.0015494, 0.0038592, -0.0056269, 0.0051419
2: 0.0063867, 0.0192950, 0.0063443, 0.0184277, -0.0114875, 0.0125711
3: -0.0007966, 0.0046430, -0.0004311, 0.0046608, -0.0052975, 0.0048409
4: 0.9936598, 1.0147634, 0.9950776, 1.0148326, -0.0205523, 0.0187808
5: 0.0010080, 0.0051135, 0.0012838, 0.0051269, -0.0039982, 0.0036536
6: -0.0123974, -0.0070547, -0.0124149, -0.0074136, -0.0047546, 0.0052031
7: -0.0103848, -0.0097033, -0.0103870, -0.0097490, -0.0006065, 0.0006637
8: -0.0059799, -0.0022885, -0.0057319, -0.0022764, -0.0035949, 0.0032851
9: -0.0067141, 0.0117658, -0.0067747, 0.0105243, -0.0164458, 0.0179971

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127173, upper bound: 0.0140090
time: 1.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127173, upper bound: 0.0140090
time: 1.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0043374, -0.0033275, -0.0043959, -0.0032932, -0.0009580, 0.0009764
1: -0.0018278, 0.0037643, -0.0020173, 0.0040878, -0.0054061, 0.0053042
2: 0.0065563, 0.0190496, 0.0058335, 0.0194731, -0.0118502, 0.0120777
3: -0.0006932, 0.0045715, -0.0008717, 0.0048761, -0.0050896, 0.0049937
4: 0.9940610, 1.0144860, 0.9933686, 1.0156677, -0.0197456, 0.0193736
5: 0.0010860, 0.0050595, 0.0009513, 0.0052894, -0.0038413, 0.0037689
6: -0.0123272, -0.0071562, -0.0126263, -0.0069810, -0.0049047, 0.0049989
7: -0.0103758, -0.0097162, -0.0104140, -0.0096938, -0.0006256, 0.0006377
8: -0.0059097, -0.0023370, -0.0060308, -0.0021303, -0.0034538, 0.0033888
9: -0.0064712, 0.0114145, -0.0075059, 0.0120208, -0.0169650, 0.0172908

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127273, upper bound: 0.0139003
time: 1.78 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127273, upper bound: 0.0139003
time: 1.73 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0043511, -0.0033076, -0.0043973, -0.0032930, -0.0009548, 0.0010100
1: -0.0019376, 0.0038402, -0.0020190, 0.0040956, -0.0055922, 0.0052865
2: 0.0063867, 0.0192950, 0.0058162, 0.0194767, -0.0118106, 0.0124936
3: -0.0007966, 0.0046430, -0.0008732, 0.0048834, -0.0052648, 0.0049770
4: 0.9936598, 1.0147634, 0.9933627, 1.0156960, -0.0204256, 0.0193088
5: 0.0010080, 0.0051135, 0.0009502, 0.0052949, -0.0039736, 0.0037563
6: -0.0123974, -0.0070547, -0.0126335, -0.0069795, -0.0048883, 0.0051710
7: -0.0103848, -0.0097033, -0.0104149, -0.0096937, -0.0006236, 0.0006596
8: -0.0059799, -0.0022885, -0.0060318, -0.0021254, -0.0035728, 0.0033774
9: -0.0067141, 0.0117658, -0.0075307, 0.0120260, -0.0169083, 0.0178861

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127273, upper bound: 0.0140090
time: 1.51 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0127273, upper bound: 0.0140090
time: 1.68 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0043570, -0.0033173, -0.0043530, -0.0033781, -0.0009465, 0.0009801
1: -0.0018839, 0.0038728, -0.0015478, 0.0038503, -0.0054266, 0.0052406
2: 0.0063139, 0.0191750, 0.0063642, 0.0184240, -0.0117082, 0.0121235
3: -0.0007460, 0.0046736, -0.0004296, 0.0046525, -0.0051089, 0.0049338
4: 0.9938560, 1.0148822, 0.9950837, 1.0148002, -0.0198205, 0.0191414
5: 0.0010461, 0.0051366, 0.0012850, 0.0051206, -0.0038559, 0.0037238
6: -0.0124275, -0.0071044, -0.0124067, -0.0074152, -0.0048459, 0.0050179
7: -0.0103886, -0.0097096, -0.0103859, -0.0097492, -0.0006181, 0.0006401
8: -0.0059456, -0.0022677, -0.0057308, -0.0022821, -0.0034669, 0.0033481
9: -0.0068182, 0.0115940, -0.0067463, 0.0105189, -0.0167617, 0.0173564

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128705, upper bound: 0.0137460
time: 1.48 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128705, upper bound: 0.0137460
time: 1.68 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0043706, -0.0032991, -0.0043546, -0.0033778, -0.0009346, 0.0010120
1: -0.0019849, 0.0039482, -0.0015494, 0.0038592, -0.0056031, 0.0051750
2: 0.0061455, 0.0194005, 0.0063443, 0.0184277, -0.0115616, 0.0125180
3: -0.0008411, 0.0047446, -0.0004311, 0.0046608, -0.0052751, 0.0048721
4: 0.9934872, 1.0151576, 0.9950776, 1.0148326, -0.0204655, 0.0189019
5: 0.0009744, 0.0051901, 0.0012838, 0.0051269, -0.0039814, 0.0036772
6: -0.0124972, -0.0070110, -0.0124149, -0.0074136, -0.0047853, 0.0051811
7: -0.0103975, -0.0096977, -0.0103870, -0.0097490, -0.0006104, 0.0006609
8: -0.0060101, -0.0022196, -0.0057319, -0.0022764, -0.0035797, 0.0033062
9: -0.0070593, 0.0119169, -0.0067747, 0.0105243, -0.0165519, 0.0179211

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128705, upper bound: 0.0138907
time: 1.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128705, upper bound: 0.0138907
time: 1.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0043570, -0.0033173, -0.0043959, -0.0032932, -0.0009727, 0.0009744
1: -0.0018839, 0.0038728, -0.0020173, 0.0040878, -0.0053953, 0.0053858
2: 0.0063139, 0.0191750, 0.0058335, 0.0194731, -0.0120325, 0.0120538
3: -0.0007460, 0.0046736, -0.0008717, 0.0048761, -0.0050795, 0.0050705
4: 0.9938560, 1.0148822, 0.9933686, 1.0156677, -0.0197065, 0.0196717
5: 0.0010461, 0.0051366, 0.0009513, 0.0052894, -0.0038337, 0.0038269
6: -0.0124275, -0.0071044, -0.0126263, -0.0069810, -0.0049802, 0.0049890
7: -0.0103886, -0.0097096, -0.0104140, -0.0096938, -0.0006353, 0.0006364
8: -0.0059456, -0.0022677, -0.0060308, -0.0021303, -0.0034470, 0.0034409
9: -0.0068182, 0.0115940, -0.0075059, 0.0120208, -0.0172260, 0.0172565

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0137460
time: 1.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0137460
time: 1.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0043706, -0.0032991, -0.0043973, -0.0032930, -0.0009636, 0.0010106
1: -0.0019849, 0.0039482, -0.0020190, 0.0040956, -0.0055956, 0.0053352
2: 0.0061455, 0.0194005, 0.0058162, 0.0194767, -0.0119195, 0.0125011
3: -0.0008411, 0.0047446, -0.0008732, 0.0048834, -0.0052680, 0.0050229
4: 0.9934872, 1.0151576, 0.9933627, 1.0156960, -0.0204378, 0.0194869
5: 0.0009744, 0.0051901, 0.0009502, 0.0052949, -0.0039760, 0.0037910
6: -0.0124972, -0.0070110, -0.0126335, -0.0069795, -0.0049334, 0.0051741
7: -0.0103975, -0.0096977, -0.0104149, -0.0096937, -0.0006293, 0.0006600
8: -0.0060101, -0.0022196, -0.0060318, -0.0021254, -0.0035749, 0.0034086
9: -0.0070593, 0.0119169, -0.0075307, 0.0120260, -0.0170642, 0.0178969

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0138907
time: 1.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0138907
time: 1.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.48 seconds
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0137652, upper bound: 0.0127283
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0137652, upper bound: 0.0127283
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0137652, upper bound: 0.0129905
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0137652, upper bound: 0.0129905
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0137587, upper bound: 0.0128704
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0137587, upper bound: 0.0128705
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0137587, upper bound: 0.0130772
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0137587, upper bound: 0.0130772
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0127173
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0127173
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0129819
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0129819
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0126860
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0126860
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0129264
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0139003, upper bound: 0.0129264
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127283, upper bound: 0.0137652
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127283, upper bound: 0.0137652
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127283, upper bound: 0.0139278
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127283, upper bound: 0.0139278
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127383, upper bound: 0.0137652
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127383, upper bound: 0.0137652
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127383, upper bound: 0.0139278
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127383, upper bound: 0.0139278
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128704, upper bound: 0.0137587
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128704, upper bound: 0.0137587
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128705, upper bound: 0.0139248
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128705, upper bound: 0.0139248
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0137587
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0137587
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0139248
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0139248
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127173, upper bound: 0.0139003
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127173, upper bound: 0.0139003
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127173, upper bound: 0.0140090
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127173, upper bound: 0.0140090
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127273, upper bound: 0.0139003
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127273, upper bound: 0.0139003
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127273, upper bound: 0.0140090
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0127273, upper bound: 0.0140090
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128705, upper bound: 0.0137460
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128705, upper bound: 0.0137460
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128705, upper bound: 0.0138907
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128705, upper bound: 0.0138907
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0137460
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0137460
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0138907
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 4, lower bound: -0.0128755, upper bound: 0.0138907

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0042948, -0.0034131, -0.0043374, -0.0033275, -0.0008944, 0.0008651
1: -0.0013538, 0.0035283, -0.0018278, 0.0037643, -0.0047901, 0.0049522
2: 0.0070836, 0.0179907, 0.0065563, 0.0190496, -0.0110638, 0.0107017
3: -0.0002470, 0.0043493, -0.0006932, 0.0045715, -0.0045097, 0.0046623
4: 0.9957922, 1.0136240, 0.9940610, 1.0144860, -0.0174960, 0.0180879
5: 0.0014228, 0.0048918, 0.0010860, 0.0050595, -0.0034037, 0.0035188
6: -0.0121089, -0.0075945, -0.0123272, -0.0071562, -0.0045792, 0.0044294
7: -0.0103480, -0.0097721, -0.0103758, -0.0097162, -0.0005841, 0.0005650
8: -0.0056069, -0.0024878, -0.0059097, -0.0023370, -0.0030603, 0.0031639
9: -0.0057163, 0.0098986, -0.0064712, 0.0114145, -0.0158392, 0.0153208

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0130552, upper bound: 0.0120861
time: 1.41 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0130139, upper bound: 0.0120313
time: 1.53 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042948, -0.0034131, -0.0043511, -0.0033076, -0.0009210, 0.0008838
1: -0.0013538, 0.0035283, -0.0019376, 0.0038402, -0.0048933, 0.0050998
2: 0.0070836, 0.0179907, 0.0063867, 0.0192950, -0.0113934, 0.0109322
3: -0.0002470, 0.0043493, -0.0007966, 0.0046430, -0.0046069, 0.0048012
4: 0.9957922, 1.0136240, 0.9936598, 1.0147634, -0.0178729, 0.0186269
5: 0.0014228, 0.0048918, 0.0010080, 0.0051135, -0.0034770, 0.0036237
6: -0.0121089, -0.0075945, -0.0123974, -0.0070547, -0.0047157, 0.0045248
7: -0.0103480, -0.0097721, -0.0103848, -0.0097033, -0.0006015, 0.0005772
8: -0.0056069, -0.0024878, -0.0059799, -0.0022885, -0.0031263, 0.0032581
9: -0.0057163, 0.0098986, -0.0067141, 0.0117658, -0.0163111, 0.0156509

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0130552, upper bound: 0.0120861
time: 1.63 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0130139, upper bound: 0.0120313
time: 1.75 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043104, -0.0033899, -0.0043374, -0.0033275, -0.0009224, 0.0009049
1: -0.0014823, 0.0036144, -0.0018278, 0.0037643, -0.0050105, 0.0051074
2: 0.0068911, 0.0182778, 0.0065563, 0.0190496, -0.0114105, 0.0111940
3: -0.0003680, 0.0044304, -0.0006932, 0.0045715, -0.0047172, 0.0048084
4: 0.9953227, 1.0139387, 0.9940610, 1.0144860, -0.0183008, 0.0186548
5: 0.0013315, 0.0049530, 0.0010860, 0.0050595, -0.0035602, 0.0036291
6: -0.0121886, -0.0074757, -0.0123272, -0.0071562, -0.0047228, 0.0046331
7: -0.0103581, -0.0097570, -0.0103758, -0.0097162, -0.0006024, 0.0005910
8: -0.0056890, -0.0024328, -0.0059097, -0.0023370, -0.0032011, 0.0032630
9: -0.0059918, 0.0103096, -0.0064712, 0.0114145, -0.0163356, 0.0160255

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0130268, upper bound: 0.0123079
time: 1.49 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0129886, upper bound: 0.0122261
time: 1.45 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043104, -0.0033899, -0.0043511, -0.0033076, -0.0009160, 0.0008914
1: -0.0014823, 0.0036144, -0.0019376, 0.0038402, -0.0049356, 0.0050720
2: 0.0068911, 0.0182778, 0.0063867, 0.0192950, -0.0113315, 0.0110266
3: -0.0003680, 0.0044304, -0.0007966, 0.0046430, -0.0046466, 0.0047751
4: 0.9953227, 1.0139387, 0.9936598, 1.0147634, -0.0180271, 0.0185256
5: 0.0013315, 0.0049530, 0.0010080, 0.0051135, -0.0035070, 0.0036040
6: -0.0121886, -0.0074757, -0.0123974, -0.0070547, -0.0046900, 0.0045638
7: -0.0103581, -0.0097570, -0.0103848, -0.0097033, -0.0005983, 0.0005822
8: -0.0056890, -0.0024328, -0.0059799, -0.0022885, -0.0031532, 0.0032404
9: -0.0059918, 0.0103096, -0.0067141, 0.0117658, -0.0162224, 0.0157859

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0130268, upper bound: 0.0123079
time: 1.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0129886, upper bound: 0.0122261
time: 1.68 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0042948, -0.0034131, -0.0043570, -0.0033173, -0.0009209, 0.0008982
1: -0.0013538, 0.0035283, -0.0018839, 0.0038728, -0.0049735, 0.0050988
2: 0.0070836, 0.0179907, 0.0063139, 0.0191750, -0.0113913, 0.0111113
3: -0.0002470, 0.0043493, -0.0007460, 0.0046736, -0.0046823, 0.0048003
4: 0.9957922, 1.0136240, 0.9938560, 1.0148822, -0.0181656, 0.0186234
5: 0.0014228, 0.0048918, 0.0010461, 0.0051366, -0.0035339, 0.0036230
6: -0.0121089, -0.0075945, -0.0124275, -0.0071044, -0.0047148, 0.0045989
7: -0.0103480, -0.0097721, -0.0103886, -0.0097096, -0.0006014, 0.0005866
8: -0.0056069, -0.0024878, -0.0059456, -0.0022677, -0.0031775, 0.0032575
9: -0.0057163, 0.0098986, -0.0068182, 0.0115940, -0.0163081, 0.0159072

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0130448, upper bound: 0.0122649
time: 1.45 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0130081, upper bound: 0.0122108
time: 1.43 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042948, -0.0034131, -0.0043706, -0.0032991, -0.0009419, 0.0009145
1: -0.0013538, 0.0035283, -0.0019849, 0.0039482, -0.0050635, 0.0052154
2: 0.0070836, 0.0179907, 0.0061455, 0.0194005, -0.0116519, 0.0113125
3: -0.0002470, 0.0043493, -0.0008411, 0.0047446, -0.0047671, 0.0049101
4: 0.9957922, 1.0136240, 0.9934872, 1.0151576, -0.0184946, 0.0190494
5: 0.0014228, 0.0048918, 0.0009744, 0.0051901, -0.0035979, 0.0037059
6: -0.0121089, -0.0075945, -0.0124972, -0.0070110, -0.0048226, 0.0046822
7: -0.0103480, -0.0097721, -0.0103975, -0.0096977, -0.0006152, 0.0005973
8: -0.0056069, -0.0024878, -0.0060101, -0.0022196, -0.0032350, 0.0033321
9: -0.0057163, 0.0098986, -0.0070593, 0.0119169, -0.0166811, 0.0161952

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0130448, upper bound: 0.0122649
time: 1.60 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0130081, upper bound: 0.0122108
time: 1.53 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043104, -0.0033899, -0.0043570, -0.0033173, -0.0009489, 0.0009377
1: -0.0014823, 0.0036144, -0.0018839, 0.0038728, -0.0051920, 0.0052540
2: 0.0068911, 0.0182778, 0.0063139, 0.0191750, -0.0117381, 0.0115995
3: -0.0003680, 0.0044304, -0.0007460, 0.0046736, -0.0048881, 0.0049464
4: 0.9953227, 1.0139387, 0.9938560, 1.0148822, -0.0189639, 0.0191903
5: 0.0013315, 0.0049530, 0.0010461, 0.0051366, -0.0036892, 0.0037333
6: -0.0121886, -0.0074757, -0.0124275, -0.0071044, -0.0048583, 0.0048010
7: -0.0103581, -0.0097570, -0.0103886, -0.0097096, -0.0006197, 0.0006124
8: -0.0056890, -0.0024328, -0.0059456, -0.0022677, -0.0033171, 0.0033567
9: -0.0059918, 0.0103096, -0.0068182, 0.0115940, -0.0168045, 0.0166062

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0130197, upper bound: 0.0124628
time: 1.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0129837, upper bound: 0.0123720
time: 1.53 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043104, -0.0033899, -0.0043706, -0.0032991, -0.0009407, 0.0009244
1: -0.0014823, 0.0036144, -0.0019849, 0.0039482, -0.0051182, 0.0052084
2: 0.0068911, 0.0182778, 0.0061455, 0.0194005, -0.0116360, 0.0114347
3: -0.0003680, 0.0044304, -0.0008411, 0.0047446, -0.0048186, 0.0049035
4: 0.9953227, 1.0139387, 0.9934872, 1.0151576, -0.0186944, 0.0190235
5: 0.0013315, 0.0049530, 0.0009744, 0.0051901, -0.0036368, 0.0037008
6: -0.0121886, -0.0074757, -0.0124972, -0.0070110, -0.0048161, 0.0047328
7: -0.0103581, -0.0097570, -0.0103975, -0.0096977, -0.0006143, 0.0006037
8: -0.0056890, -0.0024328, -0.0060101, -0.0022196, -0.0032700, 0.0033275
9: -0.0059918, 0.0103096, -0.0070593, 0.0119169, -0.0166584, 0.0163702

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0130197, upper bound: 0.0124628
time: 1.49 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0129837, upper bound: 0.0123720
time: 1.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043140, -0.0034045, -0.0043374, -0.0033275, -0.0009456, 0.0009008
1: -0.0014012, 0.0036348, -0.0018278, 0.0037643, -0.0049877, 0.0052358
2: 0.0068456, 0.0180967, 0.0065563, 0.0190496, -0.0116974, 0.0111430
3: -0.0002916, 0.0044496, -0.0006932, 0.0045715, -0.0046957, 0.0049293
4: 0.9956189, 1.0140129, 0.9940610, 1.0144860, -0.0182174, 0.0191239
5: 0.0013891, 0.0049675, 0.0010860, 0.0050595, -0.0035440, 0.0037204
6: -0.0122074, -0.0075507, -0.0123272, -0.0071562, -0.0048415, 0.0046120
7: -0.0103605, -0.0097665, -0.0103758, -0.0097162, -0.0006176, 0.0005883
8: -0.0056372, -0.0024198, -0.0059097, -0.0023370, -0.0031865, 0.0033451
9: -0.0060570, 0.0100503, -0.0064712, 0.0114145, -0.0167463, 0.0159526

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131616, upper bound: 0.0120821
time: 1.41 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131495, upper bound: 0.0120313
time: 1.41 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0043140, -0.0034045, -0.0043511, -0.0033076, -0.0009730, 0.0009194
1: -0.0014012, 0.0036348, -0.0019376, 0.0038402, -0.0050909, 0.0053876
2: 0.0068456, 0.0180967, 0.0063867, 0.0192950, -0.0120364, 0.0113735
3: -0.0002916, 0.0044496, -0.0007966, 0.0046430, -0.0047928, 0.0050722
4: 0.9956189, 1.0140129, 0.9936598, 1.0147634, -0.0185944, 0.0196781
5: 0.0013891, 0.0049675, 0.0010080, 0.0051135, -0.0036173, 0.0038282
6: -0.0122074, -0.0075507, -0.0123974, -0.0070547, -0.0049818, 0.0047074
7: -0.0103605, -0.0097665, -0.0103848, -0.0097033, -0.0006355, 0.0006005
8: -0.0056372, -0.0024198, -0.0059799, -0.0022885, -0.0032525, 0.0034420
9: -0.0060570, 0.0100503, -0.0067141, 0.0117658, -0.0172316, 0.0162826

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131616, upper bound: 0.0120821
time: 1.42 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131495, upper bound: 0.0120313
time: 1.62 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043295, -0.0033837, -0.0043374, -0.0033275, -0.0009676, 0.0009314
1: -0.0015163, 0.0037204, -0.0018278, 0.0037643, -0.0051570, 0.0053578
2: 0.0066544, 0.0183536, 0.0065563, 0.0190496, -0.0119698, 0.0115213
3: -0.0003999, 0.0045302, -0.0006932, 0.0045715, -0.0048551, 0.0050441
4: 0.9951988, 1.0143256, 0.9940610, 1.0144860, -0.0188359, 0.0195692
5: 0.0013074, 0.0050283, 0.0010860, 0.0050595, -0.0036643, 0.0038070
6: -0.0122866, -0.0074443, -0.0123272, -0.0071562, -0.0049542, 0.0047686
7: -0.0103706, -0.0097530, -0.0103758, -0.0097162, -0.0006320, 0.0006083
8: -0.0057107, -0.0023651, -0.0059097, -0.0023370, -0.0032947, 0.0034230
9: -0.0063307, 0.0104181, -0.0064712, 0.0114145, -0.0171363, 0.0164942

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131605, upper bound: 0.0123077
time: 1.36 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131452, upper bound: 0.0122261
time: 1.33 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043295, -0.0033837, -0.0043511, -0.0033076, -0.0009684, 0.0009232
1: -0.0015163, 0.0037204, -0.0019376, 0.0038402, -0.0051120, 0.0053620
2: 0.0066544, 0.0183536, 0.0063867, 0.0192950, -0.0119793, 0.0114207
3: -0.0003999, 0.0045302, -0.0007966, 0.0046430, -0.0048127, 0.0050481
4: 0.9951988, 1.0143256, 0.9936598, 1.0147634, -0.0186715, 0.0195847
5: 0.0013074, 0.0050283, 0.0010080, 0.0051135, -0.0036324, 0.0038100
6: -0.0122866, -0.0074443, -0.0123974, -0.0070547, -0.0049582, 0.0047270
7: -0.0103706, -0.0097530, -0.0103848, -0.0097033, -0.0006325, 0.0006030
8: -0.0057107, -0.0023651, -0.0059799, -0.0022885, -0.0032660, 0.0034257
9: -0.0063307, 0.0104181, -0.0067141, 0.0117658, -0.0171499, 0.0163502

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131605, upper bound: 0.0123077
time: 1.59 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131452, upper bound: 0.0122261
time: 1.42 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043140, -0.0034045, -0.0043570, -0.0033173, -0.0009373, 0.0009012
1: -0.0014012, 0.0036348, -0.0018839, 0.0038728, -0.0049897, 0.0051900
2: 0.0068456, 0.0180967, 0.0063139, 0.0191750, -0.0115950, 0.0111475
3: -0.0002916, 0.0044496, -0.0007460, 0.0046736, -0.0046976, 0.0048862
4: 0.9956189, 1.0140129, 0.9938560, 1.0148822, -0.0182249, 0.0189564
5: 0.0013891, 0.0049675, 0.0010461, 0.0051366, -0.0035455, 0.0036878
6: -0.0122074, -0.0075507, -0.0124275, -0.0071044, -0.0047991, 0.0046139
7: -0.0103605, -0.0097665, -0.0103886, -0.0097096, -0.0006122, 0.0005885
8: -0.0056372, -0.0024198, -0.0059456, -0.0022677, -0.0031878, 0.0033158
9: -0.0060570, 0.0100503, -0.0068182, 0.0115940, -0.0165997, 0.0159591

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131616, upper bound: 0.0120607
time: 1.31 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131495, upper bound: 0.0120147
time: 1.40 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0043140, -0.0034045, -0.0043706, -0.0032991, -0.0009670, 0.0009207
1: -0.0014012, 0.0036348, -0.0019849, 0.0039482, -0.0050977, 0.0053542
2: 0.0068456, 0.0180967, 0.0061455, 0.0194005, -0.0119619, 0.0113888
3: -0.0002916, 0.0044496, -0.0008411, 0.0047446, -0.0047993, 0.0050408
4: 0.9956189, 1.0140129, 0.9934872, 1.0151576, -0.0186194, 0.0195562
5: 0.0013891, 0.0049675, 0.0009744, 0.0051901, -0.0036222, 0.0038045
6: -0.0122074, -0.0075507, -0.0124972, -0.0070110, -0.0049510, 0.0047138
7: -0.0103605, -0.0097665, -0.0103975, -0.0096977, -0.0006315, 0.0006013
8: -0.0056372, -0.0024198, -0.0060101, -0.0022196, -0.0032568, 0.0034207
9: -0.0060570, 0.0100503, -0.0070593, 0.0119169, -0.0171249, 0.0163045

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131616, upper bound: 0.0120607
time: 1.62 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131495, upper bound: 0.0120147
time: 1.51 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043295, -0.0033837, -0.0043570, -0.0033173, -0.0009642, 0.0009418
1: -0.0015163, 0.0037204, -0.0018839, 0.0038728, -0.0052146, 0.0053388
2: 0.0066544, 0.0183536, 0.0063139, 0.0191750, -0.0119275, 0.0116499
3: -0.0003999, 0.0045302, -0.0007460, 0.0046736, -0.0049093, 0.0050263
4: 0.9951988, 1.0143256, 0.9938560, 1.0148822, -0.0190462, 0.0195001
5: 0.0013074, 0.0050283, 0.0010461, 0.0051366, -0.0037052, 0.0037935
6: -0.0122866, -0.0074443, -0.0124275, -0.0071044, -0.0049367, 0.0048218
7: -0.0103706, -0.0097530, -0.0103886, -0.0097096, -0.0006297, 0.0006151
8: -0.0057107, -0.0023651, -0.0059456, -0.0022677, -0.0033315, 0.0034109
9: -0.0063307, 0.0104181, -0.0068182, 0.0115940, -0.0170757, 0.0166783

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131605, upper bound: 0.0122838
time: 1.44 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131452, upper bound: 0.0122037
time: 1.37 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043295, -0.0033837, -0.0043706, -0.0032991, -0.0009625, 0.0009295
1: -0.0015163, 0.0037204, -0.0019849, 0.0039482, -0.0051467, 0.0053295
2: 0.0066544, 0.0183536, 0.0061455, 0.0194005, -0.0119068, 0.0114983
3: -0.0003999, 0.0045302, -0.0008411, 0.0047446, -0.0048454, 0.0050175
4: 0.9951988, 1.0143256, 0.9934872, 1.0151576, -0.0187983, 0.0194661
5: 0.0013074, 0.0050283, 0.0009744, 0.0051901, -0.0036570, 0.0037869
6: -0.0122866, -0.0074443, -0.0124972, -0.0070110, -0.0049281, 0.0047591
7: -0.0103706, -0.0097530, -0.0103975, -0.0096977, -0.0006286, 0.0006071
8: -0.0057107, -0.0023651, -0.0060101, -0.0022196, -0.0032881, 0.0034049
9: -0.0063307, 0.0104181, -0.0070593, 0.0119169, -0.0170460, 0.0164612

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131605, upper bound: 0.0122838
time: 1.51 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131452, upper bound: 0.0122037
time: 1.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043374, -0.0033275, -0.0042948, -0.0034131, -0.0008651, 0.0008944
1: -0.0018278, 0.0037643, -0.0013538, 0.0035283, -0.0049522, 0.0047901
2: 0.0065563, 0.0190496, 0.0070836, 0.0179907, -0.0107017, 0.0110638
3: -0.0006932, 0.0045715, -0.0002470, 0.0043493, -0.0046623, 0.0045097
4: 0.9940610, 1.0144860, 0.9957922, 1.0136240, -0.0180879, 0.0174960
5: 0.0010860, 0.0050595, 0.0014228, 0.0048918, -0.0035188, 0.0034037
6: -0.0123272, -0.0071562, -0.0121089, -0.0075945, -0.0044294, 0.0045792
7: -0.0103758, -0.0097162, -0.0103480, -0.0097721, -0.0005650, 0.0005841
8: -0.0059097, -0.0023370, -0.0056069, -0.0024878, -0.0031639, 0.0030603
9: -0.0064712, 0.0114145, -0.0057163, 0.0098986, -0.0153208, 0.0158392

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119636, upper bound: 0.0130385
time: 1.39 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120663, upper bound: 0.0129886
time: 1.30 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0043374, -0.0033275, -0.0043104, -0.0033899, -0.0009049, 0.0009224
1: -0.0018278, 0.0037643, -0.0014823, 0.0036144, -0.0051074, 0.0050105
2: 0.0065563, 0.0190496, 0.0068911, 0.0182778, -0.0111940, 0.0114105
3: -0.0006932, 0.0045715, -0.0003680, 0.0044304, -0.0048084, 0.0047172
4: 0.9940610, 1.0144860, 0.9953227, 1.0139387, -0.0186548, 0.0183008
5: 0.0010860, 0.0050595, 0.0013315, 0.0049530, -0.0036291, 0.0035602
6: -0.0123272, -0.0071562, -0.0121886, -0.0074757, -0.0046331, 0.0047228
7: -0.0103758, -0.0097162, -0.0103581, -0.0097570, -0.0005910, 0.0006024
8: -0.0059097, -0.0023370, -0.0056890, -0.0024328, -0.0032630, 0.0032011
9: -0.0064712, 0.0114145, -0.0059918, 0.0103096, -0.0160255, 0.0163356

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119636, upper bound: 0.0130385
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120663, upper bound: 0.0129886
time: 1.44 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043511, -0.0033076, -0.0042948, -0.0034131, -0.0008838, 0.0009210
1: -0.0019376, 0.0038402, -0.0013538, 0.0035283, -0.0050998, 0.0048933
2: 0.0063867, 0.0192950, 0.0070836, 0.0179907, -0.0109322, 0.0113934
3: -0.0007966, 0.0046430, -0.0002470, 0.0043493, -0.0048012, 0.0046069
4: 0.9936598, 1.0147634, 0.9957922, 1.0136240, -0.0186269, 0.0178729
5: 0.0010080, 0.0051135, 0.0014228, 0.0048918, -0.0036237, 0.0034770
6: -0.0123974, -0.0070547, -0.0121089, -0.0075945, -0.0045248, 0.0047157
7: -0.0103848, -0.0097033, -0.0103480, -0.0097721, -0.0005772, 0.0006015
8: -0.0059799, -0.0022885, -0.0056069, -0.0024878, -0.0032581, 0.0031263
9: -0.0067141, 0.0117658, -0.0057163, 0.0098986, -0.0156509, 0.0163111

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119304, upper bound: 0.0131914
time: 1.29 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120313, upper bound: 0.0131422
time: 1.40 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043511, -0.0033076, -0.0043104, -0.0033899, -0.0008914, 0.0009160
1: -0.0019376, 0.0038402, -0.0014823, 0.0036144, -0.0050720, 0.0049356
2: 0.0063867, 0.0192950, 0.0068911, 0.0182778, -0.0110266, 0.0113315
3: -0.0007966, 0.0046430, -0.0003680, 0.0044304, -0.0047751, 0.0046466
4: 0.9936598, 1.0147634, 0.9953227, 1.0139387, -0.0185256, 0.0180271
5: 0.0010080, 0.0051135, 0.0013315, 0.0049530, -0.0036040, 0.0035070
6: -0.0123974, -0.0070547, -0.0121886, -0.0074757, -0.0045638, 0.0046900
7: -0.0103848, -0.0097033, -0.0103581, -0.0097570, -0.0005822, 0.0005983
8: -0.0059799, -0.0022885, -0.0056890, -0.0024328, -0.0032404, 0.0031532
9: -0.0067141, 0.0117658, -0.0059918, 0.0103096, -0.0157859, 0.0162224

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119304, upper bound: 0.0131914
time: 1.62 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120313, upper bound: 0.0131422
time: 1.51 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043374, -0.0033275, -0.0043374, -0.0033275, -0.0008984, 0.0008984
1: -0.0018278, 0.0037643, -0.0018278, 0.0037643, -0.0049742, 0.0049742
2: 0.0065563, 0.0190496, 0.0065563, 0.0190496, -0.0111128, 0.0111128
3: -0.0006932, 0.0045715, -0.0006932, 0.0045715, -0.0046830, 0.0046830
4: 0.9940610, 1.0144860, 0.9940610, 1.0144860, -0.0181682, 0.0181682
5: 0.0010860, 0.0050595, 0.0010860, 0.0050595, -0.0035344, 0.0035344
6: -0.0123272, -0.0071562, -0.0123272, -0.0071562, -0.0045995, 0.0045995
7: -0.0103758, -0.0097162, -0.0103758, -0.0097162, -0.0005867, 0.0005867
8: -0.0059097, -0.0023370, -0.0059097, -0.0023370, -0.0031779, 0.0031779
9: -0.0064712, 0.0114145, -0.0064712, 0.0114145, -0.0159094, 0.0159094

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119736, upper bound: 0.0130385
time: 1.31 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120734, upper bound: 0.0129886
time: 1.29 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0043374, -0.0033275, -0.0043511, -0.0033076, -0.0009304, 0.0009192
1: -0.0018278, 0.0037643, -0.0019376, 0.0038402, -0.0050894, 0.0051516
2: 0.0065563, 0.0190496, 0.0063867, 0.0192950, -0.0115093, 0.0113703
3: -0.0006932, 0.0045715, -0.0007966, 0.0046430, -0.0047915, 0.0048500
4: 0.9940610, 1.0144860, 0.9936598, 1.0147634, -0.0185891, 0.0188163
5: 0.0010860, 0.0050595, 0.0010080, 0.0051135, -0.0036163, 0.0036605
6: -0.0123272, -0.0071562, -0.0123974, -0.0070547, -0.0047636, 0.0047061
7: -0.0103758, -0.0097162, -0.0103848, -0.0097033, -0.0006076, 0.0006003
8: -0.0059097, -0.0023370, -0.0059799, -0.0022885, -0.0032515, 0.0032913
9: -0.0064712, 0.0114145, -0.0067141, 0.0117658, -0.0164770, 0.0162780

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119736, upper bound: 0.0130385
time: 1.48 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120734, upper bound: 0.0129886
time: 1.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043511, -0.0033076, -0.0043374, -0.0033275, -0.0009192, 0.0009304
1: -0.0019376, 0.0038402, -0.0018278, 0.0037643, -0.0051516, 0.0050894
2: 0.0063867, 0.0192950, 0.0065563, 0.0190496, -0.0113703, 0.0115093
3: -0.0007966, 0.0046430, -0.0006932, 0.0045715, -0.0048500, 0.0047915
4: 0.9936598, 1.0147634, 0.9940610, 1.0144860, -0.0188163, 0.0185891
5: 0.0010080, 0.0051135, 0.0010860, 0.0050595, -0.0036605, 0.0036163
6: -0.0123974, -0.0070547, -0.0123272, -0.0071562, -0.0047061, 0.0047636
7: -0.0103848, -0.0097033, -0.0103758, -0.0097162, -0.0006003, 0.0006076
8: -0.0059799, -0.0022885, -0.0059097, -0.0023370, -0.0032913, 0.0032515
9: -0.0067141, 0.0117658, -0.0064712, 0.0114145, -0.0162780, 0.0164770

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119408, upper bound: 0.0131914
time: 1.27 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120384, upper bound: 0.0131422
time: 1.35 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043511, -0.0033076, -0.0043511, -0.0033076, -0.0009216, 0.0009216
1: -0.0019376, 0.0038402, -0.0019376, 0.0038402, -0.0051026, 0.0051026
2: 0.0063867, 0.0192950, 0.0063867, 0.0192950, -0.0113998, 0.0113998
3: -0.0007966, 0.0046430, -0.0007966, 0.0046430, -0.0048039, 0.0048039
4: 0.9936598, 1.0147634, 0.9936598, 1.0147634, -0.0186373, 0.0186373
5: 0.0010080, 0.0051135, 0.0010080, 0.0051135, -0.0036257, 0.0036257
6: -0.0123974, -0.0070547, -0.0123974, -0.0070547, -0.0047183, 0.0047183
7: -0.0103848, -0.0097033, -0.0103848, -0.0097033, -0.0006019, 0.0006019
8: -0.0059799, -0.0022885, -0.0059799, -0.0022885, -0.0032600, 0.0032600
9: -0.0067141, 0.0117658, -0.0067141, 0.0117658, -0.0163203, 0.0163203

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119408, upper bound: 0.0131914
time: 1.44 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120384, upper bound: 0.0131422
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043570, -0.0033173, -0.0042948, -0.0034131, -0.0008982, 0.0009209
1: -0.0018839, 0.0038728, -0.0013538, 0.0035283, -0.0050988, 0.0049735
2: 0.0063139, 0.0191750, 0.0070836, 0.0179907, -0.0111113, 0.0113913
3: -0.0007460, 0.0046736, -0.0002470, 0.0043493, -0.0048003, 0.0046823
4: 0.9938560, 1.0148822, 0.9957922, 1.0136240, -0.0186234, 0.0181656
5: 0.0010461, 0.0051366, 0.0014228, 0.0048918, -0.0036230, 0.0035339
6: -0.0124275, -0.0071044, -0.0121089, -0.0075945, -0.0045989, 0.0047148
7: -0.0103886, -0.0097096, -0.0103480, -0.0097721, -0.0005866, 0.0006014
8: -0.0059456, -0.0022677, -0.0056069, -0.0024878, -0.0032575, 0.0031775
9: -0.0068182, 0.0115940, -0.0057163, 0.0098986, -0.0159072, 0.0163081

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120645, upper bound: 0.0130298
time: 1.28 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122323, upper bound: 0.0129837
time: 1.22 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0043570, -0.0033173, -0.0043104, -0.0033899, -0.0009377, 0.0009489
1: -0.0018839, 0.0038728, -0.0014823, 0.0036144, -0.0052540, 0.0051920
2: 0.0063139, 0.0191750, 0.0068911, 0.0182778, -0.0115995, 0.0117381
3: -0.0007460, 0.0046736, -0.0003680, 0.0044304, -0.0049464, 0.0048881
4: 0.9938560, 1.0148822, 0.9953227, 1.0139387, -0.0191903, 0.0189639
5: 0.0010461, 0.0051366, 0.0013315, 0.0049530, -0.0037333, 0.0036892
6: -0.0124275, -0.0071044, -0.0121886, -0.0074757, -0.0048010, 0.0048583
7: -0.0103886, -0.0097096, -0.0103581, -0.0097570, -0.0006124, 0.0006197
8: -0.0059456, -0.0022677, -0.0056890, -0.0024328, -0.0033567, 0.0033171
9: -0.0068182, 0.0115940, -0.0059918, 0.0103096, -0.0166062, 0.0168045

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120645, upper bound: 0.0130298
time: 1.60 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122323, upper bound: 0.0129837
time: 1.67 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043706, -0.0032991, -0.0042948, -0.0034131, -0.0009145, 0.0009419
1: -0.0019849, 0.0039482, -0.0013538, 0.0035283, -0.0052154, 0.0050635
2: 0.0061455, 0.0194005, 0.0070836, 0.0179907, -0.0113125, 0.0116519
3: -0.0008411, 0.0047446, -0.0002470, 0.0043493, -0.0049101, 0.0047671
4: 0.9934872, 1.0151576, 0.9957922, 1.0136240, -0.0190494, 0.0184946
5: 0.0009744, 0.0051901, 0.0014228, 0.0048918, -0.0037059, 0.0035979
6: -0.0124972, -0.0070110, -0.0121089, -0.0075945, -0.0046822, 0.0048226
7: -0.0103975, -0.0096977, -0.0103480, -0.0097721, -0.0005973, 0.0006152
8: -0.0060101, -0.0022196, -0.0056069, -0.0024878, -0.0033321, 0.0032350
9: -0.0070593, 0.0119169, -0.0057163, 0.0098986, -0.0161952, 0.0166811

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120557, upper bound: 0.0131911
time: 1.32 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122108, upper bound: 0.0131411
time: 1.26 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043706, -0.0032991, -0.0043104, -0.0033899, -0.0009244, 0.0009407
1: -0.0019849, 0.0039482, -0.0014823, 0.0036144, -0.0052084, 0.0051182
2: 0.0061455, 0.0194005, 0.0068911, 0.0182778, -0.0114347, 0.0116360
3: -0.0008411, 0.0047446, -0.0003680, 0.0044304, -0.0049035, 0.0048186
4: 0.9934872, 1.0151576, 0.9953227, 1.0139387, -0.0190235, 0.0186944
5: 0.0009744, 0.0051901, 0.0013315, 0.0049530, -0.0037008, 0.0036368
6: -0.0124972, -0.0070110, -0.0121886, -0.0074757, -0.0047328, 0.0048161
7: -0.0103975, -0.0096977, -0.0103581, -0.0097570, -0.0006037, 0.0006143
8: -0.0060101, -0.0022196, -0.0056890, -0.0024328, -0.0033275, 0.0032700
9: -0.0070593, 0.0119169, -0.0059918, 0.0103096, -0.0163702, 0.0166584

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120557, upper bound: 0.0131911
time: 1.48 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122108, upper bound: 0.0131411
time: 1.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043570, -0.0033173, -0.0043374, -0.0033275, -0.0009363, 0.0009282
1: -0.0018839, 0.0038728, -0.0018278, 0.0037643, -0.0051396, 0.0051845
2: 0.0063139, 0.0191750, 0.0065563, 0.0190496, -0.0115827, 0.0114824
3: -0.0007460, 0.0046736, -0.0006932, 0.0045715, -0.0048387, 0.0048810
4: 0.9938560, 1.0148822, 0.9940610, 1.0144860, -0.0187723, 0.0189364
5: 0.0010461, 0.0051366, 0.0010860, 0.0050595, -0.0036520, 0.0036839
6: -0.0124275, -0.0071044, -0.0123272, -0.0071562, -0.0047940, 0.0047525
7: -0.0103886, -0.0097096, -0.0103758, -0.0097162, -0.0006115, 0.0006062
8: -0.0059456, -0.0022677, -0.0059097, -0.0023370, -0.0032836, 0.0033123
9: -0.0068182, 0.0115940, -0.0064712, 0.0114145, -0.0165821, 0.0164384

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120734, upper bound: 0.0130298
time: 1.30 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122383, upper bound: 0.0129837
time: 1.23 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0043570, -0.0033173, -0.0043511, -0.0033076, -0.0009683, 0.0009490
1: -0.0018839, 0.0038728, -0.0019376, 0.0038402, -0.0052548, 0.0053613
2: 0.0063139, 0.0191750, 0.0063867, 0.0192950, -0.0119778, 0.0117398
3: -0.0007460, 0.0046736, -0.0007966, 0.0046430, -0.0049472, 0.0050475
4: 0.9938560, 1.0148822, 0.9936598, 1.0147634, -0.0191932, 0.0195823
5: 0.0010461, 0.0051366, 0.0010080, 0.0051135, -0.0037338, 0.0038095
6: -0.0124275, -0.0071044, -0.0123974, -0.0070547, -0.0049576, 0.0048590
7: -0.0103886, -0.0097096, -0.0103848, -0.0097033, -0.0006324, 0.0006198
8: -0.0059456, -0.0022677, -0.0059799, -0.0022885, -0.0033572, 0.0034253
9: -0.0068182, 0.0115940, -0.0067141, 0.0117658, -0.0171477, 0.0168070

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120734, upper bound: 0.0130298
time: 1.40 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122383, upper bound: 0.0129837
time: 1.59 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043706, -0.0032991, -0.0043374, -0.0033275, -0.0009543, 0.0009528
1: -0.0019849, 0.0039482, -0.0018278, 0.0037643, -0.0052755, 0.0052842
2: 0.0061455, 0.0194005, 0.0065563, 0.0190496, -0.0118054, 0.0117860
3: -0.0008411, 0.0047446, -0.0006932, 0.0045715, -0.0049667, 0.0049748
4: 0.9934872, 1.0151576, 0.9940610, 1.0144860, -0.0192687, 0.0193004
5: 0.0009744, 0.0051901, 0.0010860, 0.0050595, -0.0037485, 0.0037547
6: -0.0124972, -0.0070110, -0.0123272, -0.0071562, -0.0048862, 0.0048782
7: -0.0103975, -0.0096977, -0.0103758, -0.0097162, -0.0006233, 0.0006223
8: -0.0060101, -0.0022196, -0.0059097, -0.0023370, -0.0033704, 0.0033760
9: -0.0070593, 0.0119169, -0.0064712, 0.0114145, -0.0169009, 0.0168732

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120633, upper bound: 0.0131911
time: 1.44 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122164, upper bound: 0.0131411
time: 1.25 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043706, -0.0032991, -0.0043511, -0.0033076, -0.0009595, 0.0009492
1: -0.0019849, 0.0039482, -0.0019376, 0.0038402, -0.0052555, 0.0053127
2: 0.0061455, 0.0194005, 0.0063867, 0.0192950, -0.0118691, 0.0117413
3: -0.0008411, 0.0047446, -0.0007966, 0.0046430, -0.0049478, 0.0050017
4: 0.9934872, 1.0151576, 0.9936598, 1.0147634, -0.0191956, 0.0194046
5: 0.0009744, 0.0051901, 0.0010080, 0.0051135, -0.0037343, 0.0037750
6: -0.0124972, -0.0070110, -0.0123974, -0.0070547, -0.0049126, 0.0048596
7: -0.0103975, -0.0096977, -0.0103848, -0.0097033, -0.0006266, 0.0006199
8: -0.0060101, -0.0022196, -0.0059799, -0.0022885, -0.0033576, 0.0033942
9: -0.0070593, 0.0119169, -0.0067141, 0.0117658, -0.0169922, 0.0168091

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120633, upper bound: 0.0131911
time: 1.59 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122164, upper bound: 0.0131411
time: 1.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043374, -0.0033275, -0.0043140, -0.0034045, -0.0009008, 0.0009456
1: -0.0018278, 0.0037643, -0.0014012, 0.0036348, -0.0052358, 0.0049877
2: 0.0065563, 0.0190496, 0.0068456, 0.0180967, -0.0111430, 0.0116974
3: -0.0006932, 0.0045715, -0.0002916, 0.0044496, -0.0049293, 0.0046957
4: 0.9940610, 1.0144860, 0.9956189, 1.0140129, -0.0191239, 0.0182174
5: 0.0010860, 0.0050595, 0.0013891, 0.0049675, -0.0037204, 0.0035440
6: -0.0123272, -0.0071562, -0.0122074, -0.0075507, -0.0046120, 0.0048415
7: -0.0103758, -0.0097162, -0.0103605, -0.0097665, -0.0005883, 0.0006176
8: -0.0059097, -0.0023370, -0.0056372, -0.0024198, -0.0033451, 0.0031865
9: -0.0064712, 0.0114145, -0.0060570, 0.0100503, -0.0159526, 0.0167463

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119622, upper bound: 0.0132005
time: 1.42 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120663, upper bound: 0.0131452
time: 1.25 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0043374, -0.0033275, -0.0043295, -0.0033837, -0.0009314, 0.0009676
1: -0.0018278, 0.0037643, -0.0015163, 0.0037204, -0.0053578, 0.0051570
2: 0.0065563, 0.0190496, 0.0066544, 0.0183536, -0.0115213, 0.0119698
3: -0.0006932, 0.0045715, -0.0003999, 0.0045302, -0.0050441, 0.0048551
4: 0.9940610, 1.0144860, 0.9951988, 1.0143256, -0.0195692, 0.0188359
5: 0.0010860, 0.0050595, 0.0013074, 0.0050283, -0.0038070, 0.0036643
6: -0.0123272, -0.0071562, -0.0122866, -0.0074443, -0.0047686, 0.0049542
7: -0.0103758, -0.0097162, -0.0103706, -0.0097530, -0.0006083, 0.0006320
8: -0.0059097, -0.0023370, -0.0057107, -0.0023651, -0.0034230, 0.0032947
9: -0.0064712, 0.0114145, -0.0063307, 0.0104181, -0.0164942, 0.0171363

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119622, upper bound: 0.0132005
time: 1.51 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120663, upper bound: 0.0131452
time: 1.57 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043511, -0.0033076, -0.0043140, -0.0034045, -0.0009194, 0.0009730
1: -0.0019376, 0.0038402, -0.0014012, 0.0036348, -0.0053876, 0.0050909
2: 0.0063867, 0.0192950, 0.0068456, 0.0180967, -0.0113735, 0.0120364
3: -0.0007966, 0.0046430, -0.0002916, 0.0044496, -0.0050722, 0.0047928
4: 0.9936598, 1.0147634, 0.9956189, 1.0140129, -0.0196781, 0.0185944
5: 0.0010080, 0.0051135, 0.0013891, 0.0049675, -0.0038282, 0.0036173
6: -0.0123974, -0.0070547, -0.0122074, -0.0075507, -0.0047074, 0.0049818
7: -0.0103848, -0.0097033, -0.0103605, -0.0097665, -0.0006005, 0.0006355
8: -0.0059799, -0.0022885, -0.0056372, -0.0024198, -0.0034420, 0.0032525
9: -0.0067141, 0.0117658, -0.0060570, 0.0100503, -0.0162826, 0.0172316

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119286, upper bound: 0.0133134
time: 1.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120313, upper bound: 0.0132553
time: 1.34 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043511, -0.0033076, -0.0043295, -0.0033837, -0.0009232, 0.0009684
1: -0.0019376, 0.0038402, -0.0015163, 0.0037204, -0.0053620, 0.0051120
2: 0.0063867, 0.0192950, 0.0066544, 0.0183536, -0.0114207, 0.0119793
3: -0.0007966, 0.0046430, -0.0003999, 0.0045302, -0.0050481, 0.0048127
4: 0.9936598, 1.0147634, 0.9951988, 1.0143256, -0.0195847, 0.0186715
5: 0.0010080, 0.0051135, 0.0013074, 0.0050283, -0.0038100, 0.0036324
6: -0.0123974, -0.0070547, -0.0122866, -0.0074443, -0.0047270, 0.0049582
7: -0.0103848, -0.0097033, -0.0103706, -0.0097530, -0.0006030, 0.0006325
8: -0.0059799, -0.0022885, -0.0057107, -0.0023651, -0.0034257, 0.0032660
9: -0.0067141, 0.0117658, -0.0063307, 0.0104181, -0.0163502, 0.0171499

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119286, upper bound: 0.0133134
time: 1.53 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120313, upper bound: 0.0132553
time: 1.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043374, -0.0033275, -0.0043570, -0.0033173, -0.0009282, 0.0009363
1: -0.0018278, 0.0037643, -0.0018839, 0.0038728, -0.0051845, 0.0051396
2: 0.0065563, 0.0190496, 0.0063139, 0.0191750, -0.0114824, 0.0115827
3: -0.0006932, 0.0045715, -0.0007460, 0.0046736, -0.0048810, 0.0048387
4: 0.9940610, 1.0144860, 0.9938560, 1.0148822, -0.0189364, 0.0187723
5: 0.0010860, 0.0050595, 0.0010461, 0.0051366, -0.0036839, 0.0036520
6: -0.0123272, -0.0071562, -0.0124275, -0.0071044, -0.0047525, 0.0047940
7: -0.0103758, -0.0097162, -0.0103886, -0.0097096, -0.0006062, 0.0006115
8: -0.0059097, -0.0023370, -0.0059456, -0.0022677, -0.0033123, 0.0032836
9: -0.0064712, 0.0114145, -0.0068182, 0.0115940, -0.0164384, 0.0165821

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119714, upper bound: 0.0132005
time: 1.28 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120729, upper bound: 0.0131452
time: 1.22 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0043374, -0.0033275, -0.0043706, -0.0032991, -0.0009528, 0.0009543
1: -0.0018278, 0.0037643, -0.0019849, 0.0039482, -0.0052842, 0.0052755
2: 0.0065563, 0.0190496, 0.0061455, 0.0194005, -0.0117860, 0.0118054
3: -0.0006932, 0.0045715, -0.0008411, 0.0047446, -0.0049748, 0.0049667
4: 0.9940610, 1.0144860, 0.9934872, 1.0151576, -0.0193004, 0.0192687
5: 0.0010860, 0.0050595, 0.0009744, 0.0051901, -0.0037547, 0.0037485
6: -0.0123272, -0.0071562, -0.0124972, -0.0070110, -0.0048782, 0.0048862
7: -0.0103758, -0.0097162, -0.0103975, -0.0096977, -0.0006223, 0.0006233
8: -0.0059097, -0.0023370, -0.0060101, -0.0022196, -0.0033760, 0.0033704
9: -0.0064712, 0.0114145, -0.0070593, 0.0119169, -0.0168732, 0.0169009

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119714, upper bound: 0.0132005
time: 1.56 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120729, upper bound: 0.0131452
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043511, -0.0033076, -0.0043570, -0.0033173, -0.0009490, 0.0009683
1: -0.0019376, 0.0038402, -0.0018839, 0.0038728, -0.0053613, 0.0052548
2: 0.0063867, 0.0192950, 0.0063139, 0.0191750, -0.0117398, 0.0119778
3: -0.0007966, 0.0046430, -0.0007460, 0.0046736, -0.0050475, 0.0049472
4: 0.9936598, 1.0147634, 0.9938560, 1.0148822, -0.0195823, 0.0191932
5: 0.0010080, 0.0051135, 0.0010461, 0.0051366, -0.0038095, 0.0037338
6: -0.0123974, -0.0070547, -0.0124275, -0.0071044, -0.0048590, 0.0049576
7: -0.0103848, -0.0097033, -0.0103886, -0.0097096, -0.0006198, 0.0006324
8: -0.0059799, -0.0022885, -0.0059456, -0.0022677, -0.0034253, 0.0033572
9: -0.0067141, 0.0117658, -0.0068182, 0.0115940, -0.0168070, 0.0171477

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119389, upper bound: 0.0133134
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120382, upper bound: 0.0132553
time: 1.59 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043511, -0.0033076, -0.0043706, -0.0032991, -0.0009492, 0.0009595
1: -0.0019376, 0.0038402, -0.0019849, 0.0039482, -0.0053127, 0.0052555
2: 0.0063867, 0.0192950, 0.0061455, 0.0194005, -0.0117413, 0.0118691
3: -0.0007966, 0.0046430, -0.0008411, 0.0047446, -0.0050017, 0.0049478
4: 0.9936598, 1.0147634, 0.9934872, 1.0151576, -0.0194046, 0.0191956
5: 0.0010080, 0.0051135, 0.0009744, 0.0051901, -0.0037750, 0.0037343
6: -0.0123974, -0.0070547, -0.0124972, -0.0070110, -0.0048596, 0.0049126
7: -0.0103848, -0.0097033, -0.0103975, -0.0096977, -0.0006199, 0.0006266
8: -0.0059799, -0.0022885, -0.0060101, -0.0022196, -0.0033942, 0.0033576
9: -0.0067141, 0.0117658, -0.0070593, 0.0119169, -0.0168091, 0.0169922

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119389, upper bound: 0.0133134
time: 1.59 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120382, upper bound: 0.0132553
time: 1.52 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043570, -0.0033173, -0.0043140, -0.0034045, -0.0009012, 0.0009373
1: -0.0018839, 0.0038728, -0.0014012, 0.0036348, -0.0051900, 0.0049897
2: 0.0063139, 0.0191750, 0.0068456, 0.0180967, -0.0111475, 0.0115950
3: -0.0007460, 0.0046736, -0.0002916, 0.0044496, -0.0048862, 0.0046976
4: 0.9938560, 1.0148822, 0.9956189, 1.0140129, -0.0189564, 0.0182249
5: 0.0010461, 0.0051366, 0.0013891, 0.0049675, -0.0036878, 0.0035455
6: -0.0124275, -0.0071044, -0.0122074, -0.0075507, -0.0046139, 0.0047991
7: -0.0103886, -0.0097096, -0.0103605, -0.0097665, -0.0005885, 0.0006122
8: -0.0059456, -0.0022677, -0.0056372, -0.0024198, -0.0033158, 0.0031878
9: -0.0068182, 0.0115940, -0.0060570, 0.0100503, -0.0159591, 0.0165997

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120645, upper bound: 0.0130225
time: 1.43 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122323, upper bound: 0.0129720
time: 1.33 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0043570, -0.0033173, -0.0043295, -0.0033837, -0.0009418, 0.0009642
1: -0.0018839, 0.0038728, -0.0015163, 0.0037204, -0.0053388, 0.0052146
2: 0.0063139, 0.0191750, 0.0066544, 0.0183536, -0.0116499, 0.0119275
3: -0.0007460, 0.0046736, -0.0003999, 0.0045302, -0.0050263, 0.0049093
4: 0.9938560, 1.0148822, 0.9951988, 1.0143256, -0.0195001, 0.0190462
5: 0.0010461, 0.0051366, 0.0013074, 0.0050283, -0.0037935, 0.0037052
6: -0.0124275, -0.0071044, -0.0122866, -0.0074443, -0.0048218, 0.0049367
7: -0.0103886, -0.0097096, -0.0103706, -0.0097530, -0.0006151, 0.0006297
8: -0.0059456, -0.0022677, -0.0057107, -0.0023651, -0.0034109, 0.0033315
9: -0.0068182, 0.0115940, -0.0063307, 0.0104181, -0.0166783, 0.0170757

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120645, upper bound: 0.0130225
time: 1.43 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122323, upper bound: 0.0129720
time: 1.44 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043706, -0.0032991, -0.0043140, -0.0034045, -0.0009207, 0.0009670
1: -0.0019849, 0.0039482, -0.0014012, 0.0036348, -0.0053542, 0.0050977
2: 0.0061455, 0.0194005, 0.0068456, 0.0180967, -0.0113888, 0.0119619
3: -0.0008411, 0.0047446, -0.0002916, 0.0044496, -0.0050408, 0.0047993
4: 0.9934872, 1.0151576, 0.9956189, 1.0140129, -0.0195562, 0.0186194
5: 0.0009744, 0.0051901, 0.0013891, 0.0049675, -0.0038045, 0.0036222
6: -0.0124972, -0.0070110, -0.0122074, -0.0075507, -0.0047138, 0.0049510
7: -0.0103975, -0.0096977, -0.0103605, -0.0097665, -0.0006013, 0.0006315
8: -0.0060101, -0.0022196, -0.0056372, -0.0024198, -0.0034207, 0.0032568
9: -0.0070593, 0.0119169, -0.0060570, 0.0100503, -0.0163045, 0.0171249

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120557, upper bound: 0.0131733
time: 1.27 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122108, upper bound: 0.0131215
time: 1.33 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043706, -0.0032991, -0.0043295, -0.0033837, -0.0009295, 0.0009625
1: -0.0019849, 0.0039482, -0.0015163, 0.0037204, -0.0053295, 0.0051467
2: 0.0061455, 0.0194005, 0.0066544, 0.0183536, -0.0114983, 0.0119068
3: -0.0008411, 0.0047446, -0.0003999, 0.0045302, -0.0050175, 0.0048454
4: 0.9934872, 1.0151576, 0.9951988, 1.0143256, -0.0194661, 0.0187983
5: 0.0009744, 0.0051901, 0.0013074, 0.0050283, -0.0037869, 0.0036570
6: -0.0124972, -0.0070110, -0.0122866, -0.0074443, -0.0047591, 0.0049281
7: -0.0103975, -0.0096977, -0.0103706, -0.0097530, -0.0006071, 0.0006286
8: -0.0060101, -0.0022196, -0.0057107, -0.0023651, -0.0034049, 0.0032881
9: -0.0070593, 0.0119169, -0.0063307, 0.0104181, -0.0164612, 0.0170460

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120557, upper bound: 0.0131733
time: 1.46 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122108, upper bound: 0.0131215
time: 1.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043570, -0.0033173, -0.0043570, -0.0033173, -0.0009334, 0.0009334
1: -0.0018839, 0.0038728, -0.0018839, 0.0038728, -0.0051681, 0.0051681
2: 0.0063139, 0.0191750, 0.0063139, 0.0191750, -0.0115462, 0.0115462
3: -0.0007460, 0.0046736, -0.0007460, 0.0046736, -0.0048656, 0.0048656
4: 0.9938560, 1.0148822, 0.9938560, 1.0148822, -0.0188766, 0.0188766
5: 0.0010461, 0.0051366, 0.0010461, 0.0051366, -0.0036723, 0.0036723
6: -0.0124275, -0.0071044, -0.0124275, -0.0071044, -0.0047789, 0.0047789
7: -0.0103886, -0.0097096, -0.0103886, -0.0097096, -0.0006096, 0.0006096
8: -0.0059456, -0.0022677, -0.0059456, -0.0022677, -0.0033018, 0.0033018
9: -0.0068182, 0.0115940, -0.0068182, 0.0115940, -0.0165298, 0.0165298

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120734, upper bound: 0.0130225
time: 1.40 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122383, upper bound: 0.0129720
time: 1.37 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0043570, -0.0033173, -0.0043706, -0.0032991, -0.0009678, 0.0009550
1: -0.0018839, 0.0038728, -0.0019849, 0.0039482, -0.0052875, 0.0053585
2: 0.0063139, 0.0191750, 0.0061455, 0.0194005, -0.0119716, 0.0118129
3: -0.0007460, 0.0046736, -0.0008411, 0.0047446, -0.0049780, 0.0050449
4: 0.9938560, 1.0148822, 0.9934872, 1.0151576, -0.0193127, 0.0195721
5: 0.0010461, 0.0051366, 0.0009744, 0.0051901, -0.0037571, 0.0038076
6: -0.0124275, -0.0071044, -0.0124972, -0.0070110, -0.0049550, 0.0048893
7: -0.0103886, -0.0097096, -0.0103975, -0.0096977, -0.0006321, 0.0006237
8: -0.0059456, -0.0022677, -0.0060101, -0.0022196, -0.0033781, 0.0034235
9: -0.0068182, 0.0115940, -0.0070593, 0.0119169, -0.0171388, 0.0169117

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120734, upper bound: 0.0130225
time: 1.53 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122383, upper bound: 0.0129720
time: 1.48 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0043706, -0.0032991, -0.0043570, -0.0033173, -0.0009550, 0.0009678
1: -0.0019849, 0.0039482, -0.0018839, 0.0038728, -0.0053585, 0.0052875
2: 0.0061455, 0.0194005, 0.0063139, 0.0191750, -0.0118129, 0.0119716
3: -0.0008411, 0.0047446, -0.0007460, 0.0046736, -0.0050449, 0.0049780
4: 0.9934872, 1.0151576, 0.9938560, 1.0148822, -0.0195721, 0.0193127
5: 0.0009744, 0.0051901, 0.0010461, 0.0051366, -0.0038076, 0.0037571
6: -0.0124972, -0.0070110, -0.0124275, -0.0071044, -0.0048893, 0.0049550
7: -0.0103975, -0.0096977, -0.0103886, -0.0097096, -0.0006237, 0.0006321
8: -0.0060101, -0.0022196, -0.0059456, -0.0022677, -0.0034235, 0.0033781
9: -0.0070593, 0.0119169, -0.0068182, 0.0115940, -0.0169117, 0.0171388

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120633, upper bound: 0.0131733
time: 1.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122164, upper bound: 0.0131215
time: 1.32 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0043706, -0.0032991, -0.0043706, -0.0032991, -0.0009582, 0.0009582
1: -0.0019849, 0.0039482, -0.0019849, 0.0039482, -0.0053053, 0.0053053
2: 0.0061455, 0.0194005, 0.0061455, 0.0194005, -0.0118527, 0.0118527
3: -0.0008411, 0.0047446, -0.0008411, 0.0047446, -0.0049948, 0.0049948
4: 0.9934872, 1.0151576, 0.9934872, 1.0151576, -0.0193777, 0.0193777
5: 0.0009744, 0.0051901, 0.0009744, 0.0051901, -0.0037697, 0.0037697
6: -0.0124972, -0.0070110, -0.0124972, -0.0070110, -0.0049058, 0.0049058
7: -0.0103975, -0.0096977, -0.0103975, -0.0096977, -0.0006258, 0.0006258
8: -0.0060101, -0.0022196, -0.0060101, -0.0022196, -0.0033895, 0.0033895
9: -0.0070593, 0.0119169, -0.0070593, 0.0119169, -0.0169686, 0.0169686

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120633, upper bound: 0.0131733
time: 1.62 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122164, upper bound: 0.0131215
time: 1.47 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.00 seconds
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0130552, upper bound: 0.0120861
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0130139, upper bound: 0.0120313
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0130552, upper bound: 0.0120861
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0130139, upper bound: 0.0120313
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0130268, upper bound: 0.0123079
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0129886, upper bound: 0.0122261
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0130268, upper bound: 0.0123079
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0129886, upper bound: 0.0122261
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0130448, upper bound: 0.0122649
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0130081, upper bound: 0.0122108
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0130448, upper bound: 0.0122649
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0130081, upper bound: 0.0122108
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0130197, upper bound: 0.0124628
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0129837, upper bound: 0.0123720
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0130197, upper bound: 0.0124628
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0129837, upper bound: 0.0123720
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131616, upper bound: 0.0120821
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131495, upper bound: 0.0120313
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131616, upper bound: 0.0120821
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131495, upper bound: 0.0120313
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131605, upper bound: 0.0123077
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131452, upper bound: 0.0122261
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131605, upper bound: 0.0123077
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131452, upper bound: 0.0122261
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131616, upper bound: 0.0120607
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131495, upper bound: 0.0120147
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131616, upper bound: 0.0120607
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131495, upper bound: 0.0120147
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131605, upper bound: 0.0122838
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131452, upper bound: 0.0122037
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131605, upper bound: 0.0122838
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0131452, upper bound: 0.0122037
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119636, upper bound: 0.0130385
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120663, upper bound: 0.0129886
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119636, upper bound: 0.0130385
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120663, upper bound: 0.0129886
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119304, upper bound: 0.0131914
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120313, upper bound: 0.0131422
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119304, upper bound: 0.0131914
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120313, upper bound: 0.0131422
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119736, upper bound: 0.0130385
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120734, upper bound: 0.0129886
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119736, upper bound: 0.0130385
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120734, upper bound: 0.0129886
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119408, upper bound: 0.0131914
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120384, upper bound: 0.0131422
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119408, upper bound: 0.0131914
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120384, upper bound: 0.0131422
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120645, upper bound: 0.0130298
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122323, upper bound: 0.0129837
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120645, upper bound: 0.0130298
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122323, upper bound: 0.0129837
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120557, upper bound: 0.0131911
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122108, upper bound: 0.0131411
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120557, upper bound: 0.0131911
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122108, upper bound: 0.0131411
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120734, upper bound: 0.0130298
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122383, upper bound: 0.0129837
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120734, upper bound: 0.0130298
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122383, upper bound: 0.0129837
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120633, upper bound: 0.0131911
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122164, upper bound: 0.0131411
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120633, upper bound: 0.0131911
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122164, upper bound: 0.0131411
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119622, upper bound: 0.0132005
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120663, upper bound: 0.0131452
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119622, upper bound: 0.0132005
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120663, upper bound: 0.0131452
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119286, upper bound: 0.0133134
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120313, upper bound: 0.0132553
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119286, upper bound: 0.0133134
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120313, upper bound: 0.0132553
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119714, upper bound: 0.0132005
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120729, upper bound: 0.0131452
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119714, upper bound: 0.0132005
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120729, upper bound: 0.0131452
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119389, upper bound: 0.0133134
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120382, upper bound: 0.0132553
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0119389, upper bound: 0.0133134
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120382, upper bound: 0.0132553
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120645, upper bound: 0.0130225
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122323, upper bound: 0.0129720
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120645, upper bound: 0.0130225
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122323, upper bound: 0.0129720
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120557, upper bound: 0.0131733
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122108, upper bound: 0.0131215
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120557, upper bound: 0.0131733
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122108, upper bound: 0.0131215
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120734, upper bound: 0.0130225
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122383, upper bound: 0.0129720
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120734, upper bound: 0.0130225
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122383, upper bound: 0.0129720
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120633, upper bound: 0.0131733
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122164, upper bound: 0.0131215
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0120633, upper bound: 0.0131733
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -0.0122164, upper bound: 0.0131215

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.67 + 515.79 = 520.46 seconds
