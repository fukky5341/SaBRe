## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.15154795000000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0402414, 0.0519706, -0.0402414, 0.0519706, -0.0922120, 0.0922120)
1: (0.8656265, 1.0395867, 0.8656265, 1.0395867, -0.1739601, 0.1739601)
2: (-0.0293665, 0.0697505, -0.0293665, 0.0697505, -0.0991171, 0.0991171)
3: (-0.0263494, 0.0377974, -0.0263494, 0.0377974, -0.0641469, 0.0641469)
4: (-0.0432664, 0.0205233, -0.0432664, 0.0205233, -0.0637898, 0.0637898)
5: (-0.0237454, 0.0481305, -0.0237454, 0.0481305, -0.0718759, 0.0718759)
6: (-0.0584624, 0.0301437, -0.0584624, 0.0301437, -0.0886062, 0.0886062)
7: (-0.0423451, 0.0703274, -0.0423451, 0.0703274, -0.1126725, 0.1126725)
8: (-0.0237534, 0.0443120, -0.0237534, 0.0443120, -0.0680653, 0.0680653)
9: (-0.0520953, 0.0472220, -0.0520953, 0.0472220, -0.0993173, 0.0993173)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.62 + 2.21 = 3.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.1562350, upper bound: 0.1562350

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1541907, upper bound: 0.1551434
time: 1.10 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1552235, upper bound: 0.1552235
time: 1.18 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.43 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.43
Output dim: 1, lower bound: -0.1541907, upper bound: 0.1551434
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.43
Output dim: 1, lower bound: -0.1552235, upper bound: 0.1552235

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0253705, 0.0296037, -0.0343831, 0.0435571, -0.0689275, 0.0639868
1: 0.9067354, 1.0335605, 0.8810902, 1.0373369, -0.1306016, 0.1524703
2: -0.0195861, 0.0470245, -0.0254714, 0.0612925, -0.0808786, 0.0724960
3: -0.0174054, 0.0189677, -0.0229893, 0.0305129, -0.0479183, 0.0419569
4: -0.0272883, 0.0183630, -0.0373105, 0.0185162, -0.0458045, 0.0556735
5: -0.0234339, 0.0327383, -0.0236341, 0.0424019, -0.0658358, 0.0563724
6: -0.0422011, 0.0298658, -0.0522798, 0.0300444, -0.0722455, 0.0821456
7: -0.0265568, 0.0496666, -0.0364690, 0.0624653, -0.0890220, 0.0861357
8: -0.0194304, 0.0276760, -0.0195486, 0.0372761, -0.0567065, 0.0472246
9: -0.0387670, 0.0302405, -0.0466148, 0.0394534, -0.0782205, 0.0768554

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 169

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1532105, upper bound: 0.1527322
time: 1.34 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1527090
time: 1.06 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0325555, 0.0406790, -0.0366314, 0.0470976, -0.0796531, 0.0773105
1: 0.8862906, 1.0365674, 0.8746924, 1.0382841, -0.1519935, 0.1618751
2: -0.0242615, 0.0583992, -0.0269600, 0.0648519, -0.0891134, 0.0853592
3: -0.0218398, 0.0281717, -0.0244033, 0.0333930, -0.0552328, 0.0525750
4: -0.0352731, 0.0205074, -0.0398169, 0.0191533, -0.0544264, 0.0603243
5: -0.0262345, 0.0404423, -0.0236577, 0.0448127, -0.0710472, 0.0641000
6: -0.0502360, 0.0323648, -0.0547941, 0.0300656, -0.0803015, 0.0871589
7: -0.0344590, 0.0598619, -0.0389418, 0.0656680, -0.1001270, 0.0988037
8: -0.0203911, 0.0353293, -0.0213180, 0.0396710, -0.0600621, 0.0566474
9: -0.0450083, 0.0367960, -0.0485912, 0.0427226, -0.0877309, 0.0853872

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 169

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1544183, upper bound: 0.1527703
time: 1.30 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1527703, upper bound: 0.1527703
time: 1.09 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.91 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.91
Output dim: 1, lower bound: -0.1532105, upper bound: 0.1527322
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.91
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1527090
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.91
Output dim: 1, lower bound: -0.1544183, upper bound: 0.1527703
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.91
Output dim: 1, lower bound: -0.1527703, upper bound: 0.1527703

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -0.0250240, 0.0293890, -0.0343527, 0.0435092, -0.0685332, 0.0637417
1: 0.9077213, 1.0334396, 0.8811767, 1.0373244, -0.1296031, 0.1522629
2: -0.0194693, 0.0464761, -0.0254513, 0.0612444, -0.0807137, 0.0719274
3: -0.0173042, 0.0185239, -0.0229702, 0.0304739, -0.0477781, 0.0414940
4: -0.0269365, 0.0173636, -0.0372766, 0.0184058, -0.0453423, 0.0546403
5: -0.0221643, 0.0323667, -0.0234898, 0.0423694, -0.0615950, 0.0558566
6: -0.0418136, 0.0284554, -0.0522458, 0.0299158, -0.0717294, 0.0807012
7: -0.0261757, 0.0492281, -0.0364356, 0.0624220, -0.0885977, 0.0856637
8: -0.0188883, 0.0273069, -0.0195247, 0.0372438, -0.0561320, 0.0468316
9: -0.0385657, 0.0288611, -0.0465881, 0.0394093, -0.0779749, 0.0754492

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1483602, upper bound: 0.1493898
time: 1.18 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1482340, upper bound: 0.1474815
time: 1.30 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -0.0436273, 0.0409138, -0.0339932, 0.0429431, -0.0865704, 0.0749070
1: 0.8547862, 1.0399200, 0.8821996, 1.0371730, -0.1823868, 0.1577203
2: -0.0257409, 0.0759270, -0.0252134, 0.0606752, -0.0864162, 0.1011403
3: -0.0227399, 0.0423547, -0.0227441, 0.0300134, -0.0527533, 0.0650988
4: -0.0458265, 0.0217810, -0.0368759, 0.0181623, -0.0639888, 0.0586569
5: -0.0211440, 0.0523138, -0.0225077, 0.0419839, -0.0631278, 0.0748215
6: -0.0626175, 0.0271957, -0.0518438, 0.0288794, -0.0914968, 0.0790395
7: -0.0466360, 0.0727763, -0.0360402, 0.0619099, -0.1085459, 0.1088166
8: -0.0189714, 0.0471229, -0.0192417, 0.0368608, -0.0558322, 0.0663646
9: -0.0493785, 0.0528949, -0.0462721, 0.0388866, -0.0882651, 0.0991670

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 244

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_A2_A1

### Relational analysis result of NS_A1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1487900, upper bound: 0.1475267
time: 1.03 seconds

## Relational analysis of NS_A1_A2_A2

### Relational analysis result of NS_A1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1459520, upper bound: 0.1474389
time: 1.08 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.0322336, 0.0401720, -0.0366018, 0.0470510, -0.0792846, 0.0767738
1: 0.8872066, 1.0364321, 0.8747768, 1.0382714, -0.1510648, 0.1616553
2: -0.0240484, 0.0578895, -0.0269403, 0.0648049, -0.0888532, 0.0848298
3: -0.0216374, 0.0277592, -0.0243847, 0.0333549, -0.0549923, 0.0521439
4: -0.0349142, 0.0192822, -0.0397838, 0.0191421, -0.0540563, 0.0590660
5: -0.0246344, 0.0400971, -0.0235134, 0.0447808, -0.0694152, 0.0636105
6: -0.0498759, 0.0309370, -0.0547609, 0.0299368, -0.0798127, 0.0856979
7: -0.0341049, 0.0594033, -0.0389092, 0.0656257, -0.0997306, 0.0983124
8: -0.0198422, 0.0349864, -0.0212947, 0.0396394, -0.0594816, 0.0562811
9: -0.0447253, 0.0363279, -0.0485651, 0.0426794, -0.0874047, 0.0848930

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1496859, upper bound: 0.1500557
time: 1.02 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1495540, upper bound: 0.1475305
time: 1.31 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.0503322, 0.0686733, -0.0362507, 0.0464979, -0.0968302, 0.1049240
1: 0.8357071, 1.0440531, 0.8757762, 1.0381235, -0.2024164, 0.1682768
2: -0.0360306, 0.0865417, -0.0267078, 0.0642490, -0.1002796, 0.1132495
3: -0.0330201, 0.0509437, -0.0241638, 0.0329052, -0.0659252, 0.0751075
4: -0.0550903, 0.0242995, -0.0393924, 0.0190102, -0.0741005, 0.0636919
5: -0.0248281, 0.0595032, -0.0225246, 0.0444044, -0.0692324, 0.0820278
6: -0.0701156, 0.0437144, -0.0543682, 0.0289002, -0.0990158, 0.0980827
7: -0.0540103, 0.0851847, -0.0385230, 0.0651255, -0.1191358, 0.1237077
8: -0.0321009, 0.0542650, -0.0210183, 0.0392654, -0.0713663, 0.0752833
9: -0.0606347, 0.0626443, -0.0482564, 0.0421689, -0.1028036, 0.1109007

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 244

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1498686, upper bound: 0.1475305
time: 0.93 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1475305, upper bound: 0.1475305
time: 0.97 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.44 seconds
NS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.44
Output dim: 1, lower bound: -0.1483602, upper bound: 0.1493898
NS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 3.44
Output dim: 1, lower bound: -0.1482340, upper bound: 0.1474815
NS_A1_A2_A1, status: Status.VERIFIED, split count: 3, time: 3.44
Output dim: 1, lower bound: -0.1487900, upper bound: 0.1475267
NS_A1_A2_A2, status: Status.VERIFIED, split count: 3, time: 3.44
Output dim: 1, lower bound: -0.1459520, upper bound: 0.1474389
NS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.44
Output dim: 1, lower bound: -0.1496859, upper bound: 0.1500557
NS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 3.44
Output dim: 1, lower bound: -0.1495540, upper bound: 0.1475305
NS_A2_A2_A1, status: Status.VERIFIED, split count: 3, time: 3.44
Output dim: 1, lower bound: -0.1498686, upper bound: 0.1475305
NS_A2_A2_A2, status: Status.VERIFIED, split count: 3, time: 3.44
Output dim: 1, lower bound: -0.1475305, upper bound: 0.1475305

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 3.83 + 25.39 = 29.22 seconds
