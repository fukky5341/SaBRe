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
execution time: IAR + RelationalAnalysis = 1.69 + 2.19 = 3.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.1562350, upper bound: 0.1562350

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 244

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1541907, upper bound: 0.1551434
time: 1.11 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1552235, upper bound: 0.1552235
time: 1.20 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.46 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.46
Output dim: 1, lower bound: -0.1541907, upper bound: 0.1551434
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.46
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

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 244

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1540038
time: 1.17 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1527090
time: 1.07 seconds

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

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 244

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1527703, upper bound: 0.1544183
time: 1.23 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1527703, upper bound: 0.1527703
time: 1.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.87 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.87
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1540038
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.87
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1527090
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.87
Output dim: 1, lower bound: -0.1527703, upper bound: 0.1544183
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.87
Output dim: 1, lower bound: -0.1527703, upper bound: 0.1527703

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0253394, 0.0295844, -0.0340452, 0.0430250, -0.0683645, 0.0636296
1: 0.9068240, 1.0335493, 0.8820515, 1.0371948, -0.1303709, 0.1514978
2: -0.0195756, 0.0469754, -0.0252478, 0.0607576, -0.0803332, 0.0722231
3: -0.0173963, 0.0189279, -0.0227768, 0.0300800, -0.0474763, 0.0417046
4: -0.0272567, 0.0182527, -0.0369338, 0.0181819, -0.0454386, 0.0551865
5: -0.0232900, 0.0327049, -0.0223076, 0.0420396, -0.0653296, 0.0550125
6: -0.0421663, 0.0297374, -0.0519019, 0.0286323, -0.0707986, 0.0816393
7: -0.0265226, 0.0496274, -0.0360975, 0.0619840, -0.0885065, 0.0857248
8: -0.0193811, 0.0276429, -0.0192827, 0.0369162, -0.0562973, 0.0469255
9: -0.0387490, 0.0300879, -0.0463178, 0.0389621, -0.0777111, 0.0764057

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 244

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 169

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1527090
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1527090
time: 1.20 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0249732, 0.0293575, -0.0526023, 0.0722481, -0.0972212, 0.0819599
1: 0.9078661, 1.0334222, 0.8292474, 1.0450090, -0.1371430, 0.2041749
2: -0.0194521, 0.0463956, -0.0375335, 0.0901355, -0.1095877, 0.0839290
3: -0.0172893, 0.0184587, -0.0344477, 0.0538518, -0.0711411, 0.0529064
4: -0.0268848, 0.0175225, -0.0576210, 0.0251522, -0.0520370, 0.0751435
5: -0.0223631, 0.0323122, -0.0268180, 0.0619373, -0.0814542, 0.0591302
6: -0.0417567, 0.0287008, -0.0726543, 0.0468528, -0.0886095, 0.1013550
7: -0.0261198, 0.0491637, -0.0565070, 0.0884184, -0.1145382, 0.1056707
8: -0.0189826, 0.0272527, -0.0338875, 0.0566831, -0.0756657, 0.0611402
9: -0.0385361, 0.0290799, -0.0626301, 0.0659451, -0.1044813, 0.0917101

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 244

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1487900, upper bound: 0.1475268
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1459520, upper bound: 0.1474389
time: 1.09 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0325268, 0.0406338, -0.0363010, 0.0465774, -0.0791042, 0.0769349
1: 0.8863722, 1.0365556, 0.8756327, 1.0381447, -0.1517725, 0.1609229
2: -0.0242425, 0.0583538, -0.0267412, 0.0643288, -0.0885713, 0.0850950
3: -0.0218218, 0.0281349, -0.0241955, 0.0329697, -0.0547915, 0.0523304
4: -0.0352411, 0.0203960, -0.0394486, 0.0190292, -0.0542703, 0.0598446
5: -0.0260890, 0.0404115, -0.0223245, 0.0444584, -0.0705474, 0.0627360
6: -0.0502039, 0.0322350, -0.0544246, 0.0286532, -0.0788571, 0.0866596
7: -0.0344275, 0.0598210, -0.0385784, 0.0651973, -0.0996248, 0.0983994
8: -0.0203412, 0.0352988, -0.0210580, 0.0393191, -0.0596602, 0.0563568
9: -0.0449830, 0.0367543, -0.0483008, 0.0422422, -0.0872252, 0.0850551

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 244

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 169

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1527703, upper bound: 0.1527703
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1527703, upper bound: 0.1527703
time: 1.09 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0321881, 0.0401005, -0.0546156, 0.0754185, -0.1076066, 0.0947161
1: 0.8873360, 1.0364128, 0.8235188, 1.0458566, -0.1585206, 0.2128941
2: -0.0240183, 0.0578175, -0.0388663, 0.0933227, -0.1173410, 0.0966839
3: -0.0216088, 0.0277010, -0.0357139, 0.0564307, -0.0780395, 0.0634149
4: -0.0348635, 0.0194919, -0.0598653, 0.0259084, -0.0607719, 0.0793573
5: -0.0249083, 0.0400483, -0.0285830, 0.0640960, -0.0890042, 0.0686313
6: -0.0498251, 0.0311814, -0.0749056, 0.0496361, -0.0994613, 0.1060871
7: -0.0340549, 0.0593385, -0.0587212, 0.0912863, -0.1253412, 0.1180597
8: -0.0199362, 0.0349380, -0.0354719, 0.0588275, -0.0787637, 0.0704099
9: -0.0446853, 0.0362618, -0.0643999, 0.0688725, -0.1135578, 0.1006617

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 244

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1498686, upper bound: 0.1475305
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1475305, upper bound: 0.1475305
time: 0.90 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.45 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1527090
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1527090
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.1487900, upper bound: 0.1475268
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.1459520, upper bound: 0.1474389
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.1527703, upper bound: 0.1527703
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.1527703, upper bound: 0.1527703
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.1498686, upper bound: 0.1475305
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.1475305, upper bound: 0.1475305

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0250240, 0.0293890, -0.0340452, 0.0430250, -0.0680490, 0.0634342
1: 0.9077213, 1.0334396, 0.8820515, 1.0371948, -0.1294736, 0.1513882
2: -0.0194693, 0.0464761, -0.0252478, 0.0607576, -0.0802269, 0.0717238
3: -0.0173042, 0.0185239, -0.0227768, 0.0300800, -0.0473842, 0.0413006
4: -0.0269365, 0.0173636, -0.0369338, 0.0181819, -0.0451184, 0.0542974
5: -0.0221643, 0.0323667, -0.0223076, 0.0420396, -0.0612634, 0.0546744
6: -0.0418136, 0.0284554, -0.0519019, 0.0286323, -0.0704459, 0.0803573
7: -0.0261757, 0.0492281, -0.0360975, 0.0619840, -0.0881597, 0.0853255
8: -0.0188883, 0.0273069, -0.0192827, 0.0369162, -0.0558045, 0.0465896
9: -0.0385657, 0.0288611, -0.0463178, 0.0389621, -0.0775278, 0.0751789

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 244

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1532126
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1540038
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0436273, 0.0409138, -0.0340452, 0.0430250, -0.0866523, 0.0749590
1: 0.8547862, 1.0399200, 0.8820515, 1.0371948, -0.1824086, 0.1578685
2: -0.0257409, 0.0759270, -0.0252478, 0.0607576, -0.0864986, 0.1011747
3: -0.0227399, 0.0423547, -0.0227768, 0.0300800, -0.0528199, 0.0651314
4: -0.0458265, 0.0217810, -0.0369338, 0.0181819, -0.0640084, 0.0587148
5: -0.0211440, 0.0523138, -0.0223076, 0.0420396, -0.0631836, 0.0746215
6: -0.0626175, 0.0271957, -0.0519019, 0.0286323, -0.0912498, 0.0790977
7: -0.0466360, 0.0727763, -0.0360975, 0.0619840, -0.1086199, 0.1088738
8: -0.0189714, 0.0471229, -0.0192827, 0.0369162, -0.0558876, 0.0664055
9: -0.0493785, 0.0528949, -0.0463178, 0.0389621, -0.0883407, 0.0992127

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 244

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1532126
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1540038
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0322336, 0.0401720, -0.0363010, 0.0465774, -0.0788109, 0.0764730
1: 0.8872066, 1.0364321, 0.8756327, 1.0381447, -0.1509381, 0.1607994
2: -0.0240484, 0.0578895, -0.0267412, 0.0643288, -0.0883771, 0.0846307
3: -0.0216374, 0.0277592, -0.0241955, 0.0329697, -0.0546071, 0.0519547
4: -0.0349142, 0.0192822, -0.0394486, 0.0190292, -0.0539434, 0.0587308
5: -0.0246344, 0.0400971, -0.0223245, 0.0444584, -0.0690928, 0.0624216
6: -0.0498759, 0.0309370, -0.0544246, 0.0286532, -0.0785291, 0.0853616
7: -0.0341049, 0.0594033, -0.0385784, 0.0651973, -0.0993023, 0.0979817
8: -0.0198422, 0.0349864, -0.0210580, 0.0393191, -0.0591613, 0.0560445
9: -0.0447253, 0.0363279, -0.0483008, 0.0422422, -0.0869675, 0.0846287

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 244

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1475305, upper bound: 0.1508108
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1475305, upper bound: 0.1495540
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0503322, 0.0686733, -0.0363010, 0.0465774, -0.0969096, 0.1049744
1: 0.8357071, 1.0440531, 0.8756327, 1.0381447, -0.2024376, 0.1684204
2: -0.0360306, 0.0865417, -0.0267412, 0.0643288, -0.1003594, 0.1132829
3: -0.0330201, 0.0509437, -0.0241955, 0.0329697, -0.0659898, 0.0751393
4: -0.0550903, 0.0242995, -0.0394486, 0.0190292, -0.0741195, 0.0637481
5: -0.0248281, 0.0595032, -0.0223245, 0.0444584, -0.0692864, 0.0818277
6: -0.0701156, 0.0437144, -0.0544246, 0.0286532, -0.0987688, 0.0981390
7: -0.0540103, 0.0851847, -0.0385784, 0.0651973, -0.1192076, 0.1237631
8: -0.0321009, 0.0542650, -0.0210580, 0.0393191, -0.0714200, 0.0753230
9: -0.0606347, 0.0626443, -0.0483008, 0.0422422, -0.1028769, 0.1109450

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 244

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1475305, upper bound: 0.1508108
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1475305, upper bound: 0.1495540
time: 1.11 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.81 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1532126
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1540038
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1532126
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 1, lower bound: -0.1516182, upper bound: 0.1540038
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.81
Output dim: 1, lower bound: -0.1475305, upper bound: 0.1508108
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.81
Output dim: 1, lower bound: -0.1475305, upper bound: 0.1495540
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.81
Output dim: 1, lower bound: -0.1475305, upper bound: 0.1508108
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.81
Output dim: 1, lower bound: -0.1475305, upper bound: 0.1495540

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0250240, 0.0293890, -0.0250240, 0.0293890, -0.0544130, 0.0544130
1: 0.9077213, 1.0334396, 0.9077213, 1.0334396, -0.1257184, 0.1257184
2: -0.0194693, 0.0464761, -0.0194693, 0.0464761, -0.0659453, 0.0659453
3: -0.0173042, 0.0185239, -0.0173042, 0.0185239, -0.0358280, 0.0358280
4: -0.0269365, 0.0173636, -0.0269365, 0.0173636, -0.0443001, 0.0443001
5: -0.0221643, 0.0323667, -0.0221643, 0.0323667, -0.0516344, 0.0516344
6: -0.0418136, 0.0284554, -0.0418136, 0.0284554, -0.0702690, 0.0702690
7: -0.0261757, 0.0492281, -0.0261757, 0.0492281, -0.0754038, 0.0754038
8: -0.0188883, 0.0273069, -0.0188883, 0.0273069, -0.0461952, 0.0461952
9: -0.0385657, 0.0288611, -0.0385657, 0.0288611, -0.0674267, 0.0674267

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 244

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1516669, upper bound: 0.1496504
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1496504, upper bound: 0.1496504
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0250240, 0.0293890, -0.0322336, 0.0401720, -0.0651960, 0.0616226
1: 0.9077213, 1.0334396, 0.8872066, 1.0364320, -0.1287107, 0.1462330
2: -0.0194693, 0.0464761, -0.0240484, 0.0578895, -0.0773588, 0.0705244
3: -0.0173042, 0.0185239, -0.0216374, 0.0277592, -0.0450634, 0.0401612
4: -0.0269365, 0.0173636, -0.0349142, 0.0189702, -0.0459067, 0.0522778
5: -0.0221643, 0.0323667, -0.0241745, 0.0400971, -0.0594895, 0.0554956
6: -0.0418136, 0.0284554, -0.0498759, 0.0309370, -0.0727506, 0.0783313
7: -0.0261757, 0.0492281, -0.0341049, 0.0594033, -0.0855790, 0.0833330
8: -0.0188883, 0.0273069, -0.0198422, 0.0349864, -0.0538747, 0.0471491
9: -0.0385657, 0.0288611, -0.0447253, 0.0363279, -0.0748936, 0.0735863

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 244

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1516669, upper bound: 0.1509830
time: 1.43 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1496504, upper bound: 0.1509008
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0436273, 0.0409138, -0.0250240, 0.0293890, -0.0730163, 0.0659378
1: 0.8547862, 1.0399200, 0.9077213, 1.0334396, -0.1786534, 0.1321987
2: -0.0257409, 0.0759270, -0.0194693, 0.0464761, -0.0722170, 0.0953962
3: -0.0227399, 0.0423547, -0.0173042, 0.0185239, -0.0412638, 0.0596588
4: -0.0458265, 0.0217810, -0.0269365, 0.0173636, -0.0631901, 0.0487175
5: -0.0211440, 0.0523138, -0.0221643, 0.0323667, -0.0535107, 0.0715518
6: -0.0626175, 0.0271957, -0.0418136, 0.0284554, -0.0910728, 0.0690094
7: -0.0466360, 0.0727763, -0.0261757, 0.0492281, -0.0958640, 0.0989521
8: -0.0189714, 0.0471229, -0.0188883, 0.0273069, -0.0462783, 0.0660112
9: -0.0493785, 0.0528949, -0.0385657, 0.0288611, -0.0782396, 0.0914606

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 244

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1489580, upper bound: 0.1483628
time: 1.38 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1459520, upper bound: 0.1482355
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0436273, 0.0409138, -0.0322336, 0.0401720, -0.0837993, 0.0731474
1: 0.8547862, 1.0399200, 0.8872066, 1.0364320, -0.1816458, 0.1527134
2: -0.0257409, 0.0759270, -0.0240484, 0.0578895, -0.0836304, 0.0999753
3: -0.0227399, 0.0423547, -0.0216374, 0.0277592, -0.0504992, 0.0639920
4: -0.0458265, 0.0217810, -0.0349142, 0.0189702, -0.0647967, 0.0566952
5: -0.0211440, 0.0523138, -0.0241745, 0.0400971, -0.0612410, 0.0754129
6: -0.0626175, 0.0271957, -0.0498759, 0.0309370, -0.0935545, 0.0770717
7: -0.0466360, 0.0727763, -0.0341049, 0.0594033, -0.1060392, 0.1068813
8: -0.0189714, 0.0471229, -0.0198422, 0.0349864, -0.0539578, 0.0669651
9: -0.0493785, 0.0528949, -0.0447253, 0.0363279, -0.0857064, 0.0976201

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 244

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1489580, upper bound: 0.1491649
time: 1.46 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1459520, upper bound: 0.1489586
time: 1.14 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.14 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 1, lower bound: -0.1516669, upper bound: 0.1496504
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.14
Output dim: 1, lower bound: -0.1496504, upper bound: 0.1496504
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 1, lower bound: -0.1516669, upper bound: 0.1509830
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.14
Output dim: 1, lower bound: -0.1496504, upper bound: 0.1509008
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.14
Output dim: 1, lower bound: -0.1489580, upper bound: 0.1483628
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.14
Output dim: 1, lower bound: -0.1459520, upper bound: 0.1482355
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.14
Output dim: 1, lower bound: -0.1489580, upper bound: 0.1491649
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.14
Output dim: 1, lower bound: -0.1459520, upper bound: 0.1489586

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0214056, 0.0263463, -0.0241044, 0.0288193, -0.0502249, 0.0504507
1: 0.9205754, 1.0317289, 0.9103382, 1.0331193, -0.1125439, 0.1213906
2: -0.0178135, 0.0396222, -0.0191592, 0.0450201, -0.0628337, 0.0587814
3: -0.0158691, 0.0141801, -0.0170355, 0.0173457, -0.0332148, 0.0312155
4: -0.0233904, 0.0172844, -0.0260026, 0.0173498, -0.0407402, 0.0432870
5: -0.0220651, 0.0290015, -0.0221469, 0.0313806, -0.0504348, 0.0482840
6: -0.0377120, 0.0283330, -0.0407851, 0.0284340, -0.0661459, 0.0691181
7: -0.0225202, 0.0430111, -0.0251643, 0.0480639, -0.0705841, 0.0681753
8: -0.0188412, 0.0235383, -0.0188800, 0.0263273, -0.0451685, 0.0424183
9: -0.0357110, 0.0287519, -0.0380311, 0.0288420, -0.0645529, 0.0667830

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 244

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1472443, upper bound: 0.1427897
time: 1.42 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1508506, upper bound: 0.1487646
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0214056, 0.0263463, -0.0312805, 0.0386712, -0.0600768, 0.0576269
1: 0.9205754, 1.0317289, 0.8899184, 1.0360308, -0.1154554, 0.1418104
2: -0.0178135, 0.0396222, -0.0234174, 0.0563808, -0.0741943, 0.0630396
3: -0.0158691, 0.0141801, -0.0210380, 0.0265384, -0.0424075, 0.0352181
4: -0.0233904, 0.0172844, -0.0338518, 0.0189563, -0.0423467, 0.0511362
5: -0.0220651, 0.0290015, -0.0241571, 0.0390752, -0.0582460, 0.0517670
6: -0.0377120, 0.0283330, -0.0488102, 0.0309155, -0.0686275, 0.0771432
7: -0.0225202, 0.0430111, -0.0330568, 0.0580457, -0.0805658, 0.0760678
8: -0.0188412, 0.0235383, -0.0198340, 0.0339713, -0.0528125, 0.0433722
9: -0.0357110, 0.0287519, -0.0438875, 0.0349421, -0.0706531, 0.0726394

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 244

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1474440, upper bound: 0.1447441
time: 1.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1508710, upper bound: 0.1500842
time: 1.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.22 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.22
Output dim: 1, lower bound: -0.1472443, upper bound: 0.1427897
NS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.22
Output dim: 1, lower bound: -0.1508506, upper bound: 0.1487646
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.22
Output dim: 1, lower bound: -0.1474440, upper bound: 0.1447441
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.22
Output dim: 1, lower bound: -0.1508710, upper bound: 0.1500842

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 3.88 + 65.23 = 69.10 seconds
