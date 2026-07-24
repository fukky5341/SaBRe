## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 7.125826784999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.5828104, 2.9839077, -3.5828104, 2.9839077, -6.5667181, 6.5667181)
1: (-2.8148921, 2.6596637, -2.8148921, 2.6596637, -5.4745560, 5.4745560)
2: (-3.6384213, 2.7152596, -3.6384213, 2.7152596, -6.3536806, 6.3536806)
3: (-4.0147090, 2.3897834, -4.0147090, 2.3897834, -6.4044924, 6.4044924)
4: (-3.9739902, 2.8952332, -3.9739902, 2.8952332, -6.8692236, 6.8692236)
5: (-3.4501100, 2.9820681, -3.4501100, 2.9820681, -6.4321771, 6.4321775)
6: (-3.2314649, 3.2592940, -3.2314649, 3.2592940, -6.4907575, 6.4907589)
7: (-3.3493385, 3.3119178, -3.3493385, 3.3119178, -6.6612563, 6.6612563)
8: (-5.2216048, 3.1262531, -5.2216048, 3.1262531, -8.3478584, 8.3478584)
9: (-3.0501204, 3.2086091, -3.0501204, 3.2086091, -6.2587285, 6.2587290)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.43 + 4.34 = 6.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -7.5008703, upper bound: 7.5008703

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4938696, upper bound: 7.4930696
time: 3.66 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4959173, upper bound: 7.4959173
time: 2.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.29 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.29
Output dim: 8, lower bound: -7.4938696, upper bound: 7.4930696
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.29
Output dim: 8, lower bound: -7.4959173, upper bound: 7.4959173

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1.7904269, 1.7163006, -2.8684251, 2.4657815, -4.2562084, 4.5847259
1: -1.3990718, 1.3745172, -2.2228565, 2.1429036, -3.5419755, 3.5973735
2: -1.6261857, 1.5388798, -2.8159959, 2.2397528, -3.8659384, 4.3548756
3: -1.7995743, 1.3206804, -3.1067121, 1.9638295, -3.7634039, 4.4273920
4: -1.9525543, 1.4930053, -3.1694994, 2.3271608, -4.2797151, 4.6625032
5: -1.6417933, 1.6875238, -2.7474301, 2.4465799, -4.0883727, 4.4349537
6: -1.6235850, 1.7214158, -2.5761538, 2.6398225, -4.2634077, 4.2975693
7: -1.6227205, 1.6054351, -2.6546645, 2.6200500, -4.2427702, 4.2600994
8: -2.6390195, 2.5188105, -4.1978035, 2.8312182, -5.4702368, 6.7166119
9: -1.5091753, 1.7492748, -2.4107089, 2.6002607, -4.1094356, 4.1599836

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4195438, upper bound: 7.4706562
time: 2.43 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4207828, upper bound: 7.4622532
time: 2.12 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2.3783793, 2.1090565, -2.9227881, 2.5016522, -4.8800316, 5.0318446
1: -1.8500359, 1.7999758, -2.2661645, 2.1878176, -4.0378532, 4.0661392
2: -2.2550316, 1.9139918, -2.8818772, 2.2745039, -4.5295353, 4.7958679
3: -2.4873400, 1.6777272, -3.1878562, 2.0015781, -4.4889178, 4.8655834
4: -2.6327772, 1.9358397, -3.2337775, 2.3720829, -5.0048594, 5.1696172
5: -2.2501426, 2.0904293, -2.8040721, 2.4854674, -4.7356100, 4.8945017
6: -2.1383657, 2.2156138, -2.6273608, 2.6858032, -4.8241682, 4.8429747
7: -2.1827118, 2.1469598, -2.7105036, 2.6780200, -4.8607321, 4.8574624
8: -3.5094748, 2.6777301, -4.2735291, 2.8402700, -6.3497438, 6.9512577
9: -1.9794099, 2.1979864, -2.4617519, 2.6475041, -4.6269131, 4.6597385

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4864519, upper bound: 7.4865487
time: 2.23 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4862333, upper bound: 7.4862333
time: 2.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 6.54 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.54
Output dim: 8, lower bound: -7.4195438, upper bound: 7.4706562
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.54
Output dim: 8, lower bound: -7.4207828, upper bound: 7.4622532
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.54
Output dim: 8, lower bound: -7.4864519, upper bound: 7.4865487
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 6.54
Output dim: 8, lower bound: -7.4862333, upper bound: 7.4862333

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -1.6644319, 1.6408154, -1.7137214, 1.6766438, -3.3410759, 3.3545365
1: -1.3037171, 1.2875750, -1.3387966, 1.3188577, -2.6225748, 2.6263716
2: -1.5025301, 1.4622022, -1.5445948, 1.4970777, -2.9996076, 3.0067966
3: -1.6561375, 1.2500294, -1.7023755, 1.2845348, -2.9406724, 2.9524050
4: -1.8075714, 1.4033145, -1.8540708, 1.4405560, -3.2481275, 3.2573848
5: -1.5255907, 1.6014646, -1.5698130, 1.6229903, -3.1485806, 3.1712770
6: -1.5169406, 1.6209831, -1.5617654, 1.6670033, -3.1839437, 3.1827483
7: -1.5079299, 1.4973835, -1.5466461, 1.5385100, -3.0464399, 3.0440297
8: -2.4495409, 2.4957640, -2.5137484, 2.5328274, -4.9823685, 5.0095124
9: -1.4193006, 1.6540914, -1.4585522, 1.6871978, -3.1064978, 3.1126437

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2944888, upper bound: 7.3607205
time: 2.89 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3387020, upper bound: 7.3851328
time: 2.01 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1.0090883, 1.2884505, -2.1625345, 1.9999852, -3.0090735, 3.4509850
1: -0.8247454, 0.8654059, -1.6673034, 1.6167608, -2.4415064, 2.5327094
2: -0.9132095, 1.0760412, -2.0080607, 1.7999389, -2.7131484, 3.0841019
3: -0.9257931, 0.9128846, -2.2135172, 1.5098573, -2.4356503, 3.1264014
4: -1.0728855, 0.9504691, -2.3626003, 1.7670398, -2.8399253, 3.3130693
5: -0.9506341, 1.1727417, -1.9931209, 1.9198492, -2.8704832, 3.1658623
6: -0.9841619, 1.1213677, -1.9582313, 2.1091380, -3.0932999, 3.0795989
7: -0.9276341, 0.9779597, -1.9550350, 1.9271657, -2.8547997, 2.9329946
8: -1.4985094, 2.3866138, -3.2175388, 2.6737447, -4.1722536, 5.6041527
9: -0.9675943, 1.1732383, -1.7942113, 2.0251856, -2.9927800, 2.9674497

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2964815, upper bound: 7.3628522
time: 2.40 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3405962, upper bound: 7.3800475
time: 2.03 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -2.2414036, 2.0121465, -1.7424040, 1.6885873, -3.9299908, 3.7545497
1: -1.7445176, 1.6992686, -1.3619242, 1.3413614, -3.0858788, 3.0611923
2: -2.1032104, 1.8233771, -1.5734639, 1.5119345, -3.6151450, 3.3968410
3: -2.3262606, 1.5948508, -1.7370912, 1.3050642, -3.6313248, 3.3319421
4: -2.4745994, 1.8275082, -1.8897538, 1.4616427, -3.9362421, 3.7172613
5: -2.1042488, 1.9917607, -1.5972019, 1.6396700, -3.7439189, 3.5889623
6: -2.0180717, 2.0976107, -1.5861967, 1.6864560, -3.7045267, 3.6838069
7: -2.0498803, 2.0154262, -1.5754290, 1.5658225, -3.6157019, 3.5908546
8: -3.3070512, 2.6428494, -2.5528138, 2.5296478, -5.8366990, 5.1956625
9: -1.8579632, 2.0911312, -1.4786639, 1.7078984, -3.5658617, 3.5697951

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3618147, upper bound: 7.3981574
time: 14.87 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4104931, upper bound: 7.4234588
time: 2.41 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1.5750521, 1.5871346, -2.2112510, 2.0287881, -3.6038399, 3.7983856
1: -1.2371783, 1.2307738, -1.7049586, 1.6558721, -2.8930504, 2.9357324
2: -1.4212860, 1.4074912, -2.0603659, 1.8289073, -3.2501931, 3.4678571
3: -1.5526693, 1.2224444, -2.2737639, 1.5430254, -3.0956943, 3.4962080
4: -1.7029769, 1.3438687, -2.4196382, 1.8037325, -3.5067096, 3.7635069
5: -1.4480603, 1.5361688, -2.0454061, 1.9529350, -3.4009953, 3.5815749
6: -1.4529839, 1.5492222, -2.0008845, 2.1454561, -3.5984399, 3.5501068
7: -1.4245747, 1.4231308, -2.0029531, 1.9748700, -3.3994446, 3.4260838
8: -2.3133109, 2.5131917, -3.2842946, 2.6772974, -4.9906077, 5.7974858
9: -1.3618596, 1.5874465, -1.8330259, 2.0605557, -3.4224150, 3.4204724

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3631342, upper bound: 7.3933769
time: 2.11 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4113788, upper bound: 7.4113788
time: 1.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.22 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 8, lower bound: -7.2944888, upper bound: 7.3607205
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 8, lower bound: -7.3387020, upper bound: 7.3851328
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 8, lower bound: -7.2964815, upper bound: 7.3628522
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 8, lower bound: -7.3405962, upper bound: 7.3800475
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 8, lower bound: -7.3618147, upper bound: 7.3981574
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 8, lower bound: -7.4104931, upper bound: 7.4234588
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 8, lower bound: -7.3631342, upper bound: 7.3933769
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 8, lower bound: -7.4113788, upper bound: 7.4113788

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.6604033, 1.0719094, -1.1966481, 1.3934041, -2.0538073, 2.2685575
1: -0.5588167, 0.6168309, -0.9579214, 0.9811317, -1.5399485, 1.5747523
2: -0.6362213, 0.8040456, -1.0710146, 1.1921558, -1.8283771, 1.8750602
3: -0.5170172, 0.7032932, -1.1208987, 1.0197713, -1.5367886, 1.8241919
4: -0.6913581, 0.6755425, -1.2676277, 1.0810935, -1.7724516, 1.9431702
5: -0.6804694, 0.8595977, -1.1076059, 1.2754225, -1.9558918, 1.9672036
6: -0.6927806, 0.8457949, -1.1383796, 1.2613182, -1.9540988, 1.9841745
7: -0.6048710, 0.6796276, -1.0862125, 1.1153589, -1.7202300, 1.7658401
8: -0.8807111, 2.3415484, -1.7585901, 2.4494071, -3.3301182, 4.1001387
9: -0.7199733, 0.9059470, -1.0995560, 1.3104388, -2.0304122, 2.0055029

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2934616, upper bound: 7.3607060
time: 2.36 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2932975, upper bound: 7.3600370
time: 2.34 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.3096693, 1.4660015, -1.2346320, 1.4101517, -2.7198210, 2.7006335
1: -1.0405762, 1.0486480, -0.9851384, 1.0049642, -2.0455403, 2.0337863
2: -1.1898177, 1.2593513, -1.1024454, 1.2142060, -2.4040236, 2.3617966
3: -1.2675786, 1.0562760, -1.1619952, 1.0387847, -2.3063633, 2.2182713
4: -1.4081628, 1.1638970, -1.3082589, 1.1067939, -2.5149567, 2.4721560
5: -1.2093939, 1.3512235, -1.1401706, 1.3023731, -2.5117669, 2.4913940
6: -1.2382910, 1.3785274, -1.1680647, 1.2887155, -2.5270066, 2.5465922
7: -1.1964425, 1.2081563, -1.1188446, 1.1429815, -2.3394241, 2.3270011
8: -1.9551749, 2.4784641, -1.8142805, 2.4540932, -4.4092684, 4.2927446
9: -1.1787066, 1.4025815, -1.1250383, 1.3382610, -2.5169678, 2.5276198

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3384537, upper bound: 7.3849593
time: 2.34 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3383722, upper bound: 7.3842035
time: 1.83 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4509433, 0.9213976, -1.6396568, 1.6871601, -2.1381035, 2.5610545
1: -0.4026869, 0.4461986, -1.2748151, 1.2618518, -1.6645386, 1.7210137
2: -0.5223874, 0.5849748, -1.4834788, 1.4803048, -2.0026922, 2.0684536
3: -0.3565547, 0.5281928, -1.6199077, 1.2372180, -1.5937726, 2.1481004
4: -0.5099156, 0.4918909, -1.7636709, 1.3985820, -1.9084976, 2.2555618
5: -0.5068033, 0.6638021, -1.5015780, 1.5583098, -2.0651131, 2.1653800
6: -0.5085055, 0.6725510, -1.5235827, 1.6913557, -2.1998613, 2.1961336
7: -0.4387167, 0.4899987, -1.4764766, 1.4778095, -1.9165262, 1.9664752
8: -0.5416775, 2.2865796, -2.4526048, 2.5804062, -3.1220837, 4.7391844
9: -0.5972656, 0.7325609, -1.4168640, 1.6374439, -2.2347095, 2.1494250

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2959633, upper bound: 7.3628395
time: 1.98 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2959589, upper bound: 7.3628395
time: 1.90 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.8082110, 1.1952378, -1.7323375, 1.7389759, -2.5471869, 2.9275751
1: -0.6771582, 0.7197545, -1.3428243, 1.3244066, -2.0015647, 2.0625789
2: -0.7505199, 0.9400219, -1.5697773, 1.5357760, -2.2862954, 2.5097992
3: -0.6920498, 0.7939886, -1.7237737, 1.2855995, -1.9776492, 2.5177624
4: -0.8618427, 0.7997097, -1.8690655, 1.4627256, -2.3245683, 2.6687751
5: -0.8035487, 0.9961737, -1.5832663, 1.6250799, -2.4286284, 2.5794396
6: -0.8441677, 1.0034873, -1.6001687, 1.7637310, -2.6078987, 2.6036561
7: -0.7418897, 0.8184930, -1.5587249, 1.5549569, -2.2968466, 2.3772173
8: -1.1982960, 2.4036372, -2.5884182, 2.5962853, -3.7945812, 4.9920554
9: -0.8310171, 1.0241507, -1.4827703, 1.7047936, -2.5358107, 2.5069208

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3403828, upper bound: 7.3800356
time: 2.43 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3403809, upper bound: 7.3800356
time: 1.78 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.9232294, 1.2408991, -1.2146335, 1.3978631, -2.3210926, 2.4555326
1: -0.7652404, 0.8029776, -0.9715042, 0.9941047, -1.7593451, 1.7744818
2: -0.8442199, 1.0180205, -1.0862646, 1.2015872, -2.0458071, 2.1042852
3: -0.8268901, 0.8764247, -1.1425086, 1.0315875, -1.8584776, 2.0189333
4: -0.9827690, 0.8856735, -1.2898800, 1.0930655, -2.0758345, 2.1755536
5: -0.8840476, 1.0880773, -1.1238694, 1.2828383, -2.1668859, 2.2119467
6: -0.9215509, 1.0408099, -1.1533250, 1.2699888, -2.1915398, 2.1941347
7: -0.8510733, 0.9153663, -1.1032848, 1.1308347, -1.9819080, 2.0186510
8: -1.3361745, 2.4181969, -1.7771221, 2.4435511, -3.7797256, 4.1953192
9: -0.9174763, 1.1059439, -1.1124413, 1.3227754, -2.2402515, 2.2183852

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3616130, upper bound: 7.3981433
time: 2.18 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3615113, upper bound: 7.3970817
time: 1.91 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.8811039, 1.7922350, -1.2763115, 1.4265852, -3.3076892, 3.0685463
1: -1.4638401, 1.4535459, -1.0163610, 1.0326304, -2.4964705, 2.4699068
2: -1.7348163, 1.6005261, -1.1390870, 1.2368448, -2.9716611, 2.7396131
3: -1.9108193, 1.3705553, -1.2101375, 1.0629994, -2.9738188, 2.5806928
4: -2.0610080, 1.5687672, -1.3572567, 1.1357692, -3.1967773, 2.9260240
5: -1.7235694, 1.7350659, -1.1768166, 1.3250451, -3.0486145, 2.9118824
6: -1.7219322, 1.8292836, -1.2021866, 1.3154142, -3.0373464, 3.0314703
7: -1.7075202, 1.6908091, -1.1569352, 1.1776798, -2.8852000, 2.8477445
8: -2.7978184, 2.6187303, -1.8644521, 2.4520597, -5.2498779, 4.4831824
9: -1.5920621, 1.8225181, -1.1540279, 1.3673198, -2.9593816, 2.9765460

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4103592, upper bound: 7.4232525
time: 2.37 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4102677, upper bound: 7.4221819
time: 1.89 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5856534, 1.0233067, -1.6682224, 1.6983217, -2.2839751, 2.6915293
1: -0.5025458, 0.5577666, -1.2962345, 1.2825959, -1.7851417, 1.8540010
2: -0.6077563, 0.7245248, -1.5092120, 1.4945880, -2.1023443, 2.2337368
3: -0.4535851, 0.6540283, -1.6521312, 1.2557218, -1.7093070, 2.3061595
4: -0.6227089, 0.6177711, -1.7963661, 1.4182355, -2.0409443, 2.4141374
5: -0.6234531, 0.7938094, -1.5263480, 1.5771606, -2.2006137, 2.3201573
6: -0.6357372, 0.7883623, -1.5472497, 1.7074611, -2.3431983, 2.3356118
7: -0.5504196, 0.6163483, -1.5014143, 1.5029955, -2.0534151, 2.1177626
8: -0.7565401, 2.3532240, -2.4880211, 2.5793347, -3.3358748, 4.8412452
9: -0.6793424, 0.8502942, -1.4367967, 1.6566249, -2.3359673, 2.2870908

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3630314, upper bound: 7.3933313
time: 2.51 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3630307, upper bound: 7.3933313
time: 1.79 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.2644219, 1.4421397, -1.7899113, 1.7665401, -3.0309620, 3.2320509
1: -1.0062795, 1.0237105, -1.3872120, 1.3654062, -2.3716855, 2.4109225
2: -1.1537962, 1.2332265, -1.6256673, 1.5680896, -2.7218857, 2.8588939
3: -1.2165635, 1.0479270, -1.7902460, 1.3190756, -2.5356390, 2.8381729
4: -1.3581568, 1.1342630, -1.9358253, 1.5033934, -2.8615503, 3.0700884
5: -1.1755419, 1.3162020, -1.6356277, 1.6631547, -2.8386965, 2.9518299
6: -1.2121418, 1.3432291, -1.6478865, 1.8037691, -3.0159109, 2.9911156
7: -1.1547265, 1.1776512, -1.6111139, 1.6061581, -2.7608848, 2.7887650
8: -1.8841169, 2.5052302, -2.6679316, 2.5990410, -4.4831581, 5.1731615
9: -1.1554089, 1.3685433, -1.5233201, 1.7464607, -2.9018695, 2.8918633

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4113645, upper bound: 7.4113736
time: 1.89 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4113612, upper bound: 7.4113613
time: 2.20 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.07 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.2934616, upper bound: 7.3607060
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.2932975, upper bound: 7.3600370
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.3384537, upper bound: 7.3849593
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.3383722, upper bound: 7.3842035
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.2959633, upper bound: 7.3628395
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.2959589, upper bound: 7.3628395
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.3403828, upper bound: 7.3800356
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.3403809, upper bound: 7.3800356
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.3616130, upper bound: 7.3981433
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.3615113, upper bound: 7.3970817
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.4103592, upper bound: 7.4232525
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.4102677, upper bound: 7.4221819
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.3630314, upper bound: 7.3933313
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.3630307, upper bound: 7.3933313
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.4113645, upper bound: 7.4113736
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.07
Output dim: 8, lower bound: -7.4113612, upper bound: 7.4113613

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.6022237, 1.0322185, -0.6739707, 1.0705833, -1.6728070, 1.7061892
1: -0.5164574, 0.5741402, -0.5767761, 0.6354084, -1.1518658, 1.1509163
2: -0.6047728, 0.7439234, -0.6379288, 0.8169422, -1.4217150, 1.3818523
3: -0.4612060, 0.6668717, -0.5382153, 0.7215713, -1.1827774, 1.2050869
4: -0.6342062, 0.6293182, -0.7066220, 0.6888401, -1.3230463, 1.3359402
5: -0.6326343, 0.8144579, -0.6935887, 0.8912386, -1.5238730, 1.5080466
6: -0.6418812, 0.7995944, -0.7039378, 0.8600124, -1.5018936, 1.5035322
7: -0.5608115, 0.6285416, -0.6222413, 0.6945822, -1.2553936, 1.2507830
8: -0.7866328, 2.3113065, -0.9218965, 2.2957733, -3.0824060, 3.2332029
9: -0.6806982, 0.8606857, -0.7289648, 0.9140950, -1.5947933, 1.5896505

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2904377, upper bound: 7.3607060
time: 2.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2904377, upper bound: 7.3607060
time: 2.34 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.6233006, 1.0475907, -1.0058517, 1.2936472, -1.9169477, 2.0534425
1: -0.5312918, 0.5905225, -0.8247293, 0.8637244, -1.3950162, 1.4152519
2: -0.6166010, 0.7671356, -0.9069967, 1.0766062, -1.6932073, 1.6741323
3: -0.4789834, 0.6808735, -0.9149143, 0.9257681, -1.4047515, 1.5957878
4: -0.6558170, 0.6447660, -1.0643413, 0.9480805, -1.6038976, 1.7091073
5: -0.6509556, 0.8311492, -0.9446214, 1.1585288, -1.8094845, 1.7757707
6: -0.6617235, 0.8174412, -0.9833688, 1.1197238, -1.7814473, 1.8008099
7: -0.5750794, 0.6479888, -0.9233294, 0.9755685, -1.5506480, 1.5713181
8: -0.8209350, 2.3241143, -1.4824998, 2.3954215, -3.2163565, 3.8066142
9: -0.6954219, 0.8779669, -0.9734226, 1.1667858, -1.8622077, 1.8513895

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2902558, upper bound: 7.3600370
time: 2.03 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2902558, upper bound: 7.3600370
time: 2.23 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.1801815, 1.3954059, -0.6994615, 1.0869027, -2.2670841, 2.0948675
1: -0.9456593, 0.9699237, -0.5983433, 0.6541771, -1.5998365, 1.5682670
2: -1.0737381, 1.1823547, -0.6541464, 0.8403158, -1.9140539, 1.8365011
3: -1.1238887, 0.9928282, -0.5658172, 0.7391605, -1.8630493, 1.5586455
4: -1.2638850, 1.0757276, -0.7351555, 0.7101219, -1.9740069, 1.8108830
5: -1.0970939, 1.2677398, -0.7121854, 0.9199894, -2.0170834, 1.9799252
6: -1.1315682, 1.2796471, -0.7263233, 0.8775436, -2.0091119, 2.0059705
7: -1.0805194, 1.1126935, -0.6463754, 0.7167480, -1.7972674, 1.7590690
8: -1.7693151, 2.4411712, -0.9715132, 2.2997284, -4.0690436, 3.4126844
9: -1.0900190, 1.3071127, -0.7464838, 0.9337733, -2.0237923, 2.0535965

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3384537, upper bound: 7.3849593
time: 1.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3384537, upper bound: 7.3849593
time: 2.54 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.2276496, 1.4208734, -1.0390149, 1.3085237, -2.5361733, 2.4598885
1: -0.9801375, 0.9982856, -0.8478414, 0.8854210, -1.8655585, 1.8461270
2: -1.1148107, 1.2107056, -0.9348449, 1.0959561, -2.2107668, 2.1455505
3: -1.1765786, 1.0158782, -0.9498053, 0.9427016, -2.1192803, 1.9656835
4: -1.3159691, 1.1077981, -1.0986683, 0.9713022, -2.2872713, 2.2064664
5: -1.1380867, 1.2974572, -0.9707209, 1.1841996, -2.3222861, 2.2681782
6: -1.1704690, 1.3157332, -1.0085931, 1.1430318, -2.3135009, 2.3243263
7: -1.1224835, 1.1464082, -0.9505291, 1.0003759, -2.1228595, 2.0969372
8: -1.8384951, 2.4556632, -1.5309809, 2.3991601, -4.2376552, 3.9866443
9: -1.1230202, 1.3420751, -0.9933393, 1.1923594, -2.3153796, 2.3354144

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3383722, upper bound: 7.3842035
time: 1.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3383722, upper bound: 7.3842035
time: 2.10 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.4170818, 0.8927680, -1.0292594, 1.3419545, -1.7590363, 1.9220275
1: -0.3768610, 0.4228941, -0.8356001, 0.8787787, -1.2556397, 1.2584943
2: -0.4984785, 0.5541636, -0.9329625, 1.1123397, -1.6108183, 1.4871261
3: -0.3337274, 0.4966508, -0.9522721, 0.9231151, -1.2568425, 1.4489229
4: -0.4869863, 0.4562470, -1.0905058, 0.9776077, -1.4645941, 1.5467528
5: -0.4766631, 0.6314623, -0.9811282, 1.1659238, -1.6425869, 1.6125906
6: -0.4763687, 0.6381921, -1.0384731, 1.2192764, -1.6956451, 1.6766652
7: -0.4116370, 0.4618566, -0.9448408, 1.0089806, -1.4206177, 1.4066974
8: -0.4827521, 2.2584829, -1.5912855, 2.4270742, -2.9098263, 3.8497684
9: -0.5777852, 0.7032723, -0.9955012, 1.1899787, -1.7677639, 1.6987734

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2959633, upper bound: 7.3628395
time: 2.02 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2959633, upper bound: 7.3628395
time: 1.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4305769, 0.9043104, -1.4146618, 1.5600498, -1.9906267, 2.3189721
1: -0.3872223, 0.4317050, -1.1097922, 1.1139578, -1.5011801, 1.5414972
2: -0.5083133, 0.5661226, -1.2731721, 1.3461752, -1.8544885, 1.8392947
3: -0.3431571, 0.5086558, -1.3660259, 1.1223458, -1.4655030, 1.8746817
4: -0.4959988, 0.4702547, -1.5098547, 1.2415053, -1.7375040, 1.9801093
5: -0.4886996, 0.6447248, -1.3062600, 1.4081556, -1.8968551, 1.9509847
6: -0.4894492, 0.6520681, -1.3387386, 1.5131075, -2.0025568, 1.9908067
7: -0.4219680, 0.4729443, -1.2768059, 1.2908885, -1.7128565, 1.7497501
8: -0.5068536, 2.2704787, -2.1382823, 2.5178933, -3.0247469, 4.4087610
9: -0.5857224, 0.7149873, -1.2591357, 1.4716612, -2.0573835, 1.9741230

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2959589, upper bound: 7.3628395
time: 1.90 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2959589, upper bound: 7.3628395
time: 2.08 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.7345006, 1.1426119, -1.1173278, 1.3845747, -2.1190753, 2.2599397
1: -0.6176111, 0.6685187, -0.8968343, 0.9354728, -1.5530839, 1.5653529
2: -0.6928087, 0.8755065, -1.0079947, 1.1636461, -1.8564548, 1.8835012
3: -0.6057630, 0.7457688, -1.0472268, 0.9681171, -1.5738801, 1.7929956
4: -0.7785339, 0.7392524, -1.1843126, 1.0387971, -1.8173311, 1.9235650
5: -0.7450264, 0.9285606, -1.0538694, 1.2272201, -1.9722464, 1.9824300
6: -0.7753348, 0.9475566, -1.1068400, 1.2829427, -2.0582776, 2.0543966
7: -0.6722565, 0.7518897, -1.0185847, 1.0731752, -1.7454317, 1.7704744
8: -1.0627915, 2.3720562, -1.7199322, 2.4388015, -3.5015931, 4.0919886
9: -0.7751667, 0.9671547, -1.0514584, 1.2556810, -2.0308478, 2.0186131

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3403828, upper bound: 7.3800356
time: 2.09 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3403828, upper bound: 7.3800356
time: 2.13 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.7620908, 1.1626146, -1.5049447, 1.6069827, -2.3690734, 2.6675591
1: -0.6395695, 0.6871092, -1.1756589, 1.1741209, -1.8136904, 1.8627682
2: -0.7135624, 0.8995209, -1.3551672, 1.3979249, -2.1114874, 2.2546880
3: -0.6367436, 0.7637855, -1.4667265, 1.1695652, -1.8063087, 2.2305121
4: -0.8097941, 0.7612563, -1.6106284, 1.3030185, -2.1128125, 2.3718846
5: -0.7660409, 0.9547557, -1.3834776, 1.4737411, -2.2397819, 2.3382332
6: -0.8010005, 0.9686634, -1.4112663, 1.5828334, -2.3838339, 2.3799298
7: -0.6979823, 0.7757157, -1.3551803, 1.3646350, -2.0626173, 2.1308961
8: -1.1134890, 2.3849695, -2.2631965, 2.5326209, -3.6461101, 4.6481657
9: -0.7957430, 0.9886754, -1.3216002, 1.5367991, -2.3325419, 2.3102756

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3403809, upper bound: 7.3800356
time: 2.00 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3403809, upper bound: 7.3800356
time: 1.87 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.8322498, 1.1850362, -0.7004451, 1.0875987, -1.9198484, 1.8854814
1: -0.7034584, 0.7412930, -0.6004770, 0.6564432, -1.3599017, 1.3417699
2: -0.7666458, 0.9527197, -0.6551537, 0.8420749, -1.6087207, 1.6078734
3: -0.7213194, 0.8246922, -0.5694261, 0.7436039, -1.4649234, 1.3941183
4: -0.8897752, 0.8151414, -0.7407029, 0.7096941, -1.5994692, 1.5558443
5: -0.8161122, 1.0174716, -0.7133935, 0.9138858, -1.7299980, 1.7308650
6: -0.8505735, 0.9686354, -0.7277776, 0.8724749, -1.7230484, 1.6964130
7: -0.7706822, 0.8414289, -0.6504713, 0.7206440, -1.4913261, 1.4919002
8: -1.1909038, 2.3845396, -0.9634861, 2.2923608, -3.4832644, 3.3480258
9: -0.8507230, 1.0374013, -0.7486471, 0.9334549, -1.7841779, 1.7860484

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3599337, upper bound: 7.3981433
time: 1.94 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3599337, upper bound: 7.3981433
time: 3.49 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.8662504, 1.2060694, -1.0215731, 1.2971675, -2.1634178, 2.2276425
1: -0.7270438, 0.7641703, -0.8367071, 0.8758706, -1.6029143, 1.6008775
2: -0.7938067, 0.9779952, -0.9207725, 1.0848358, -1.8786424, 1.8987677
3: -0.7601589, 0.8447755, -0.9336939, 0.9369257, -1.6970847, 1.7784693
4: -0.9239636, 0.8415502, -1.0838915, 0.9583921, -1.8823557, 1.9254417
5: -0.8419056, 1.0449027, -0.9574349, 1.1653671, -2.0072727, 2.0023375
6: -0.8778191, 0.9954851, -0.9947326, 1.1261113, -2.0039303, 1.9902177
7: -0.7998810, 0.8692393, -0.9380343, 0.9893066, -1.7891877, 1.8072736
8: -1.2466459, 2.3986623, -1.4974959, 2.3889756, -3.6356215, 3.8961582
9: -0.8759086, 1.0626280, -0.9833414, 1.1782351, -2.0541437, 2.0459695

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3598566, upper bound: 7.3970817
time: 2.44 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3598566, upper bound: 7.3970817
time: 2.22 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.7376125, 1.7008811, -0.7367371, 1.1100731, -2.8476856, 2.4376183
1: -1.3567759, 1.3506123, -0.6305752, 0.6832844, -2.0400603, 1.9811870
2: -1.5935736, 1.5093957, -0.6816792, 0.8732042, -2.4667778, 2.1910748
3: -1.7495670, 1.2929313, -0.6110197, 0.7698605, -2.5194268, 1.9039510
4: -1.8968751, 1.4654326, -0.7809832, 0.7395154, -2.6363907, 2.2464159
5: -1.5915512, 1.6370357, -0.7414942, 0.9523318, -2.5438831, 2.3785300
6: -1.5990515, 1.7101737, -0.7607927, 0.8974773, -2.4965286, 2.4709663
7: -1.5777082, 1.5705233, -0.6843454, 0.7531301, -2.3308382, 2.2548685
8: -2.5787437, 2.5714588, -1.0322177, 2.2984281, -4.8771715, 3.6036763
9: -1.4868275, 1.7142080, -0.7756845, 0.9608727, -2.4477000, 2.4898925

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4103592, upper bound: 7.4232034
time: 2.02 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4103592, upper bound: 7.4232525
time: 2.27 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.7908785, 1.7348039, -1.0771101, 1.3224270, -3.1133056, 2.8119140
1: -1.3959680, 1.3882332, -0.8746793, 0.9113801, -2.3073480, 2.2629123
2: -1.6456136, 1.5434850, -0.9671730, 1.1169690, -2.7625825, 2.5106580
3: -1.8089863, 1.3218861, -0.9923444, 0.9650909, -2.7740772, 2.3142304
4: -1.9575517, 1.5034589, -1.1415920, 0.9965069, -2.9540586, 2.6450510
5: -1.6403434, 1.6739486, -1.0028265, 1.2037027, -2.8440456, 2.6767750
6: -1.6447177, 1.7541381, -1.0391105, 1.1643343, -2.8090510, 2.7932487
7: -1.6251174, 1.6149375, -0.9836514, 1.0298382, -2.6549556, 2.5985889
8: -2.6595345, 2.5902803, -1.5770597, 2.3961518, -5.0556860, 4.1673393
9: -1.5261647, 1.7541101, -1.0182581, 1.2194474, -2.7456119, 2.7723682

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4102677, upper bound: 7.4221612
time: 2.08 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4102677, upper bound: 7.4221819
time: 2.00 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.5397759, 0.9850761, -1.0909514, 1.3678867, -1.9076626, 2.0760274
1: -0.4668141, 0.5190037, -0.8799457, 0.9205451, -1.3873591, 1.3989494
2: -0.5812287, 0.6706985, -0.9866579, 1.1472933, -1.7285221, 1.6573564
3: -0.4156176, 0.6193910, -1.0210080, 0.9581566, -1.3737742, 1.6403990
4: -0.5796647, 0.5789957, -1.1591802, 1.0204566, -1.6001213, 1.7381759
5: -0.5805899, 0.7545478, -1.0311568, 1.2047954, -1.7853853, 1.7857046
6: -0.5892011, 0.7479651, -1.0852578, 1.2559580, -1.8451591, 1.8332229
7: -0.5152521, 0.5720899, -0.9984978, 1.0560002, -1.5712523, 1.5705878
8: -0.6773130, 2.3235409, -1.6713262, 2.4295657, -3.1068788, 3.9948671
9: -0.6494801, 0.8090690, -1.0354115, 1.2359234, -1.8854035, 1.8444805

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3630314, upper bound: 7.3933313
time: 2.09 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3630314, upper bound: 7.3933313
time: 2.43 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.5566978, 0.9995926, -1.4417863, 1.5701677, -2.1268654, 2.4413788
1: -0.4798536, 0.5324590, -1.1299434, 1.1336652, -1.6135188, 1.6624024
2: -0.5918155, 0.6891727, -1.2969722, 1.3600088, -1.9518244, 1.9861449
3: -0.4292242, 0.6326338, -1.3968021, 1.1402116, -1.5694358, 2.0294359
4: -0.5937822, 0.5937127, -1.5412186, 1.2597491, -1.8535314, 2.1349313
5: -0.5954599, 0.7705187, -1.3293345, 1.4263984, -2.0218582, 2.0998530
6: -0.6071437, 0.7619562, -1.3611140, 1.5274518, -2.1345954, 2.1230702
7: -0.5288408, 0.5881307, -1.3006811, 1.3139343, -1.8427751, 1.8888118
8: -0.7064505, 2.3359118, -2.1677895, 2.5162687, -3.2227192, 4.5037012
9: -0.6607888, 0.8240585, -1.2784271, 1.4898912, -2.1506801, 2.1024857

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3630307, upper bound: 7.3933313
time: 2.41 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3630307, upper bound: 7.3933313
time: 1.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.1341789, 1.3720675, -1.2004883, 1.4206176, -2.5547965, 2.5725558
1: -0.9116240, 0.9446407, -0.9568105, 0.9885737, -1.9001977, 1.9014512
2: -1.0396098, 1.1545315, -1.0801492, 1.2103846, -2.2499943, 2.2346807
3: -1.0739248, 0.9824536, -1.1381456, 1.0139905, -2.0879154, 2.1205993
4: -1.2162730, 1.0442920, -1.2757158, 1.0950434, -2.3113165, 2.3200078
5: -1.0620821, 1.2352834, -1.1240492, 1.2777529, -2.3398349, 2.3593326
6: -1.1056513, 1.2438573, -1.1691148, 1.3359118, -2.4415631, 2.4129720
7: -1.0397555, 1.0830868, -1.0923781, 1.1348926, -2.1746480, 2.1754651
8: -1.6957815, 2.4674864, -1.8285248, 2.4439468, -4.1397285, 4.2960110
9: -1.0651143, 1.2725036, -1.1080163, 1.3153993, -2.3805137, 2.3805199

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4113645, upper bound: 7.4113736
time: 2.12 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4113645, upper bound: 7.4113736
time: 1.85 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.1819932, 1.3973868, -1.5611608, 1.6318026, -2.8137958, 2.9585476
1: -0.9455951, 0.9734703, -1.2175348, 1.2134522, -2.1590474, 2.1910052
2: -1.0805985, 1.1835191, -1.4063170, 1.4284523, -2.5090508, 2.5898361
3: -1.1251314, 1.0067142, -1.5308901, 1.2021115, -2.3272429, 2.5376043
4: -1.2673559, 1.0768542, -1.6752803, 1.3408420, -2.6081979, 2.7521346
5: -1.1032308, 1.2649094, -1.4317906, 1.5104622, -2.6136930, 2.6967001
6: -1.1439996, 1.2801731, -1.4570086, 1.6201575, -2.7641571, 2.7371817
7: -1.0812266, 1.1172030, -1.4049407, 1.4135070, -2.4947336, 2.5221438
8: -1.7654586, 2.4824023, -2.3321135, 2.5346520, -4.3001108, 4.8145161
9: -1.0983988, 1.3076875, -1.3609004, 1.5759259, -2.6743248, 2.6685879

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4113612, upper bound: 7.4113613
time: 1.57 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4113612, upper bound: 7.4113613
time: 1.71 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.29 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.2904377, upper bound: 7.3607060
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.2904377, upper bound: 7.3607060
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.2902558, upper bound: 7.3600370
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.2902558, upper bound: 7.3600370
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3384537, upper bound: 7.3849593
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3384537, upper bound: 7.3849593
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3383722, upper bound: 7.3842035
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3383722, upper bound: 7.3842035
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.2959633, upper bound: 7.3628395
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.2959633, upper bound: 7.3628395
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.2959589, upper bound: 7.3628395
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.2959589, upper bound: 7.3628395
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3403828, upper bound: 7.3800356
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3403828, upper bound: 7.3800356
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3403809, upper bound: 7.3800356
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3403809, upper bound: 7.3800356
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3599337, upper bound: 7.3981433
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3599337, upper bound: 7.3981433
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3598566, upper bound: 7.3970817
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3598566, upper bound: 7.3970817
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.4103592, upper bound: 7.4232034
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.4103592, upper bound: 7.4232525
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.4102677, upper bound: 7.4221612
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.4102677, upper bound: 7.4221819
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3630314, upper bound: 7.3933313
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3630314, upper bound: 7.3933313
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3630307, upper bound: 7.3933313
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.3630307, upper bound: 7.3933313
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.4113645, upper bound: 7.4113736
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.4113645, upper bound: 7.4113736
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.4113612, upper bound: 7.4113613
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 8, lower bound: -7.4113612, upper bound: 7.4113613

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3783033, 0.8678583, -0.6739707, 1.0705833, -1.4488866, 1.5418290
1: -0.3470758, 0.3975554, -0.5767761, 0.6354084, -0.9824842, 0.9743315
2: -0.4751938, 0.5209014, -0.6379288, 0.8169422, -1.2921360, 1.1588303
3: -0.3097731, 0.4606312, -0.5382153, 0.7215713, -1.0313444, 0.9988464
4: -0.4647347, 0.4246072, -0.7066220, 0.6888401, -1.1535748, 1.1312293
5: -0.4433243, 0.5906065, -0.6935887, 0.8912386, -1.3345630, 1.2841952
6: -0.4468969, 0.6024786, -0.7039378, 0.8600124, -1.3069093, 1.3064165
7: -0.3836117, 0.4336890, -0.6222413, 0.6945822, -1.0781939, 1.0559303
8: -0.4263002, 2.2455223, -0.9218965, 2.2957733, -2.7220736, 3.1674187
9: -0.5624911, 0.6710683, -0.7289648, 0.9140950, -1.4765861, 1.4000330

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2904377, upper bound: 7.3607060
time: 2.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2904377, upper bound: 7.3607060
time: 1.78 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.5261134, 1.0079412, -0.6739707, 1.0705833, -1.5966967, 1.6819119
1: -0.4442931, 0.4856760, -0.5767761, 0.6354084, -1.0797015, 1.0624521
2: -0.5669682, 0.6623378, -0.6379288, 0.8169422, -1.3839104, 1.3002667
3: -0.4130421, 0.5696286, -0.5382153, 0.7215713, -1.1346135, 1.1078439
4: -0.5645721, 0.5521503, -0.7066220, 0.6888401, -1.2534122, 1.2587724
5: -0.5834848, 0.6834676, -0.6935887, 0.8912386, -1.4747233, 1.3770564
6: -0.5939755, 0.8285921, -0.7039378, 0.8600124, -1.4539880, 1.5325298
7: -0.4851544, 0.5609637, -0.6222413, 0.6945822, -1.1797365, 1.1832049
8: -0.7330458, 2.3610268, -0.9218965, 2.2957733, -3.0288191, 3.2829232
9: -0.6420557, 0.7904769, -0.7289648, 0.9140950, -1.5561508, 1.5194417

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2904377, upper bound: 7.3607060
time: 2.61 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2904377, upper bound: 7.3607060
time: 2.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3911785, 0.8781557, -1.0058517, 1.2936472, -1.6848257, 1.8840075
1: -0.3567599, 0.4055300, -0.8247293, 0.8637244, -1.2204843, 1.2302594
2: -0.4835346, 0.5322151, -0.9069967, 1.0766062, -1.5601408, 1.4392118
3: -0.3183548, 0.4712301, -0.9149143, 0.9257681, -1.2441230, 1.3861444
4: -0.4725828, 0.4348520, -1.0643413, 0.9480805, -1.4206634, 1.4991933
5: -0.4546919, 0.6028759, -0.9446214, 1.1585288, -1.6132207, 1.5474973
6: -0.4579631, 0.6156703, -0.9833688, 1.1197238, -1.5776869, 1.5990391
7: -0.3922257, 0.4433523, -0.9233294, 0.9755685, -1.3677943, 1.3666818
8: -0.4469463, 2.2573762, -1.4824998, 2.3954215, -2.8423676, 3.7398760
9: -0.5692567, 0.6820841, -0.9734226, 1.1667858, -1.7360425, 1.6555067

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2902558, upper bound: 7.3600370
time: 2.35 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2902558, upper bound: 7.3600370
time: 2.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.5411456, 1.0196595, -1.0058517, 1.2936472, -1.8347929, 2.0255113
1: -0.4548426, 0.4962285, -0.8247293, 0.8637244, -1.3185670, 1.3209578
2: -0.5768038, 0.6776195, -0.9069967, 1.0766062, -1.6534100, 1.5846162
3: -0.4233616, 0.5828517, -0.9149143, 0.9257681, -1.3491297, 1.4977660
4: -0.5772454, 0.5665039, -1.0643413, 0.9480805, -1.5253260, 1.6308452
5: -0.5956576, 0.6988931, -0.9446214, 1.1585288, -1.7541864, 1.6435146
6: -0.6070620, 0.8424712, -0.9833688, 1.1197238, -1.7267858, 1.8258400
7: -0.4971397, 0.5741321, -0.9233294, 0.9755685, -1.4727081, 1.4974616
8: -0.7588208, 2.3713553, -1.4824998, 2.3954215, -3.1542423, 3.8538551
9: -0.6512109, 0.8030875, -0.9734226, 1.1667858, -1.8179967, 1.7765100

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2902558, upper bound: 7.3600370
time: 2.39 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2902558, upper bound: 7.3600370
time: 2.43 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.8213495, 1.1880453, -0.6524650, 1.0521817, -1.8735312, 1.8405102
1: -0.6895053, 0.7324307, -0.5595080, 0.6218922, -1.3113974, 1.2919387
2: -0.7571179, 0.9437820, -0.6236323, 0.7955788, -1.5526967, 1.5674143
3: -0.7087191, 0.8078192, -0.5160440, 0.7072311, -1.4159502, 1.3238633
4: -0.8741490, 0.8098165, -0.6838694, 0.6710073, -1.5451562, 1.4936858
5: -0.8089724, 1.0267889, -0.6762617, 0.8780213, -1.6869936, 1.7030506
6: -0.8449974, 0.9962063, -0.6824772, 0.8410568, -1.6860542, 1.6786835
7: -0.7571899, 0.8276304, -0.6036313, 0.6752087, -1.4323986, 1.4312617
8: -1.2165058, 2.3568790, -0.8830496, 2.2715445, -3.4880502, 3.2399287
9: -0.8346833, 1.0315689, -0.7122262, 0.8979390, -1.7326223, 1.7437950

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3356866, upper bound: 7.3482706
time: 1.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3356866, upper bound: 7.3849593
time: 2.16 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.0776675, 1.3412449, -0.6798444, 1.0722868, -2.1499543, 2.0210893
1: -0.8736141, 0.9083317, -0.5821863, 0.6407756, -1.5143896, 1.4905181
2: -0.9861754, 1.1198763, -0.6411119, 0.8217516, -1.8079270, 1.7609881
3: -1.0133747, 0.9436170, -0.5449710, 0.7260435, -1.7394181, 1.4885881
4: -1.1545190, 1.0060172, -0.7134111, 0.6939924, -1.8485113, 1.7194283
5: -1.0091349, 1.2065884, -0.6973723, 0.9023911, -1.9115260, 1.9039607
6: -1.0513515, 1.2012881, -0.7082237, 0.8622312, -1.9135828, 1.9095118
7: -0.9922954, 1.0385352, -0.6283440, 0.6994059, -1.6917013, 1.6668792
8: -1.6183665, 2.4121742, -0.9347602, 2.2900367, -3.9084032, 3.3469343
9: -1.0197767, 1.2312454, -0.7323995, 0.9186783, -1.9384550, 1.9636449

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3356866, upper bound: 7.3482706
time: 2.37 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3356866, upper bound: 7.3849593
time: 2.45 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.8546511, 1.2104809, -0.9635826, 1.2633549, -2.1180060, 2.1740634
1: -0.7143145, 0.7560540, -0.7942635, 0.8349257, -1.5492402, 1.5503175
2: -0.7846508, 0.9703988, -0.8669555, 1.0457711, -1.8304219, 1.8373544
3: -0.7481736, 0.8280708, -0.8666871, 0.9015115, -1.6496851, 1.6947578
4: -0.9105437, 0.8363509, -1.0180327, 0.9153845, -1.8259282, 1.8543836
5: -0.8351782, 1.0554485, -0.9107673, 1.1319150, -1.9670932, 1.9662158
6: -0.8740258, 1.0229273, -0.9470892, 1.0824500, -1.9564757, 1.9700165
7: -0.7873307, 0.8565540, -0.8841677, 0.9408159, -1.7281467, 1.7407217
8: -1.2738934, 2.3698075, -1.4160151, 2.3677726, -3.6416659, 3.7858226
9: -0.8602692, 1.0569360, -0.9418715, 1.1316698, -1.9919389, 1.9988075

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3355696, upper bound: 7.3473733
time: 2.00 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3355696, upper bound: 7.3842035
time: 2.08 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.1255220, 1.3660806, -1.0059509, 1.2904030, -2.4159250, 2.3720317
1: -0.9060792, 0.9365075, -0.8250173, 0.8646664, -1.7707455, 1.7615248
2: -1.0273085, 1.1482164, -0.9066187, 1.0754111, -2.1027195, 2.0548351
3: -1.0645906, 0.9660985, -0.9150787, 0.9258232, -1.9904137, 1.8811772
4: -1.2051921, 1.0382874, -1.0645829, 0.9478711, -2.1530633, 2.1028705
5: -1.0500562, 1.2363605, -0.9442001, 1.1630545, -2.2131107, 2.1805606
6: -1.0882683, 1.2370577, -0.9819947, 1.1180130, -2.2062812, 2.2190523
7: -1.0326028, 1.0727382, -0.9232545, 0.9751702, -2.0077729, 1.9959927
8: -1.6886237, 2.4266267, -1.4835141, 2.3884854, -4.0771093, 3.9101408
9: -1.0521889, 1.2662637, -0.9724936, 1.1668057, -2.2189946, 2.2387574

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3355696, upper bound: 7.3473733
time: 2.87 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3355696, upper bound: 7.3842035
time: 2.10 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3286878, 0.8163460, -0.9574858, 1.2983978, -1.6270857, 1.7738318
1: -0.3081098, 0.3671333, -0.7837427, 0.8294733, -1.1375830, 1.1508759
2: -0.4453541, 0.4715777, -0.8673200, 1.0633845, -1.5087386, 1.3388977
3: -0.2784572, 0.4204646, -0.8709659, 0.8831962, -1.1616535, 1.2914305
4: -0.4331292, 0.3879401, -1.0127301, 0.9230655, -1.3561946, 1.4006703
5: -0.4023151, 0.5572499, -0.9249626, 1.1149309, -1.5172460, 1.4822125
6: -0.4021399, 0.5418121, -0.9794523, 1.1612753, -1.5634152, 1.5212643
7: -0.3516772, 0.3877113, -0.8799917, 0.9520656, -1.3037429, 1.2677029
8: -0.3468599, 2.2045350, -1.4798533, 2.3999052, -2.7467651, 3.6843882
9: -0.5353960, 0.6282471, -0.9453634, 1.1321533, -1.6675493, 1.5736105

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2931625, upper bound: 7.3171781
time: 1.84 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2931625, upper bound: 7.3628395
time: 2.35 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3890590, 0.8700409, -0.9981769, 1.3236507, -1.7127097, 1.8682177
1: -0.3556631, 0.4050535, -0.8127098, 0.8580031, -1.2136662, 1.2177633
2: -0.4801306, 0.5281917, -0.9048375, 1.0917706, -1.5719012, 1.4330292
3: -0.3158977, 0.4725525, -0.9173427, 0.9064931, -1.2223908, 1.3898952
4: -0.4699343, 0.4326759, -1.0566709, 0.9541031, -1.4240373, 1.4893467
5: -0.4523340, 0.6063890, -0.9564137, 1.1445782, -1.5969123, 1.5628028
6: -0.4516869, 0.6084024, -1.0120931, 1.1946247, -1.6463115, 1.6204954
7: -0.3917261, 0.4398189, -0.9173003, 0.9845363, -1.3762624, 1.3571193
8: -0.4366053, 2.2364864, -1.5437708, 2.4171131, -2.8537183, 3.7802572
9: -0.5632542, 0.6795791, -0.9742379, 1.1652095, -1.7284638, 1.6538171

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2931625, upper bound: 7.3171781
time: 1.94 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2931625, upper bound: 7.3628395
time: 2.33 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3410109, 0.8279513, -1.3223025, 1.5057881, -1.8467990, 2.1502538
1: -0.3179950, 0.3759453, -1.0423281, 1.0561900, -1.3741850, 1.4182734
2: -0.4539731, 0.4824525, -1.1888404, 1.2887920, -1.7427652, 1.6712929
3: -0.2880446, 0.4304986, -1.2654728, 1.0731893, -1.3612338, 1.6959714
4: -0.4419237, 0.3987080, -1.4072965, 1.1772300, -1.6191537, 1.8060045
5: -0.4123716, 0.5712485, -1.2257917, 1.3472532, -1.7596248, 1.7970402
6: -0.4133470, 0.5534695, -1.2659802, 1.4397776, -1.8531246, 1.8194498
7: -0.3601148, 0.3986252, -1.1956025, 1.2208200, -1.5809348, 1.5942278
8: -0.3672892, 2.2166140, -2.0079315, 2.4843440, -2.8516333, 4.2245455
9: -0.5422921, 0.6401391, -1.1936860, 1.4033310, -1.9456232, 1.8338251

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2931131, upper bound: 7.3171272
time: 1.98 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2931131, upper bound: 7.3628395
time: 2.02 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.4030014, 0.8812332, -1.3767037, 1.5379885, -1.9409900, 2.2579370
1: -0.3661677, 0.4137633, -1.0820817, 1.0899543, -1.4561219, 1.4958450
2: -0.4896776, 0.5405312, -1.2386416, 1.3227693, -1.8124468, 1.7791728
3: -0.3254603, 0.4840786, -1.3246081, 1.1025593, -1.4280196, 1.8086867
4: -0.4785597, 0.4437129, -1.4675936, 1.2150799, -1.6936396, 1.9113064
5: -0.4646896, 0.6198086, -1.2732625, 1.3832040, -1.8478935, 1.8930711
6: -0.4637611, 0.6227113, -1.3085734, 1.4829774, -1.9467385, 1.9312847
7: -0.4014172, 0.4503762, -1.2434921, 1.2610633, -1.6624806, 1.6938683
8: -0.4592841, 2.2490435, -2.0850406, 2.5062182, -2.9655023, 4.3340840
9: -0.5705932, 0.6916361, -1.2325195, 1.4436543, -2.0142474, 1.9241556

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2931131, upper bound: 7.3171272
time: 1.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2931131, upper bound: 7.3628395
time: 2.01 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.5598723, 1.0049210, -1.0350870, 1.3401879, -1.9000602, 2.0400081
1: -0.4788301, 0.5326676, -0.8406290, 0.8843108, -1.3631408, 1.3732967
2: -0.5902750, 0.6866012, -0.9385306, 1.1127074, -1.7029824, 1.6251318
3: -0.4348123, 0.6281958, -0.9599273, 0.9263342, -1.3611465, 1.5881231
4: -0.5913262, 0.5981910, -1.0973109, 0.9820046, -1.5733309, 1.6955019
5: -0.5994302, 0.7858242, -0.9856218, 1.1760826, -1.7755128, 1.7714460
6: -0.6095599, 0.7987146, -1.0412977, 1.2218271, -1.8313870, 1.8400123
7: -0.5273271, 0.5858058, -0.9503335, 1.0126858, -1.5400128, 1.5361392
8: -0.7557518, 2.3100932, -1.6005189, 2.4113398, -3.1670916, 3.9106121
9: -0.6526526, 0.8256094, -0.9972757, 1.1943707, -1.8470234, 1.8228850

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3376189, upper bound: 7.3376975
time: 2.21 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3376189, upper bound: 7.3800356
time: 2.27 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.6779746, 1.0995460, -1.0824159, 1.3660989, -2.0440736, 2.1819620
1: -0.5708556, 0.6301671, -0.8727590, 0.9139036, -1.4847591, 1.5029261
2: -0.6534929, 0.8224619, -0.9784722, 1.1422092, -1.7957020, 1.8009341
3: -0.5442775, 0.7099512, -1.0097657, 0.9506909, -1.4949684, 1.7197169
4: -0.7141368, 0.6927486, -1.1469817, 1.0147460, -1.7288828, 1.8397303
5: -0.7026132, 0.8759805, -1.0244795, 1.2057843, -1.9083976, 1.9004600
6: -0.7229297, 0.9026353, -1.0792475, 1.2567611, -1.9796908, 1.9818828
7: -0.6198274, 0.7009338, -0.9893214, 1.0475304, -1.6673578, 1.6902552
8: -0.9556762, 2.3461504, -1.6690122, 2.4287024, -3.3843784, 4.0151625
9: -0.7332206, 0.9234602, -1.0284743, 1.2298329, -1.9630535, 1.9519346

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3376189, upper bound: 7.3376975
time: 1.90 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3376189, upper bound: 7.3800356
time: 3.93 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5788376, 1.0215440, -1.4111431, 1.5520642, -2.1309018, 2.4326870
1: -0.4937651, 0.5490422, -1.1075394, 1.1132056, -1.6069707, 1.6565816
2: -0.6017936, 0.7091085, -1.2691706, 1.3400786, -1.9418721, 1.9782791
3: -0.4510809, 0.6426243, -1.3631945, 1.1201264, -1.5712073, 2.0058188
4: -0.6094877, 0.6142873, -1.5062459, 1.2383488, -1.8478365, 2.1205332
5: -0.6171273, 0.8032970, -1.3025365, 1.4114581, -2.0285854, 2.1058335
6: -0.6297170, 0.8155540, -1.3341274, 1.5073619, -2.1370788, 2.1496816
7: -0.5421941, 0.6040002, -1.2738605, 1.2877204, -1.8299146, 1.8778607
8: -0.7896277, 2.3218803, -2.1322112, 2.4985499, -3.2881775, 4.4540915
9: -0.6652183, 0.8431377, -1.2546415, 1.4679236, -2.1331420, 2.0977793

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3376076, upper bound: 7.3376076
time: 2.13 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3376076, upper bound: 7.3800356
time: 2.11 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.7048448, 1.1199613, -1.4663585, 1.5847552, -2.2895999, 2.5863199
1: -0.5929368, 0.6482201, -1.1476040, 1.1488832, -1.7418200, 1.7958241
2: -0.6728213, 0.8469897, -1.3197050, 1.3743122, -2.0471334, 2.1666946
3: -0.5737160, 0.7267138, -1.4238329, 1.1495047, -1.7232206, 2.1505468
4: -0.7445225, 0.7148474, -1.5674919, 1.2765288, -2.0210514, 2.2823393
5: -0.7231868, 0.9015878, -1.3502822, 1.4481626, -2.1713493, 2.2518702
6: -0.7481452, 0.9234438, -1.3796824, 1.5517769, -2.2999220, 2.3031263
7: -0.6447636, 0.7245263, -1.3216017, 1.3325572, -1.9773209, 2.0461280
8: -1.0058606, 2.3593938, -2.2095973, 2.5205402, -3.5264008, 4.5689912
9: -0.7524616, 0.9444844, -1.2941835, 1.5086054, -2.2610669, 2.2386680

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3376076, upper bound: 7.3376076
time: 2.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3376076, upper bound: 7.3800356
time: 1.74 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4769457, 0.9419547, -0.7004451, 1.0875987, -1.5645443, 1.6423998
1: -0.4233640, 0.4681646, -0.6004770, 0.6564432, -1.0798073, 1.0686415
2: -0.5445726, 0.6094370, -0.6551537, 0.8420749, -1.3866475, 1.2645907
3: -0.3746712, 0.5599605, -0.5694261, 0.7436039, -1.1182752, 1.1293867
4: -0.5313589, 0.5201499, -0.7407029, 0.7096941, -1.2410530, 1.2608528
5: -0.5309169, 0.6849322, -0.7133935, 0.9138858, -1.4448028, 1.3983257
6: -0.5343598, 0.6961375, -0.7277776, 0.8724749, -1.4068346, 1.4239151
7: -0.4631944, 0.5175283, -0.6504713, 0.7206440, -1.1838384, 1.1679995
8: -0.5742691, 2.3067970, -0.9634861, 2.2923608, -2.8666298, 3.2702832
9: -0.6152157, 0.7579942, -0.7486471, 0.9334549, -1.5486705, 1.5066413

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3599337, upper bound: 7.3981433
time: 2.45 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3599337, upper bound: 7.3981433
time: 2.21 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.6499211, 1.1044664, -0.7004451, 1.0875987, -1.7375197, 1.8049116
1: -0.5398597, 0.5951070, -0.6004770, 0.6564432, -1.1963029, 1.1955839
2: -0.6471159, 0.8075863, -0.6551537, 0.8420749, -1.4891908, 1.4627399
3: -0.5128810, 0.6812047, -0.5694261, 0.7436039, -1.2564850, 1.2506309
4: -0.6870871, 0.6650043, -0.7407029, 0.7096941, -1.3967812, 1.4057071
5: -0.6939795, 0.8071368, -0.7133935, 0.9138858, -1.6078653, 1.5205303
6: -0.7189497, 0.9341853, -0.7277776, 0.8724749, -1.5914246, 1.6619629
7: -0.5868236, 0.6870992, -0.6504713, 0.7206440, -1.3074677, 1.3375704
8: -0.9374655, 2.4342887, -0.9634861, 2.2923608, -3.2298265, 3.3977747
9: -0.7278386, 0.9038346, -0.7486471, 0.9334549, -1.6612935, 1.6524817

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3599337, upper bound: 7.3981433
time: 2.53 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3599337, upper bound: 7.3981433
time: 2.00 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4907718, 0.9530501, -1.0215731, 1.2971675, -1.7879393, 1.9746232
1: -0.4332009, 0.4781317, -0.8367071, 0.8758706, -1.3090715, 1.3148389
2: -0.5540079, 0.6218978, -0.9207725, 1.0848358, -1.6388437, 1.5426702
3: -0.3845750, 0.5719993, -0.9336939, 0.9369257, -1.3215008, 1.5056932
4: -0.5414371, 0.5335393, -1.0838915, 0.9583921, -1.4998293, 1.6174308
5: -0.5426317, 0.6981490, -0.9574349, 1.1653671, -1.7079989, 1.6555839
6: -0.5471056, 0.7093716, -0.9947326, 1.1261113, -1.6732168, 1.7041042
7: -0.4744697, 0.5283908, -0.9380343, 0.9893066, -1.4637764, 1.4664251
8: -0.5979142, 2.3184896, -1.4974959, 2.3889756, -2.9868898, 3.8159854
9: -0.6233261, 0.7700409, -0.9833414, 1.1782351, -1.8015611, 1.7533822

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3598566, upper bound: 7.3970817
time: 1.92 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3598566, upper bound: 7.3970817
time: 2.35 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.6727024, 1.1203536, -1.0215731, 1.2971675, -1.9698700, 2.1419268
1: -0.5569439, 0.6117700, -0.8367071, 0.8758706, -1.4328145, 1.4484771
2: -0.6600581, 0.8311144, -0.9207725, 1.0848358, -1.7448939, 1.7518868
3: -0.5352215, 0.6956028, -0.9336939, 0.9369257, -1.4721472, 1.6292967
4: -0.7102271, 0.6834766, -1.0838915, 0.9583921, -1.6686192, 1.7673681
5: -0.7123961, 0.8262386, -0.9574349, 1.1653671, -1.8777633, 1.7836735
6: -0.7393851, 0.9528290, -0.9947326, 1.1261113, -1.8654964, 1.9475616
7: -0.6043996, 0.7071620, -0.9380343, 0.9893066, -1.5937061, 1.6451963
8: -0.9760864, 2.4448063, -1.4974959, 2.3889756, -3.3650620, 3.9423022
9: -0.7434005, 0.9217475, -0.9833414, 1.1782351, -1.9216355, 1.9050889

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3598566, upper bound: 7.3970817
time: 1.91 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3598566, upper bound: 7.3970817
time: 1.85 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.3284142, 1.4594016, -0.6869816, 1.0737057, -2.4021199, 2.1463828
1: -1.0545180, 1.0799904, -0.5900174, 0.6488885, -1.7034066, 1.6700078
2: -1.2082275, 1.2598070, -0.6449015, 0.8285521, -2.0367796, 1.9047085
3: -1.2893515, 1.0813105, -0.5559654, 0.7349299, -2.0242813, 1.6372759
4: -1.4309726, 1.1775802, -0.7262413, 0.6986863, -2.1296589, 1.9038215
5: -1.2282274, 1.3754160, -0.7022441, 0.9079163, -2.1361437, 2.0776601
6: -1.2583177, 1.3782438, -0.7137868, 0.8585736, -2.1168914, 2.0920305
7: -1.2139376, 1.2238892, -0.6392779, 0.7081927, -1.9221303, 1.8631670
8: -1.9772065, 2.4647288, -0.9391929, 2.2700279, -4.2472343, 3.4039211
9: -1.1943930, 1.4141235, -0.7372773, 0.9225751, -2.1169682, 2.1514008

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3104361, upper bound: 7.3270851
time: 2.22 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3109984, upper bound: 7.3206757
time: 2.41 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.6239269, 1.6304362, -0.7160797, 1.0951943, -2.7191212, 2.3465159
1: -1.2729762, 1.2729547, -0.6136514, 0.6688226, -1.9417988, 1.8866061
2: -1.4835610, 1.4374479, -0.6660560, 0.8546796, -2.3382406, 2.1035039
3: -1.6221051, 1.2320062, -0.5874828, 0.7555386, -2.3776436, 1.8194890
4: -1.7679971, 1.3848754, -0.7581221, 0.7224475, -2.4904447, 2.1429975
5: -1.4881897, 1.5612482, -0.7248921, 0.9340088, -2.4221985, 2.2861404
6: -1.5034410, 1.6161917, -0.7411835, 0.8813530, -2.3847938, 2.3573751
7: -1.4760374, 1.4744046, -0.6653718, 0.7339616, -2.2099991, 2.1397762
8: -2.4066362, 2.5345998, -0.9935354, 2.2886946, -4.6953306, 3.5281353
9: -1.4040254, 1.6294414, -0.7595967, 0.9451481, -2.3491726, 2.3890381

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3104361, upper bound: 7.3270851
time: 2.28 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3109984, upper bound: 7.3206789
time: 2.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.3850341, 1.4926423, -0.9960468, 1.2774868, -2.6625209, 2.4886889
1: -1.0952976, 1.1173284, -0.8181163, 0.8602874, -1.9555850, 1.9354447
2: -1.2593641, 1.2950931, -0.8971619, 1.0657426, -2.3251066, 2.1922550
3: -1.3511548, 1.1104448, -0.9057524, 0.9231055, -2.2742603, 2.0161972
4: -1.4948516, 1.2174461, -1.0564402, 0.9384950, -2.4333467, 2.2738862
5: -1.2780269, 1.4137000, -0.9363781, 1.1523209, -2.4303479, 2.3500781
6: -1.3063306, 1.4221905, -0.9714995, 1.1026721, -2.4090028, 2.3936901
7: -1.2633181, 1.2680405, -0.9158325, 0.9683086, -2.2316265, 2.1838732
8: -2.0580149, 2.4820716, -1.4567235, 2.3648071, -4.4228220, 3.9387951
9: -1.2342318, 1.4554439, -0.9644945, 1.1570121, -2.3912439, 2.4199383

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3104006, upper bound: 7.3266576
time: 2.16 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3108510, upper bound: 7.3196253
time: 2.59 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.6778631, 1.6641020, -1.0426835, 1.3042904, -2.9821534, 2.7067854
1: -1.3121152, 1.3105890, -0.8513166, 0.8902530, -2.2023683, 2.1619055
2: -1.5354978, 1.4712522, -0.9382916, 1.0956401, -2.6311378, 2.4095438
3: -1.6821398, 1.2602084, -0.9557061, 0.9479759, -2.6301155, 2.2159145
4: -1.8292382, 1.4229023, -1.1054821, 0.9726012, -2.8018394, 2.5283844
5: -1.5371804, 1.5981417, -0.9740822, 1.1827304, -2.7199109, 2.5722239
6: -1.5491534, 1.6603806, -1.0106266, 1.1391006, -2.6882539, 2.6710072
7: -1.5238678, 1.5192152, -0.9550424, 1.0043480, -2.5282159, 2.4742575
8: -2.4878652, 2.5532260, -1.5266773, 2.3854640, -4.8733292, 4.0799031
9: -1.4437163, 1.6691215, -0.9952903, 1.1938802, -2.6375966, 2.6644118

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3104006, upper bound: 7.3266576
time: 2.26 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3108510, upper bound: 7.3196253
time: 1.78 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4496224, 0.9071100, -1.0119259, 1.3237333, -1.7733557, 1.9190359
1: -0.4035218, 0.4500773, -0.8246219, 0.8696791, -1.2732009, 1.2746992
2: -0.5259115, 0.5784683, -0.9181300, 1.0969841, -1.6228957, 1.4965982
3: -0.3563831, 0.5368321, -0.9355083, 0.9173121, -1.2736952, 1.4723403
4: -0.5108311, 0.4937452, -1.0743500, 0.9639280, -1.4747591, 1.5680952
5: -0.5066083, 0.6788969, -0.9662351, 1.1538162, -1.6604245, 1.6451321
6: -0.5031930, 0.6584223, -1.0203037, 1.1960027, -1.6991957, 1.6787260
7: -0.4422124, 0.4889604, -0.9315070, 0.9966429, -1.4388554, 1.4204674
8: -0.5251525, 2.2723305, -1.5540308, 2.4022627, -2.9274151, 3.8263612
9: -0.5944222, 0.7344696, -0.9824901, 1.1754727, -1.7698948, 1.7169597

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3349509, upper bound: 7.3174297
time: 2.63 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3349510, upper bound: 7.3917543
time: 2.53 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.5065281, 0.9581416, -1.0562713, 1.3495121, -1.8560402, 2.0144129
1: -0.4450177, 0.4919754, -0.8563346, 0.8990360, -1.3440537, 1.3483100
2: -0.5612206, 0.6353639, -0.9574074, 1.1259412, -1.6871617, 1.5927713
3: -0.3939156, 0.5910597, -0.9839084, 0.9409195, -1.3348351, 1.5749681
4: -0.5524921, 0.5496833, -1.1222790, 0.9965307, -1.5490228, 1.6719624
5: -0.5545800, 0.7240157, -1.0022781, 1.1834062, -1.7379863, 1.7262938
6: -0.5569334, 0.7180506, -1.0578439, 1.2303385, -1.7872719, 1.7758945
7: -0.4882692, 0.5408727, -0.9696338, 1.0304637, -1.5187328, 1.5105065
8: -0.6208160, 2.3013420, -1.6208954, 2.4195447, -3.0403607, 3.9222374
9: -0.6275903, 0.7824947, -1.0126233, 1.2101600, -1.8377503, 1.7951180

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3349509, upper bound: 7.3174297
time: 2.35 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3349510, upper bound: 7.3917735
time: 2.15 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4652423, 0.9198347, -1.3494124, 1.5155402, -1.9807825, 2.2692471
1: -0.4148593, 0.4617111, -1.0624912, 1.0750538, -1.4899131, 1.5242022
2: -0.5364776, 0.5928230, -1.2126187, 1.3026257, -1.8391032, 1.8054417
3: -0.3675578, 0.5510132, -1.2957599, 1.0912132, -1.4587710, 1.8467731
4: -0.5218996, 0.5092793, -1.4383701, 1.1953039, -1.7172036, 1.9476494
5: -0.5198647, 0.6941161, -1.2487419, 1.3651593, -1.8850240, 1.9428580
6: -0.5177299, 0.6735063, -1.2854100, 1.4535230, -1.9712529, 1.9589163
7: -0.4549867, 0.5016037, -1.2198241, 1.2408649, -1.6958516, 1.7214278
8: -0.5521200, 2.2843013, -2.0374374, 2.4826646, -3.0347846, 4.3217387
9: -0.6033519, 0.7482266, -1.2128118, 1.4215566, -2.0249085, 1.9610384

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3349047, upper bound: 7.3174087
time: 6.09 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3349048, upper bound: 7.3917543
time: 2.71 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.5227454, 0.9709175, -1.4038008, 1.5480751, -2.0708206, 2.3747184
1: -0.4558104, 0.5043885, -1.1022491, 1.1090355, -1.5648459, 1.6066376
2: -0.5715485, 0.6518309, -1.2623643, 1.3365998, -1.9081483, 1.9141952
3: -0.4048449, 0.6047130, -1.3546741, 1.1204920, -1.5253369, 1.9593871
4: -0.5655165, 0.5642238, -1.4985374, 1.2333467, -1.7988632, 2.0627613
5: -0.5675531, 0.7400358, -1.2963989, 1.4011184, -1.9686716, 2.0364347
6: -0.5719452, 0.7324939, -1.3297431, 1.4971445, -2.0690897, 2.0622370
7: -0.5012094, 0.5554544, -1.2673635, 1.2833574, -1.7845668, 1.8228179
8: -0.6486216, 2.3133564, -2.1145363, 2.5045028, -3.1531243, 4.4278927
9: -0.6380969, 0.7957622, -1.2515483, 1.4619129, -2.1000099, 2.0473106

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3349047, upper bound: 7.3174087
time: 2.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3349048, upper bound: 7.3917735
time: 2.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.8577917, 1.2108977, -1.1153831, 1.3745146, -2.2323062, 2.3262808
1: -0.7183729, 0.7603508, -0.8976265, 0.9372163, -1.6555892, 1.6579773
2: -0.7969975, 0.9712486, -1.0076764, 1.1579064, -1.9549040, 1.9789250
3: -0.7597868, 0.8352802, -1.0487947, 0.9701715, -1.7299583, 1.8840749
4: -0.9208673, 0.8388460, -1.1861503, 1.0369759, -1.9578432, 2.0249963
5: -0.8439783, 1.0523629, -1.0517123, 1.2254076, -2.0693860, 2.1040752
6: -0.8836071, 1.0254371, -1.1018265, 1.2717221, -2.1553292, 2.1272635
7: -0.7938092, 0.8653019, -1.0197512, 1.0730565, -1.8668656, 1.8850532
8: -1.2745475, 2.3940873, -1.7056198, 2.4159129, -3.6904602, 4.0997071
9: -0.8687246, 1.0624551, -1.0493249, 1.2535532, -2.1222777, 2.1117799

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3800475, upper bound: 7.3404959
time: 3.16 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3800475, upper bound: 7.4113707
time: 3.42 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.0349026, 1.3186668, -1.1644642, 1.4009132, -2.4358158, 2.4831309
1: -0.8425653, 0.8830842, -0.9314606, 0.9666647, -1.8092301, 1.8145448
2: -0.9542188, 1.0929799, -1.0484197, 1.1879632, -2.1421821, 2.1413996
3: -0.9659823, 0.9327859, -1.0999174, 0.9953585, -1.9613409, 2.0327034
4: -1.1100891, 0.9748161, -1.2370713, 1.0703881, -2.1804771, 2.2118874
5: -0.9784657, 1.1738151, -1.0932713, 1.2549778, -2.2334435, 2.2670865
6: -1.0263143, 1.1686102, -1.1402855, 1.3086507, -2.3349650, 2.3088956
7: -0.9545902, 1.0097282, -1.0612690, 1.1080968, -2.0626869, 2.0709972
8: -1.5477071, 2.4373980, -1.7764766, 2.4334002, -3.9811072, 4.2138748
9: -0.9971094, 1.1986709, -1.0830531, 1.2889495, -2.2860589, 2.2817240

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3800475, upper bound: 7.3404959
time: 2.44 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3800475, upper bound: 7.4113707
time: 2.47 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.8974151, 1.2348149, -1.4673250, 1.5769132, -2.4743283, 2.7021399
1: -0.7449700, 0.7868603, -1.1493198, 1.1521120, -1.8970821, 1.9361801
2: -0.8288338, 0.9991621, -1.3199941, 1.3706832, -2.1995170, 2.3191562
3: -0.8048531, 0.8577217, -1.4272091, 1.1525716, -1.9574246, 2.2849307
4: -0.9612576, 0.8689695, -1.5710475, 1.2761467, -2.2374043, 2.4400170
5: -0.8733776, 1.0832773, -1.3507246, 1.4480199, -2.3213975, 2.4340019
6: -0.9143056, 1.0560765, -1.3798844, 1.5443518, -2.4586575, 2.4359608
7: -0.8271830, 0.8967884, -1.3235064, 1.3358330, -2.1630158, 2.2202947
8: -1.3362455, 2.4079187, -2.2008429, 2.5004401, -3.8366857, 4.6087618
9: -0.8977774, 1.0913345, -1.2938143, 1.5071425, -2.4049201, 2.3851488

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 42

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3800355, upper bound: 7.3403809
time: 2.94 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3800356, upper bound: 7.4113603
time: 2.27 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.0800669, 1.3435661, -1.5226145, 1.6095536, -2.6896205, 2.8661804
1: -0.8742276, 0.9118930, -1.1894145, 1.1881229, -2.0623505, 2.1013074
2: -0.9945601, 1.1211103, -1.3708203, 1.4048904, -2.3994505, 2.4919305
3: -1.0164238, 0.9555094, -1.4880178, 1.1820947, -2.1985185, 2.4435272
4: -1.1599014, 1.0071852, -1.6322595, 1.3142085, -2.4741099, 2.6394448
5: -1.0160652, 1.2041125, -1.3984534, 1.4848802, -2.5009456, 2.6025658
6: -1.0638008, 1.2028298, -1.4252918, 1.5891541, -2.6529551, 2.6281216
7: -0.9937143, 1.0435827, -1.3712151, 1.3814274, -2.3751416, 2.4147978
8: -1.6163027, 2.4521811, -2.2784336, 2.5225585, -4.1388612, 4.7306147
9: -1.0283446, 1.2332532, -1.3335241, 1.5477457, -2.5760903, 2.5667772

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3800355, upper bound: 7.3096446
time: 22.74 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3800356, upper bound: 7.4113603
time: 2.55 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 27.21 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2904377, upper bound: 7.3607060
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2904377, upper bound: 7.3607060
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2904377, upper bound: 7.3607060
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2904377, upper bound: 7.3607060
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2902558, upper bound: 7.3600370
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2902558, upper bound: 7.3600370
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2902558, upper bound: 7.3600370
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2902558, upper bound: 7.3600370
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3356866, upper bound: 7.3482706
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3356866, upper bound: 7.3849593
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3356866, upper bound: 7.3482706
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3356866, upper bound: 7.3849593
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3355696, upper bound: 7.3473733
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3355696, upper bound: 7.3842035
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3355696, upper bound: 7.3473733
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3355696, upper bound: 7.3842035
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2931625, upper bound: 7.3171781
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2931625, upper bound: 7.3628395
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2931625, upper bound: 7.3171781
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2931625, upper bound: 7.3628395
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2931131, upper bound: 7.3171272
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2931131, upper bound: 7.3628395
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2931131, upper bound: 7.3171272
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.2931131, upper bound: 7.3628395
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3376189, upper bound: 7.3376975
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3376189, upper bound: 7.3800356
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3376189, upper bound: 7.3376975
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3376189, upper bound: 7.3800356
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3376076, upper bound: 7.3376076
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3376076, upper bound: 7.3800356
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3376076, upper bound: 7.3376076
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3376076, upper bound: 7.3800356
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3599337, upper bound: 7.3981433
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3599337, upper bound: 7.3981433
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3599337, upper bound: 7.3981433
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3599337, upper bound: 7.3981433
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3598566, upper bound: 7.3970817
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3598566, upper bound: 7.3970817
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3598566, upper bound: 7.3970817
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3598566, upper bound: 7.3970817
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3104361, upper bound: 7.3270851
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3109984, upper bound: 7.3206757
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3104361, upper bound: 7.3270851
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3109984, upper bound: 7.3206789
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3104006, upper bound: 7.3266576
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3108510, upper bound: 7.3196253
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3104006, upper bound: 7.3266576
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3108510, upper bound: 7.3196253
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3349509, upper bound: 7.3174297
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3349510, upper bound: 7.3917543
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3349509, upper bound: 7.3174297
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3349510, upper bound: 7.3917735
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3349047, upper bound: 7.3174087
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3349048, upper bound: 7.3917543
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3349047, upper bound: 7.3174087
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3349048, upper bound: 7.3917735
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3800475, upper bound: 7.3404959
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3800475, upper bound: 7.4113707
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3800475, upper bound: 7.3404959
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3800475, upper bound: 7.4113707
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3800355, upper bound: 7.3403809
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3800356, upper bound: 7.4113603
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3800355, upper bound: 7.3096446
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.21
Output dim: 8, lower bound: -7.3800356, upper bound: 7.4113603

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3552179, 0.8484029, -0.5469491, 0.9754868, -1.3307047, 1.3953520
1: -0.3293309, 0.3833984, -0.4779562, 0.5367007, -0.8660316, 0.8613546
2: -0.4607645, 0.4993531, -0.5699401, 0.6776872, -1.1384517, 1.0692933
3: -0.2952593, 0.4415436, -0.4173080, 0.6337188, -0.9289781, 0.8588516
4: -0.4505318, 0.4073985, -0.5784138, 0.5865576, -1.0370893, 0.9858122
5: -0.4245262, 0.5700010, -0.5866563, 0.7970445, -1.2215707, 1.1566573
6: -0.4281837, 0.5776489, -0.5853475, 0.7581998, -1.1863835, 1.1629965
7: -0.3683506, 0.4153537, -0.5219342, 0.5754406, -0.9437912, 0.9372879
8: -0.3891976, 2.2226636, -0.7097298, 2.2365782, -2.6257758, 2.9323936
9: -0.5498999, 0.6511742, -0.6402165, 0.8141180, -1.3640180, 1.2913907

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2215807, upper bound: 7.2844740
time: 2.13 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2177719, upper bound: 7.2836422
time: 2.37 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3686046, 0.8599597, -0.6200979, 1.0334644, -1.4020691, 1.4800576
1: -0.3396806, 0.3915809, -0.5336416, 0.5978763, -0.9375569, 0.9252225
2: -0.4694315, 0.5118505, -0.6079372, 0.7639616, -1.2333931, 1.1197877
3: -0.3038073, 0.4525387, -0.4812834, 0.6857345, -0.9895417, 0.9338221
4: -0.4588899, 0.4173955, -0.6524084, 0.6429884, -1.1018783, 1.0698038
5: -0.4354905, 0.5820025, -0.6522503, 0.8462371, -1.2817277, 1.2342529
6: -0.4391145, 0.5921106, -0.6558796, 0.8189125, -1.2580270, 1.2479901
7: -0.3771923, 0.4260521, -0.5754893, 0.6461445, -1.0233368, 1.0015414
8: -0.4109582, 2.2377505, -0.8249887, 2.2689970, -2.6799552, 3.0627394
9: -0.5576203, 0.6629032, -0.6913602, 0.8745662, -1.4321866, 1.3542633

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2215807, upper bound: 7.2844740
time: 2.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2177719, upper bound: 7.2836422
time: 12.18 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.5004543, 0.9869771, -0.5469491, 0.9754868, -1.4759412, 1.5339262
1: -0.4262029, 0.4674537, -0.4779562, 0.5367007, -0.9629036, 0.9454099
2: -0.5503538, 0.6353014, -0.5699401, 0.6776872, -1.2280409, 1.2052414
3: -0.3962288, 0.5463980, -0.4173080, 0.6337188, -1.0299478, 0.9637060
4: -0.5428556, 0.5278756, -0.5784138, 0.5865576, -1.1294131, 1.1062894
5: -0.5626387, 0.6590369, -0.5866563, 0.7970445, -1.3596833, 1.2456932
6: -0.5713298, 0.8037386, -0.5853475, 0.7581998, -1.3295296, 1.3890861
7: -0.4648051, 0.5378085, -0.5219342, 0.5754406, -1.0402458, 1.0597427
8: -0.6884762, 2.3385019, -0.7097298, 2.2365782, -2.9250546, 3.0482316
9: -0.6255128, 0.7688481, -0.6402165, 0.8141180, -1.4396307, 1.4090645

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2934616, upper bound: 7.3607060
time: 3.36 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2934616, upper bound: 7.3607060
time: 2.16 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.5152417, 0.9992193, -0.6200979, 1.0334644, -1.5487062, 1.6193172
1: -0.4366371, 0.4779362, -0.5336416, 0.5978763, -1.0345134, 1.0115778
2: -0.5601244, 0.6508513, -0.6079372, 0.7639616, -1.3240860, 1.2587885
3: -0.4060107, 0.5597498, -0.4812834, 0.6857345, -1.0917451, 1.0410333
4: -0.5554234, 0.5418258, -0.6524084, 0.6429884, -1.1984118, 1.1942341
5: -0.5747517, 0.6731049, -0.6522503, 0.8462371, -1.4209888, 1.3253553
6: -0.5845352, 0.8181216, -0.6558796, 0.8189125, -1.4034477, 1.4740012
7: -0.4765264, 0.5510909, -0.5754893, 0.6461445, -1.1226709, 1.1265802
8: -0.7143319, 2.3529859, -0.8249887, 2.2689970, -2.9833288, 3.1779747
9: -0.6352750, 0.7814502, -0.6913602, 0.8745662, -1.5098412, 1.4728104

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2934616, upper bound: 7.3607060
time: 2.25 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2934616, upper bound: 7.3607060
time: 2.88 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3678351, 0.8585917, -0.7206373, 1.1073325, -1.4751675, 1.5792291
1: -0.3389192, 0.3910192, -0.6116335, 0.6640330, -1.0029521, 1.0026528
2: -0.4686868, 0.5105377, -0.6702287, 0.8544418, -1.3231286, 1.1807663
3: -0.3037587, 0.4514832, -0.5822470, 0.7495188, -1.0532774, 1.0337303
4: -0.4582800, 0.4171032, -0.7548556, 0.7238810, -1.1821611, 1.1719589
5: -0.4348206, 0.5821437, -0.7259074, 0.9376544, -1.3724750, 1.3080511
6: -0.4381329, 0.5906916, -0.7435095, 0.8926885, -1.3308214, 1.3342011
7: -0.3767230, 0.4248438, -0.6618718, 0.7293131, -1.1060361, 1.0867157
8: -0.4091632, 2.2341895, -1.0089425, 2.2974458, -2.7066090, 3.2431321
9: -0.5563434, 0.6620563, -0.7576196, 0.9485472, -1.5048907, 1.4196759

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 94

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2214514, upper bound: 7.2827628
time: 1.90 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2176400, upper bound: 7.2824132
time: 2.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3815587, 0.8703744, -0.9242233, 1.2430172, -1.6245759, 1.7945977
1: -0.3494726, 0.3995212, -0.7679257, 0.8061182, -1.1555908, 1.1674469
2: -0.4777201, 0.5233046, -0.8321868, 1.0194312, -1.4971514, 1.3554914
3: -0.3124820, 0.4630016, -0.8195488, 0.8802949, -1.1927769, 1.2825505
4: -0.4668269, 0.4275499, -0.9766794, 0.8845255, -1.3513525, 1.4042293
5: -0.4463302, 0.5943931, -0.8831294, 1.0981741, -1.5445044, 1.4775224
6: -0.4498537, 0.6054610, -0.9173689, 1.0534801, -1.5033338, 1.5228299
7: -0.3858255, 0.4358044, -0.8461590, 0.9099938, -1.2958193, 1.2819633
8: -0.4316313, 2.2497997, -1.3541970, 2.3657260, -2.7973573, 3.6039968
9: -0.5643836, 0.6740453, -0.9149953, 1.1004010, -1.6647847, 1.5890406

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2214514, upper bound: 7.2827628
time: 2.21 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2176400, upper bound: 7.2824132
time: 1.73 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.5151089, 0.9985133, -0.7206373, 1.1073325, -1.6224414, 1.7191507
1: -0.4365385, 0.4778539, -0.6116335, 0.6640330, -1.1005714, 1.0894874
2: -0.5598980, 0.6502057, -0.6702287, 0.8544418, -1.4143398, 1.3204343
3: -0.4063151, 0.5594128, -0.5822470, 0.7495188, -1.1558340, 1.1416597
4: -0.5552803, 0.5419478, -0.7548556, 0.7238810, -1.2791612, 1.2968035
5: -0.5745835, 0.6741216, -0.7259074, 0.9376544, -1.5122380, 1.4000291
6: -0.5840961, 0.8173862, -0.7435095, 0.8926885, -1.4767846, 1.5608957
7: -0.4765790, 0.5505884, -0.6618718, 0.7293131, -1.2058921, 1.2124602
8: -0.7138139, 2.3484433, -1.0089425, 2.2974458, -3.0112596, 3.3573856
9: -0.6342426, 0.7811809, -0.7576196, 0.9485472, -1.5827899, 1.5388005

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2932975, upper bound: 7.3600370
time: 2.16 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2932975, upper bound: 7.3600370
time: 2.86 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.5302957, 1.0110347, -0.9242233, 1.2430172, -1.7733129, 1.9352580
1: -0.4472610, 0.4885842, -0.7679257, 0.8061182, -1.2533791, 1.2565098
2: -0.5699807, 0.6661438, -0.8321868, 1.0194312, -1.5894120, 1.4983306
3: -0.4163441, 0.5730846, -0.8195488, 0.8802949, -1.2966390, 1.3926334
4: -0.5681520, 0.5562775, -0.9766794, 0.8845255, -1.4526775, 1.5329568
5: -0.5869976, 0.6885975, -0.8831294, 1.0981741, -1.6851717, 1.5717268
6: -0.5976421, 0.8321300, -0.9173689, 1.0534801, -1.6511222, 1.7494988
7: -0.4885831, 0.5642595, -0.8461590, 0.9099938, -1.3985770, 1.4104185
8: -0.7403262, 2.3633990, -1.3541970, 2.3657260, -3.1060522, 3.7175961
9: -0.6443368, 0.7941024, -0.9149953, 1.1004010, -1.7447379, 1.7090977

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2932975, upper bound: 7.3600370
time: 2.48 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2932975, upper bound: 7.3600370
time: 2.17 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.8213495, 1.1880453, -0.3831510, 0.8464654, -1.6678150, 1.5711963
1: -0.6895053, 0.7324307, -0.3534597, 0.4072016, -1.0967069, 1.0858904
2: -0.7571179, 0.9437820, -0.4695104, 0.5230985, -1.2802165, 1.4132924
3: -0.7087191, 0.8078192, -0.3110040, 0.4745259, -1.1832449, 1.1188232
4: -0.8741490, 0.8098165, -0.4633655, 0.4274715, -1.3016205, 1.2731819
5: -0.8089724, 1.0267889, -0.4456294, 0.6312782, -1.4402505, 1.4724183
6: -0.8449974, 0.9962063, -0.4358237, 0.6032888, -1.4482862, 1.4320300
7: -0.7571899, 0.8276304, -0.3894730, 0.4342014, -1.1913912, 1.2171034
8: -1.2165058, 2.3568790, -0.4307873, 2.1939747, -3.4104805, 2.7876663
9: -0.8346833, 1.0315689, -0.5512367, 0.6761147, -1.5107980, 1.5828056

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3333772, upper bound: 7.3482706
time: 1.96 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3333772, upper bound: 7.3482706
time: 1.82 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.8213495, 1.1880453, -0.5203414, 0.9576836, -1.7790332, 1.7083867
1: -0.6895053, 0.7324307, -0.4574630, 0.5167202, -1.2062254, 1.1898937
2: -0.7571179, 0.9437820, -0.5630240, 0.6544417, -1.4115596, 1.5068060
3: -0.7087191, 0.8078192, -0.4028019, 0.6127278, -1.3214469, 1.2106211
4: -0.8741490, 0.8098165, -0.5646796, 0.5650096, -1.4391586, 1.3744961
5: -0.8089724, 1.0267889, -0.5662748, 0.7626490, -1.5716214, 1.5930637
6: -0.8449974, 0.9962063, -0.5664137, 0.7275816, -1.5725790, 1.5626200
7: -0.7571899, 0.8276304, -0.5053349, 0.5610210, -1.3182108, 1.3329653
8: -1.2165058, 2.3568790, -0.6483524, 2.2619362, -3.4784420, 3.0052314
9: -0.8346833, 1.0315689, -0.6323596, 0.7964390, -1.6311224, 1.6639285

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3333772, upper bound: 7.3849593
time: 1.97 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3333772, upper bound: 7.3849593
time: 1.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.0776675, 1.3412449, -0.3966444, 0.8586739, -1.9363413, 1.7378893
1: -0.8736141, 0.9083317, -0.3640020, 0.4157972, -1.2894113, 1.2723337
2: -0.9861754, 1.1198763, -0.4786158, 0.5356674, -1.5218428, 1.5984920
3: -1.0133747, 0.9436170, -0.3194468, 0.4863495, -1.4997241, 1.2630638
4: -1.1545190, 1.0060172, -0.4720237, 0.4399318, -1.5944507, 1.4780409
5: -1.0091349, 1.2065884, -0.4575397, 0.6435435, -1.6526784, 1.6641281
6: -1.0513515, 1.2012881, -0.4482622, 0.6179244, -1.6692760, 1.6495503
7: -0.9922954, 1.0385352, -0.3988339, 0.4455555, -1.4378510, 1.4373691
8: -1.6183665, 2.4121742, -0.4544714, 2.2087970, -3.8271635, 2.8666456
9: -1.0197767, 1.2312454, -0.5595968, 0.6878370, -1.7076137, 1.7908422

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3333772, upper bound: 7.3482706
time: 1.84 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3333772, upper bound: 7.3482706
time: 1.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.0776675, 1.3412449, -0.5381970, 0.9742446, -2.0519121, 1.8794420
1: -0.8736141, 0.9083317, -0.4717688, 0.5325652, -1.4061792, 1.3801005
2: -0.9861754, 1.1198763, -0.5737513, 0.6767949, -1.6629703, 1.6936276
3: -1.0133747, 0.9436170, -0.4176638, 0.6267221, -1.6400968, 1.3612808
4: -1.1545190, 1.0060172, -0.5821297, 0.5800956, -1.7346146, 1.5881469
5: -1.0091349, 1.2065884, -0.5833294, 0.7779570, -1.7870920, 1.7899178
6: -1.0513515, 1.2012881, -0.5855519, 0.7451302, -1.7964817, 1.7868400
7: -0.9922954, 1.0385352, -0.5187669, 0.5786547, -1.5709500, 1.5573022
8: -1.6183665, 2.4121742, -0.6828876, 2.2769434, -3.8953099, 3.0950618
9: -1.0197767, 1.2312454, -0.6448538, 0.8136253, -1.8334020, 1.8760992

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3333772, upper bound: 7.3849593
time: 1.81 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3333772, upper bound: 7.3849593
time: 2.20 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.8546511, 1.2104809, -0.5118766, 0.9597982, -1.8144493, 1.7223575
1: -0.7143145, 0.7560540, -0.4489578, 0.4966556, -1.2109702, 1.2050118
2: -0.7846508, 0.9703988, -0.5517112, 0.6398275, -1.4244783, 1.5221100
3: -0.7481736, 0.8280708, -0.3938155, 0.5923987, -1.3405724, 1.2218863
4: -0.9105437, 0.8363509, -0.5510662, 0.5527905, -1.4633342, 1.3874171
5: -0.8351782, 1.0554485, -0.5528636, 0.7478326, -1.5830108, 1.6083121
6: -0.8740258, 1.0229273, -0.5556728, 0.7297044, -1.6037302, 1.5786002
7: -0.7873307, 0.8565540, -0.4882433, 0.5407324, -1.3280631, 1.3447974
8: -1.2738934, 2.3698075, -0.6551397, 2.2535834, -3.5274768, 3.0249472
9: -0.8602692, 1.0569360, -0.6197333, 0.7826464, -1.6429156, 1.6766694

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 42

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3332642, upper bound: 7.3473733
time: 1.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3332642, upper bound: 7.3473733
time: 1.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.8546511, 1.2104809, -0.7142990, 1.1040964, -1.9587475, 1.9247799
1: -0.7143145, 0.7560540, -0.6083534, 0.6611443, -1.3754587, 1.3644073
2: -0.7846508, 0.9703988, -0.6751815, 0.8520179, -1.6366687, 1.6455803
3: -0.7481736, 0.8280708, -0.5844821, 0.7468508, -1.4950244, 1.4125528
4: -0.9105437, 0.8363509, -0.7620696, 0.7186595, -1.6292032, 1.5984205
5: -0.8351782, 1.0554485, -0.7247425, 0.9228042, -1.7579824, 1.7801911
6: -0.8740258, 1.0229273, -0.7451391, 0.8879086, -1.7619343, 1.7680664
7: -0.7873307, 0.8565540, -0.6602376, 0.7336480, -1.5209787, 1.5167916
8: -1.2738934, 2.3698075, -0.9925009, 2.3269584, -3.6008518, 3.3623085
9: -0.8602692, 1.0569360, -0.7614726, 0.9480184, -1.8082876, 1.8184086

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3332642, upper bound: 7.3842035
time: 1.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3332642, upper bound: 7.3842035
time: 1.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.1255220, 1.3660806, -0.5283585, 0.9741951, -2.0997171, 1.8944392
1: -0.9060792, 0.9365075, -0.4603655, 0.5100121, -1.4160913, 1.3968730
2: -1.0273085, 1.1482164, -0.5627244, 0.6572570, -1.6845655, 1.7109407
3: -1.0645906, 0.9660985, -0.4046300, 0.6068757, -1.6714664, 1.3707286
4: -1.2051921, 1.0382874, -0.5645781, 0.5677439, -1.7729360, 1.6028655
5: -1.0500562, 1.2363605, -0.5665471, 0.7630433, -1.8130996, 1.8029077
6: -1.0882683, 1.2370577, -0.5715892, 0.7453480, -1.8336163, 1.8086469
7: -1.0326028, 1.0727382, -0.5014941, 0.5560901, -1.5886929, 1.5742322
8: -1.6886237, 2.4266267, -0.6843897, 2.2695906, -3.9582143, 3.1110163
9: -1.0521889, 1.2662637, -0.6321476, 0.7964820, -1.8486708, 1.8984113

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3332642, upper bound: 7.3473733
time: 2.35 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3332642, upper bound: 7.3473733
time: 2.36 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.1255220, 1.3660806, -0.7447640, 1.1263869, -2.2519088, 2.1108446
1: -0.9060792, 0.9365075, -0.6332210, 0.6821379, -1.5882170, 1.5697285
2: -1.0273085, 1.1482164, -0.6979166, 0.8791463, -1.9064548, 1.8461330
3: -1.0645906, 0.9660985, -0.6180944, 0.7685222, -1.8331127, 1.5841930
4: -1.2051921, 1.0382874, -0.7959818, 0.7433320, -1.9485240, 1.8342693
5: -1.0500562, 1.2363605, -0.7489616, 0.9496105, -1.9996667, 1.9853221
6: -1.0882683, 1.2370577, -0.7738609, 0.9117023, -1.9999706, 2.0109186
7: -1.0326028, 1.0727382, -0.6879624, 0.7607861, -1.7933888, 1.7607006
8: -1.6886237, 2.4266267, -1.0492041, 2.3456683, -4.0342922, 3.4758308
9: -1.0521889, 1.2662637, -0.7849712, 0.9714479, -2.0236368, 2.0512350

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3332642, upper bound: 7.3842035
time: 2.30 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3332642, upper bound: 7.3842035
time: 2.34 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3286878, 0.8163460, -0.5408846, 1.0073111, -1.3359990, 1.3572307
1: -0.3081098, 0.3671333, -0.4591901, 0.5110075, -0.8191172, 0.8263234
2: -0.4453541, 0.4715777, -0.5729125, 0.6782074, -1.1235615, 1.0444902
3: -0.2784572, 0.4204646, -0.4224918, 0.5961804, -0.8746377, 0.8429564
4: -0.4331292, 0.3879401, -0.5752495, 0.5769160, -1.0100452, 0.9631896
5: -0.4023151, 0.5572499, -0.5934584, 0.7441016, -1.1464167, 1.1507082
6: -0.4021399, 0.5418121, -0.6062131, 0.8349524, -1.2370923, 1.1480252
7: -0.3516772, 0.3877113, -0.5037438, 0.5769769, -0.9286541, 0.8914551
8: -0.3468599, 2.2045350, -0.7561051, 2.3165691, -2.6634290, 2.9606400
9: -0.5353960, 0.6282471, -0.6442462, 0.8039651, -1.3393611, 1.2724934

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1897216, upper bound: 7.2024976
time: 2.10 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1829970, upper bound: 7.2031861
time: 1.84 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3286878, 0.8163460, -0.7501806, 1.1640257, -1.4927135, 1.5665267
1: -0.3081098, 0.3671333, -0.6310252, 0.6842138, -0.9923235, 0.9981585
2: -0.4453541, 0.4715777, -0.7096369, 0.9015298, -1.3468839, 1.1812146
3: -0.2784572, 0.4204646, -0.6365123, 0.7542941, -1.0327513, 1.0569769
4: -0.4331292, 0.3879401, -0.8039209, 0.7572623, -1.1903914, 1.1918610
5: -0.4023151, 0.5572499, -0.7701505, 0.9430154, -1.3453305, 1.3274004
6: -0.4021399, 0.5418121, -0.8087141, 0.9989783, -1.4011182, 1.3505261
7: -0.3516772, 0.3877113, -0.6914809, 0.7801436, -1.1318209, 1.0791922
8: -0.3468599, 2.2045350, -1.1274624, 2.3969493, -2.7438092, 3.3319974
9: -0.5353960, 0.6282471, -0.7968447, 0.9831666, -1.5185626, 1.4250919

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1897216, upper bound: 7.2505753
time: 2.00 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1829970, upper bound: 7.2516525
time: 1.84 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3890590, 0.8700409, -0.5572048, 1.0215225, -1.4105815, 1.4272456
1: -0.3556631, 0.4050535, -0.4699133, 0.5243959, -0.8800590, 0.8749668
2: -0.4801306, 0.5281917, -0.5834042, 0.6958510, -1.1759815, 1.1115959
3: -0.3158977, 0.4725525, -0.4330887, 0.6099771, -0.9258748, 0.9056412
4: -0.4699343, 0.4326759, -0.5883676, 0.5913666, -1.0613009, 1.0210435
5: -0.4523340, 0.6063890, -0.6064373, 0.7595062, -1.2118402, 1.2128264
6: -0.4516869, 0.6084024, -0.6224301, 0.8497404, -1.3014274, 1.2308325
7: -0.3917261, 0.4398189, -0.5169876, 0.5921385, -0.9838645, 0.9568065
8: -0.4366053, 2.2364864, -0.7843096, 2.3310637, -2.7676690, 3.0207961
9: -0.5632542, 0.6795791, -0.6562840, 0.8172339, -1.3804882, 1.3358631

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1897216, upper bound: 7.2024976
time: 2.63 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1829970, upper bound: 7.2031861
time: 2.06 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3890590, 0.8700409, -0.7780148, 1.1847799, -1.5738388, 1.6480557
1: -0.3556631, 0.4050535, -0.6531084, 0.7034538, -1.0591168, 1.0581619
2: -0.4801306, 0.5281917, -0.7316617, 0.9264519, -1.4065825, 1.2598534
3: -0.3158977, 0.4725525, -0.6689183, 0.7724828, -1.0883805, 1.1414707
4: -0.4699343, 0.4326759, -0.8348164, 0.7801372, -1.2500714, 1.2674923
5: -0.4523340, 0.6063890, -0.7923228, 0.9680418, -1.4203758, 1.3987118
6: -0.4516869, 0.6084024, -0.8356036, 1.0207112, -1.4723980, 1.4440060
7: -0.3917261, 0.4398189, -0.7173139, 0.8058722, -1.1975982, 1.1571329
8: -0.4366053, 2.2364864, -1.1799490, 2.4115157, -2.8481209, 3.4164355
9: -0.5632542, 0.6795791, -0.8185070, 1.0045964, -1.5678506, 1.4980861

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1897216, upper bound: 7.2505753
time: 1.92 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1829970, upper bound: 7.2516525
time: 1.85 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3410109, 0.8279513, -0.6911799, 1.1368167, -1.4778277, 1.5191312
1: -0.3179950, 0.3759453, -0.5734826, 0.6358735, -0.9538684, 0.9494280
2: -0.4539731, 0.4824525, -0.6591180, 0.8483775, -1.3023506, 1.1415704
3: -0.2880446, 0.4304986, -0.5514018, 0.7078201, -0.9958646, 0.9819003
4: -0.4419237, 0.3987080, -0.7223427, 0.7022587, -1.1441824, 1.1210507
5: -0.4123716, 0.5712485, -0.7216972, 0.8695143, -1.2818859, 1.2929456
6: -0.4133470, 0.5534695, -0.7547137, 0.9701247, -1.3834717, 1.3081832
7: -0.3601148, 0.3986252, -0.6227636, 0.7189339, -1.0790486, 1.0213888
8: -0.3672892, 2.2166140, -1.0231345, 2.3709395, -2.7382288, 3.2397485
9: -0.5422921, 0.6401391, -0.7466612, 0.9303859, -1.4726781, 1.3868003

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1896834, upper bound: 7.2024976
time: 1.82 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1829885, upper bound: 7.2031861
time: 2.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3410109, 0.8279513, -0.9599378, 1.3107057, -1.6517166, 1.7878890
1: -0.3179950, 0.3759453, -0.7835268, 0.8263028, -1.1442978, 1.1594721
2: -0.4539731, 0.4824525, -0.8767243, 1.0670080, -1.5209812, 1.3591768
3: -0.2880446, 0.4304986, -0.8712076, 0.8831156, -1.1711602, 1.3017062
4: -0.4419237, 0.3987080, -1.0234220, 0.9205928, -1.3625166, 1.4221300
5: -0.4123716, 0.5712485, -0.9298132, 1.1107414, -1.5231130, 1.5010617
6: -0.4133470, 0.5534695, -0.9898437, 1.1640526, -1.5773996, 1.5433133
7: -0.3601148, 0.3986252, -0.8766977, 0.9569302, -1.3170450, 1.2753229
8: -0.3672892, 2.2166140, -1.4849511, 2.4557502, -2.8230395, 3.7015653
9: -0.5422921, 0.6401391, -0.9518197, 1.1371343, -1.6794264, 1.5919588

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1896834, upper bound: 7.2505753
time: 2.10 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1829885, upper bound: 7.2516525
time: 2.33 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.4030014, 0.8812332, -0.7182382, 1.1573544, -1.5603558, 1.5994713
1: -0.3661677, 0.4137633, -0.5952643, 0.6543461, -1.0205138, 1.0090276
2: -0.4896776, 0.5405312, -0.6782303, 0.8739872, -1.3636647, 1.2187614
3: -0.3254603, 0.4840786, -0.5813102, 0.7243759, -1.0498362, 1.0653888
4: -0.4785597, 0.4437129, -0.7528372, 0.7242160, -1.2027757, 1.1965500
5: -0.4646896, 0.6198086, -0.7427810, 0.8935397, -1.3582293, 1.3625896
6: -0.4637611, 0.6227113, -0.7797973, 0.9920164, -1.4557775, 1.4025086
7: -0.4014172, 0.4503762, -0.6477106, 0.7419355, -1.1433527, 1.0980868
8: -0.4592841, 2.2490435, -1.0725533, 2.3870482, -2.8463323, 3.3215966
9: -0.5705932, 0.6916361, -0.7664979, 0.9520399, -1.5226331, 1.4581339

Time for backsubstitution: 2.06 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 6.77 + 595.27 = 602.04 seconds
