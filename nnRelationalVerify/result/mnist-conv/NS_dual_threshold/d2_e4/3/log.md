## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.294357096


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7342215, 0.7342215)
1: (2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5336686, 0.5336686)
2: (-6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5237415, 0.5237415)
3: (-11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5927124, 0.5927125)
4: (-4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.6310592, 0.6310592)
5: (-12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5609331, 0.5609331)
6: (-9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6248200, 0.6248200)
7: (-3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4181032, 0.4181032)
8: (-3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4160961, 0.4160961)
9: (-11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4786108, 0.4786108)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.56 + 34.49 = 58.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.2973304, upper bound: 0.2973307

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5734
type: B, layer: 1, pos: 5734
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5734

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922994, upper bound: 0.2973261
time: 5.63 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973260, upper bound: 0.2973255
time: 6.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.21 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.21
Output dim: 1, lower bound: -0.2922994, upper bound: 0.2973261
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.21
Output dim: 1, lower bound: -0.2973260, upper bound: 0.2973255

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -8.6366272, -7.4428334, -8.6367912, -7.4385543, -0.7239096, 0.7195573
1: 2.8359103, 3.7671263, 2.8284652, 3.7671299, -0.5157082, 0.5232511
2: -6.2359533, -5.2857752, -6.2360425, -5.2820935, -0.5164936, 0.5127643
3: -11.2082558, -10.0484447, -11.2083073, -10.0362854, -0.5751991, 0.5633301
4: -3.9963717, -3.0591626, -4.0081587, -3.0590663, -0.5991445, 0.6111042
5: -12.1397696, -11.0236378, -12.1397753, -11.0232229, -0.5580468, 0.5576177
6: -9.7622900, -8.5193605, -9.7666702, -8.5193129, -0.6143708, 0.6187150
7: -3.9428413, -2.9734912, -3.9429138, -2.9683967, -0.4086664, 0.4035296
8: -3.1051149, -2.1754127, -3.1051307, -2.1704984, -0.4083323, 0.4034185
9: -11.8274498, -10.8484554, -11.8275185, -10.8389702, -0.4643834, 0.4548557

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2922996, upper bound: 0.2922976
time: 5.50 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922977, upper bound: 0.2973262
time: 7.11 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.6531315, -7.4316969, -8.6370096, -7.4326563, -0.7404599, 0.7293153
1: 2.8165851, 3.7909191, 2.8181765, 3.7671344, -0.5235007, 0.5406926
2: -6.2491841, -5.2762256, -6.2361579, -5.2770157, -0.5252248, 0.5188299
3: -11.2483654, -10.0188551, -11.2083759, -10.0195122, -0.5964694, 0.5728962
4: -4.0260959, -3.0185142, -4.0244174, -3.0589390, -0.6152551, 0.6335886
5: -12.1413212, -11.0214882, -12.1397896, -11.0226460, -0.5624359, 0.5606136
6: -9.7755442, -8.5045338, -9.7727070, -8.5192490, -0.6218660, 0.6323264
7: -3.9606645, -2.9604299, -3.9430113, -2.9613719, -0.4209969, 0.4113854
8: -3.1214476, -2.1636806, -3.1051564, -2.1637249, -0.4228899, 0.4078695
9: -11.8589468, -10.8251781, -11.8276062, -10.8258915, -0.4837551, 0.4636161

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973263, upper bound: 0.2922985
time: 4.24 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973280, upper bound: 0.2973272
time: 3.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.59 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 30.59
Output dim: 1, lower bound: -0.2922996, upper bound: 0.2922976
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 30.59
Output dim: 1, lower bound: -0.2922977, upper bound: 0.2973262
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.59
Output dim: 1, lower bound: -0.2973263, upper bound: 0.2922985
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.59
Output dim: 1, lower bound: -0.2973280, upper bound: 0.2973272

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -8.6366272, -7.4428334, -8.6531315, -7.4316969, -0.7297869, 0.7273681
1: 2.8359103, 3.7671263, 2.8165851, 3.7909191, -0.5227283, 0.5300392
2: -6.2359533, -5.2857752, -6.2491841, -5.2762256, -0.5182219, 0.5150125
3: -11.2082558, -10.0484447, -11.2483654, -10.0188551, -0.5759110, 0.5671898
4: -3.9963717, -3.0591626, -4.0260959, -3.0185142, -0.6029208, 0.6139719
5: -12.1397696, -11.0236378, -12.1413212, -11.0214882, -0.5591631, 0.5585759
6: -9.7622900, -8.5193605, -9.7755442, -8.5045338, -0.6219120, 0.6285191
7: -3.9428413, -2.9734912, -3.9606645, -2.9604299, -0.4110159, 0.4072764
8: -3.1051149, -2.1754127, -3.1214476, -2.1636806, -0.4137038, 0.4105681
9: -11.8274498, -10.8484554, -11.8589468, -10.8251781, -0.4674567, 0.4603900

Time for backsubstitution: 22.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922934, upper bound: 0.2964900
time: 4.29 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922946, upper bound: 0.2973212
time: 3.55 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.6531315, -7.4316969, -8.6366272, -7.4428334, -0.7273679, 0.7297871
1: 2.8165851, 3.7909191, 2.8359103, 3.7671263, -0.5300391, 0.5227282
2: -6.2491841, -5.2762256, -6.2359533, -5.2857752, -0.5150124, 0.5182219
3: -11.2483654, -10.0188551, -11.2082558, -10.0484447, -0.5671897, 0.5759112
4: -4.0260959, -3.0185142, -3.9963717, -3.0591626, -0.6139719, 0.6029207
5: -12.1413212, -11.0214882, -12.1397696, -11.0236378, -0.5585759, 0.5591631
6: -9.7755442, -8.5045338, -9.7622900, -8.5193605, -0.6285191, 0.6219120
7: -3.9606645, -2.9604299, -3.9428413, -2.9734912, -0.4072765, 0.4110159
8: -3.1214476, -2.1636806, -3.1051149, -2.1754127, -0.4105681, 0.4137037
9: -11.8589468, -10.8251781, -11.8274498, -10.8484554, -0.4603900, 0.4674567

Time for backsubstitution: 22.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964891, upper bound: 0.2922914
time: 3.76 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973183, upper bound: 0.2922931
time: 3.80 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.6531315, -7.4316969, -8.6531315, -7.4316969, -0.7296638, 0.7296641
1: 2.8165851, 3.7909191, 2.8165851, 3.7909191, -0.5235159, 0.5235159
2: -6.2491841, -5.2762256, -6.2491841, -5.2762256, -0.5188911, 0.5188911
3: -11.2483654, -10.0188551, -11.2483654, -10.0188551, -0.5733231, 0.5733230
4: -4.0260959, -3.0185142, -4.0260959, -3.0185142, -0.6153867, 0.6153867
5: -12.1413212, -11.0214882, -12.1413212, -11.0214882, -0.5635061, 0.5644808
6: -9.7755442, -8.5045338, -9.7755442, -8.5045338, -0.6235497, 0.6235498
7: -3.9606645, -2.9604299, -3.9606645, -2.9604299, -0.4115672, 0.4115672
8: -3.1214476, -2.1636806, -3.1214476, -2.1636806, -0.4081160, 0.4081160
9: -11.8589468, -10.8251781, -11.8589468, -10.8251781, -0.4639804, 0.4639804

Time for backsubstitution: 22.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964902, upper bound: 0.2922915
time: 3.85 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973194, upper bound: 0.2922931
time: 3.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.18 seconds
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 30.18
Output dim: 1, lower bound: -0.2922934, upper bound: 0.2964900
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 30.18
Output dim: 1, lower bound: -0.2922946, upper bound: 0.2973212
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.18
Output dim: 1, lower bound: -0.2964891, upper bound: 0.2922914
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.18
Output dim: 1, lower bound: -0.2973183, upper bound: 0.2922931
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.18
Output dim: 1, lower bound: -0.2964902, upper bound: 0.2922915
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.18
Output dim: 1, lower bound: -0.2973194, upper bound: 0.2922931

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -8.6366272, -7.4428334, -8.6526709, -7.4318562, -0.7294564, 0.7266002
1: 2.8359103, 3.7671263, 2.8173995, 3.7908382, -0.5226405, 0.5288992
2: -6.2359533, -5.2857752, -6.2489948, -5.2779164, -0.5165139, 0.5148600
3: -11.2082558, -10.0484447, -11.2473907, -10.0192680, -0.5755658, 0.5660186
4: -3.9963717, -3.0591626, -4.0252419, -3.0185723, -0.6026778, 0.6127901
5: -12.1397696, -11.0236378, -12.1405430, -11.0217752, -0.5589590, 0.5577199
6: -9.7622900, -8.5193605, -9.7747593, -8.5046301, -0.6216109, 0.6276519
7: -3.9428413, -2.9734912, -3.9605629, -2.9605465, -0.4109228, 0.4070637
8: -3.1051149, -2.1754127, -3.1213350, -2.1644812, -0.4126538, 0.4104230
9: -11.8274498, -10.8484554, -11.8588963, -10.8261204, -0.4663376, 0.4602631

Time for backsubstitution: 21.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914920, upper bound: 0.2964904
time: 3.92 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914920, upper bound: 0.2964915
time: 4.07 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -8.6366253, -7.4428329, -8.6754646, -7.4174623, -0.7326798, 0.7406888
1: 2.8359156, 3.7671270, 2.8089066, 3.8263052, -0.5332644, 0.5395497
2: -6.2359529, -5.2857857, -6.3108883, -5.2748771, -0.5200560, 0.5292330
3: -11.2082500, -10.0484447, -11.2487354, -9.9567242, -0.5942967, 0.5680146
4: -3.9963670, -3.0591629, -4.0298638, -2.9932082, -0.6064651, 0.6196797
5: -12.1397619, -11.0236397, -12.1414404, -10.9829836, -0.5722959, 0.5599456
6: -9.7622852, -8.5193586, -9.7903929, -8.4814510, -0.6235974, 0.6426513
7: -3.9428411, -2.9734914, -3.9684496, -2.9467435, -0.4195839, 0.4120321
8: -3.1051145, -2.1754141, -3.1563926, -2.1617312, -0.4160674, 0.4198827
9: -11.8274479, -10.8484592, -11.8812180, -10.8179846, -0.4751205, 0.4620953

Time for backsubstitution: 22.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914906, upper bound: 0.2973205
time: 4.12 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914906, upper bound: 0.2973174
time: 5.65 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.6526709, -7.4318562, -8.6366272, -7.4428334, -0.7266002, 0.7294562
1: 2.8173995, 3.7908382, 2.8359103, 3.7671263, -0.5288991, 0.5226405
2: -6.2489948, -5.2779164, -6.2359533, -5.2857752, -0.5148600, 0.5165138
3: -11.2473907, -10.0192680, -11.2082558, -10.0484447, -0.5660186, 0.5755657
4: -4.0252419, -3.0185723, -3.9963717, -3.0591626, -0.6127899, 0.6026778
5: -12.1405430, -11.0217752, -12.1397696, -11.0236378, -0.5577199, 0.5589590
6: -9.7747593, -8.5046301, -9.7622900, -8.5193605, -0.6276519, 0.6216111
7: -3.9605629, -2.9605465, -3.9428413, -2.9734912, -0.4070637, 0.4109228
8: -3.1213350, -2.1644812, -3.1051149, -2.1754127, -0.4104229, 0.4126538
9: -11.8588963, -10.8261204, -11.8274498, -10.8484554, -0.4602631, 0.4663376

Time for backsubstitution: 22.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964891, upper bound: 0.2914902
time: 3.80 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964891, upper bound: 0.2922926
time: 3.95 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.6754646, -7.4174623, -8.6366253, -7.4428329, -0.7406888, 0.7326798
1: 2.8089066, 3.8263052, 2.8359156, 3.7671270, -0.5395498, 0.5332643
2: -6.3108883, -5.2748771, -6.2359529, -5.2857857, -0.5292332, 0.5200560
3: -11.2487354, -9.9567242, -11.2082500, -10.0484447, -0.5680146, 0.5942968
4: -4.0298638, -2.9932082, -3.9963670, -3.0591629, -0.6196797, 0.6064649
5: -12.1414404, -10.9829836, -12.1397619, -11.0236397, -0.5599456, 0.5722959
6: -9.7903929, -8.4814510, -9.7622852, -8.5193586, -0.6426516, 0.6235971
7: -3.9684496, -2.9467435, -3.9428411, -2.9734914, -0.4120322, 0.4195839
8: -3.1563926, -2.1617312, -3.1051145, -2.1754141, -0.4198827, 0.4160674
9: -11.8812180, -10.8179846, -11.8274479, -10.8484592, -0.4620953, 0.4751204

Time for backsubstitution: 22.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973187, upper bound: 0.2914920
time: 4.14 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973187, upper bound: 0.2922948
time: 4.03 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.6526709, -7.4318562, -8.6531315, -7.4316969, -0.7289171, 0.7293334
1: 2.8173995, 3.7908382, 2.8165851, 3.7909191, -0.5223775, 0.5234305
2: -6.2489948, -5.2779164, -6.2491841, -5.2762256, -0.5187371, 0.5171900
3: -11.2473907, -10.0192680, -11.2483654, -10.0188551, -0.5721594, 0.5729718
4: -4.0252419, -3.0185723, -4.0260959, -3.0185142, -0.6142070, 0.6151452
5: -12.1405430, -11.0217752, -12.1413212, -11.0214882, -0.5626409, 0.5642772
6: -9.7747593, -8.5046301, -9.7755442, -8.5045338, -0.6226838, 0.6232270
7: -3.9605629, -2.9605465, -3.9606645, -2.9604299, -0.4113510, 0.4114738
8: -3.1213350, -2.1644812, -3.1214476, -2.1636806, -0.4079720, 0.4070693
9: -11.8588963, -10.8261204, -11.8589468, -10.8251781, -0.4638548, 0.4628615

Time for backsubstitution: 22.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964899, upper bound: 0.2914885
time: 4.14 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964899, upper bound: 0.2922908
time: 3.87 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.6754646, -7.4174623, -8.6531277, -7.4316988, -0.7495995, 0.7366178
1: 2.8089066, 3.8263052, 2.8165884, 3.7909184, -0.5401434, 0.5414450
2: -6.3108883, -5.2748771, -6.2491856, -5.2762356, -0.5372801, 0.5223753
3: -11.2487354, -9.9567242, -11.2483578, -10.0188580, -0.5778732, 0.5950006
4: -4.0298638, -2.9932082, -4.0260925, -3.0185142, -0.6235391, 0.6246006
5: -12.1414404, -10.9829836, -12.1413145, -11.0214901, -0.5648854, 0.5753256
6: -9.7903929, -8.4814510, -9.7755413, -8.5045338, -0.6393504, 0.6334639
7: -3.9684496, -2.9467435, -3.9606636, -2.9604301, -0.4201481, 0.4217204
8: -3.1563926, -2.1617312, -3.1214485, -2.1636834, -0.4251269, 0.4174166
9: -11.8812180, -10.8179846, -11.8589468, -10.8251829, -0.4715661, 0.4772941

Time for backsubstitution: 22.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973195, upper bound: 0.2914903
time: 4.50 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973195, upper bound: 0.2922931
time: 4.05 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.34 seconds
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 31.34
Output dim: 1, lower bound: -0.2914920, upper bound: 0.2964904
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 31.34
Output dim: 1, lower bound: -0.2914920, upper bound: 0.2964915
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 31.34
Output dim: 1, lower bound: -0.2914906, upper bound: 0.2973205
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 31.34
Output dim: 1, lower bound: -0.2914906, upper bound: 0.2973174
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.34
Output dim: 1, lower bound: -0.2964891, upper bound: 0.2914902
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.34
Output dim: 1, lower bound: -0.2964891, upper bound: 0.2922926
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.34
Output dim: 1, lower bound: -0.2973187, upper bound: 0.2914920
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.34
Output dim: 1, lower bound: -0.2973187, upper bound: 0.2922948
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.34
Output dim: 1, lower bound: -0.2964899, upper bound: 0.2914885
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.34
Output dim: 1, lower bound: -0.2964899, upper bound: 0.2922908
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.34
Output dim: 1, lower bound: -0.2973195, upper bound: 0.2914903
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.34
Output dim: 1, lower bound: -0.2973195, upper bound: 0.2922931

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -8.6361675, -7.4429913, -8.6526709, -7.4318562, -0.7287087, 0.7262571
1: 2.8367260, 3.7670465, 2.8173995, 3.7908382, -0.5215006, 0.5288117
2: -6.2357626, -5.2874660, -6.2489948, -5.2779164, -0.5163614, 0.5131518
3: -11.2072811, -10.0488548, -11.2473907, -10.0192680, -0.5743949, 0.5656785
4: -3.9955168, -3.0592208, -4.0252419, -3.0185723, -0.6014967, 0.6125473
5: -12.1389875, -11.0239277, -12.1405430, -11.0217752, -0.5581026, 0.5575171
6: -9.7615032, -8.5194559, -9.7747593, -8.5046301, -0.6207392, 0.6273291
7: -3.9427392, -2.9736078, -3.9605629, -2.9605465, -0.4107102, 0.4069703
8: -3.1050024, -2.1762128, -3.1213350, -2.1644812, -0.4125085, 0.4093730
9: -11.8273964, -10.8493996, -11.8588963, -10.8261204, -0.4662105, 0.4591444

Time for backsubstitution: 22.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of NS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914913, upper bound: 0.2962262
time: 3.98 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914927, upper bound: 0.2964901
time: 3.88 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -8.6572046, -7.4286318, -8.6526709, -7.4318562, -0.7421684, 0.7281153
1: 2.8282523, 3.8013082, 2.8173995, 3.7908382, -0.5312676, 0.5385616
2: -6.2962008, -5.2844429, -6.2489948, -5.2779164, -0.5299250, 0.5162392
3: -11.2086258, -9.9884844, -11.2473907, -10.0192680, -0.5760510, 0.5829457
4: -4.0001287, -3.0343277, -4.0252419, -3.0185723, -0.6068099, 0.6160678
5: -12.1398859, -10.9860859, -12.1405430, -11.0217752, -0.5591240, 0.5690356
6: -9.7765846, -8.4962826, -9.7747593, -8.5046301, -0.6343186, 0.6296356
7: -3.9506109, -2.9606798, -3.9605629, -2.9605465, -0.4153991, 0.4149101
8: -3.1388903, -2.1735001, -3.1213350, -2.1644812, -0.4213063, 0.4120572
9: -11.8497076, -10.8416977, -11.8588963, -10.8261204, -0.4680351, 0.4663670

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of NS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of NS_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914913, upper bound: 0.2962280
time: 4.09 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914927, upper bound: 0.2964879
time: 3.95 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -8.6361675, -7.4429913, -8.6737862, -7.4174900, -0.7314095, 0.7389095
1: 2.8367260, 3.7670465, 2.8089509, 3.8251002, -0.5312505, 0.5385872
2: -6.2357626, -5.2874660, -6.3094320, -5.2749066, -0.5194390, 0.5267150
3: -11.2072811, -10.0488548, -11.2487345, -9.9589138, -0.5915945, 0.5673345
4: -3.9955168, -3.0592208, -4.0298228, -2.9936793, -0.6050169, 0.6178235
5: -12.1389875, -11.0239277, -12.1414404, -10.9839544, -0.5709937, 0.5585377
6: -9.7615032, -8.5194559, -9.7897043, -8.4814577, -0.6227205, 0.6408513
7: -3.9427392, -2.9736078, -3.9684374, -2.9476163, -0.4186422, 0.4116590
8: -3.1050024, -2.1762128, -3.1552286, -2.1617684, -0.4151926, 0.4181633
9: -11.8273964, -10.8493996, -11.8812046, -10.8184366, -0.4734493, 0.4609687

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of NS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914903, upper bound: 0.2973103
time: 3.77 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914903, upper bound: 0.2973181
time: 3.86 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -8.6751480, -7.4284849, -8.6916924, -7.4173422, -0.7605066, 0.7569292
1: 2.8278883, 3.8141346, 2.8086019, 3.8379273, -0.5510420, 0.5601774
2: -6.3116951, -5.2843080, -6.3249254, -5.2747788, -0.5441973, 0.5388272
3: -11.2086334, -9.9650421, -11.2487450, -9.9354830, -0.6116316, 0.6050196
4: -4.0004892, -3.0293078, -4.0301704, -2.9886613, -0.6141007, 0.6269984
5: -12.1398859, -10.9759655, -12.1414404, -10.9738436, -0.5768405, 0.5761045
6: -9.7879639, -8.4962358, -9.8011351, -8.4814081, -0.6501698, 0.6576819
7: -3.9507289, -2.9515543, -3.9685566, -2.9384925, -0.4317743, 0.4281821
8: -3.1512899, -2.1733737, -3.1676240, -2.1616426, -0.4333286, 0.4284578
9: -11.8497982, -10.8369122, -11.8812962, -10.8136559, -0.4834671, 0.4746853

Time for backsubstitution: 22.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of NS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914903, upper bound: 0.2962268
time: 5.25 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914903, upper bound: 0.2964890
time: 4.24 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.6526709, -7.4318562, -8.6361675, -7.4429913, -0.7262573, 0.7287085
1: 2.8173995, 3.7908382, 2.8367260, 3.7670465, -0.5288116, 0.5215005
2: -6.2489948, -5.2779164, -6.2357626, -5.2874660, -0.5131518, 0.5163614
3: -11.2473907, -10.0192680, -11.2072811, -10.0488548, -0.5656786, 0.5743949
4: -4.0252419, -3.0185723, -3.9955168, -3.0592208, -0.6125474, 0.6014967
5: -12.1405430, -11.0217752, -12.1389875, -11.0239277, -0.5575171, 0.5581028
6: -9.7747593, -8.5046301, -9.7615032, -8.5194559, -0.6273289, 0.6207390
7: -3.9605629, -2.9605465, -3.9427392, -2.9736078, -0.4069703, 0.4107101
8: -3.1213350, -2.1644812, -3.1050024, -2.1762128, -0.4093729, 0.4125085
9: -11.8588963, -10.8261204, -11.8273964, -10.8493996, -0.4591444, 0.4662105

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2962264, upper bound: 0.2914908
time: 5.77 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964888, upper bound: 0.2914910
time: 3.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.6526709, -7.4318562, -8.6572046, -7.4286318, -0.7281151, 0.7421684
1: 2.8173995, 3.7908382, 2.8282523, 3.8013082, -0.5385616, 0.5312676
2: -6.2489948, -5.2779164, -6.2962008, -5.2844429, -0.5162394, 0.5299251
3: -11.2473907, -10.0192680, -11.2086258, -9.9884844, -0.5829457, 0.5760511
4: -4.0252419, -3.0185723, -4.0001287, -3.0343277, -0.6160676, 0.6068099
5: -12.1405430, -11.0217752, -12.1398859, -10.9860859, -0.5690358, 0.5591240
6: -9.7747593, -8.5046301, -9.7765846, -8.4962826, -0.6296358, 0.6343188
7: -3.9605629, -2.9605465, -3.9506109, -2.9606798, -0.4149101, 0.4153991
8: -3.1213350, -2.1644812, -3.1388903, -2.1735001, -0.4120573, 0.4213063
9: -11.8588963, -10.8261204, -11.8497076, -10.8416977, -0.4663670, 0.4680351

Time for backsubstitution: 22.11 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.05 + 558.49 = 616.54 seconds
