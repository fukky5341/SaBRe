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
execution time: IAR + RelationalAnalysis = 23.42 + 33.48 = 56.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.2973304, upper bound: 0.2973307

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5734
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5734

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922994, upper bound: 0.2973261
time: 5.54 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973260, upper bound: 0.2973255
time: 6.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.00 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.00
Output dim: 1, lower bound: -0.2922994, upper bound: 0.2973261
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.00
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

Time for backsubstitution: 22.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2922996, upper bound: 0.2922976
time: 5.44 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922996, upper bound: 0.2973263
time: 6.23 seconds

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

Time for backsubstitution: 21.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973263, upper bound: 0.2922985
time: 4.01 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973280, upper bound: 0.2973272
time: 3.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.41 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 29.41
Output dim: 1, lower bound: -0.2922996, upper bound: 0.2922976
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.41
Output dim: 1, lower bound: -0.2922996, upper bound: 0.2973263
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.41
Output dim: 1, lower bound: -0.2973263, upper bound: 0.2922985
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.41
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

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914920, upper bound: 0.2973220
time: 3.73 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922926, upper bound: 0.2973197
time: 4.26 seconds

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

Time for backsubstitution: 22.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964891, upper bound: 0.2922914
time: 3.79 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973183, upper bound: 0.2922931
time: 3.84 seconds

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

Time for backsubstitution: 22.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964902, upper bound: 0.2922915
time: 3.96 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973194, upper bound: 0.2922931
time: 3.95 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.26 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.26
Output dim: 1, lower bound: -0.2914920, upper bound: 0.2973220
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.26
Output dim: 1, lower bound: -0.2922926, upper bound: 0.2973197
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.26
Output dim: 1, lower bound: -0.2964891, upper bound: 0.2922914
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.26
Output dim: 1, lower bound: -0.2973183, upper bound: 0.2922931
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.26
Output dim: 1, lower bound: -0.2964902, upper bound: 0.2922915
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.26
Output dim: 1, lower bound: -0.2973194, upper bound: 0.2922931

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.6361675, -7.4429913, -8.6531315, -7.4316969, -0.7290392, 0.7270250
1: 2.8367260, 3.7670465, 2.8165851, 3.7909191, -0.5215884, 0.5299517
2: -6.2357626, -5.2874660, -6.2491841, -5.2762256, -0.5180693, 0.5133044
3: -11.2072811, -10.0488548, -11.2483654, -10.0188551, -0.5747404, 0.5668496
4: -3.9955168, -3.0592208, -4.0260959, -3.0185142, -0.6017396, 0.6137292
5: -12.1389875, -11.0239277, -12.1413212, -11.0214882, -0.5583067, 0.5583732
6: -9.7615032, -8.5194559, -9.7755442, -8.5045338, -0.6210399, 0.6281960
7: -3.9427392, -2.9736078, -3.9606645, -2.9604299, -0.4108033, 0.4071831
8: -3.1050024, -2.1762128, -3.1214476, -2.1636806, -0.4135585, 0.4095182
9: -11.8273964, -10.8493996, -11.8589468, -10.8251781, -0.4673296, 0.4592712

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914920, upper bound: 0.2964888
time: 3.65 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914920, upper bound: 0.2973204
time: 3.93 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.6588860, -7.4286060, -8.6531277, -7.4316988, -0.7439494, 0.7293856
1: 2.8282053, 3.8025131, 2.8165884, 3.7909184, -0.5321416, 0.5405754
2: -6.2976565, -5.2844133, -6.2491856, -5.2762356, -0.5324425, 0.5166847
3: -11.2086267, -9.9862938, -11.2483578, -10.0188580, -0.5768294, 0.5856488
4: -4.0001717, -3.0338564, -4.0260925, -3.0185142, -0.6082522, 0.6175170
5: -12.1398859, -10.9851131, -12.1413145, -11.0214901, -0.5605328, 0.5703369
6: -9.7772741, -8.4962759, -9.7755413, -8.5045338, -0.6356194, 0.6305096
7: -3.9506223, -2.9598076, -3.9606636, -2.9604301, -0.4157721, 0.4158521
8: -3.1400537, -2.1734629, -3.1214485, -2.1636834, -0.4230255, 0.4128969
9: -11.8497181, -10.8412476, -11.8589468, -10.8251829, -0.4691617, 0.4679209

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 820

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922923, upper bound: 0.2973097
time: 3.68 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922923, upper bound: 0.2973194
time: 4.31 seconds

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

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964891, upper bound: 0.2914902
time: 3.78 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964891, upper bound: 0.2922926
time: 3.80 seconds

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

Time for backsubstitution: 22.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 820

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973180, upper bound: 0.2922846
time: 3.68 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973180, upper bound: 0.2922946
time: 3.89 seconds

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

Time for backsubstitution: 22.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964899, upper bound: 0.2914885
time: 3.99 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964899, upper bound: 0.2922908
time: 3.77 seconds

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

Time for backsubstitution: 22.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 820

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973192, upper bound: 0.2922829
time: 3.95 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973188, upper bound: 0.2922906
time: 3.94 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.18 seconds
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.18
Output dim: 1, lower bound: -0.2914920, upper bound: 0.2964888
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.18
Output dim: 1, lower bound: -0.2914920, upper bound: 0.2973204
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.18
Output dim: 1, lower bound: -0.2922923, upper bound: 0.2973097
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.18
Output dim: 1, lower bound: -0.2922923, upper bound: 0.2973194
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.18
Output dim: 1, lower bound: -0.2964891, upper bound: 0.2914902
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.18
Output dim: 1, lower bound: -0.2964891, upper bound: 0.2922926
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.18
Output dim: 1, lower bound: -0.2973180, upper bound: 0.2922846
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.18
Output dim: 1, lower bound: -0.2973180, upper bound: 0.2922946
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.18
Output dim: 1, lower bound: -0.2964899, upper bound: 0.2914885
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.18
Output dim: 1, lower bound: -0.2964899, upper bound: 0.2922908
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.18
Output dim: 1, lower bound: -0.2973192, upper bound: 0.2922829
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.18
Output dim: 1, lower bound: -0.2973188, upper bound: 0.2922906

## BFS NS instance: NS_A1_B2_A1_B1

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

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2912242, upper bound: 0.2964892
time: 3.93 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914903, upper bound: 0.2964911
time: 4.22 seconds

## BFS NS instance: NS_A1_B2_A1_B2

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

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2912242, upper bound: 0.2973217
time: 4.07 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914903, upper bound: 0.2973206
time: 4.17 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.6579227, -7.4293747, -8.6522617, -7.4333558, -0.7379704, 0.7245820
1: 2.8289342, 3.8018479, 2.8181040, 3.7902660, -0.5300421, 0.5372462
2: -6.2969522, -5.2873249, -6.2487125, -5.2830954, -0.5247262, 0.5128562
3: -11.2050180, -9.9885082, -11.2393093, -10.0218735, -0.5716883, 0.5751854
4: -3.9984899, -3.0340674, -4.0221729, -3.0186212, -0.6060920, 0.6130201
5: -12.1381149, -10.9866228, -12.1368732, -11.0240822, -0.5579426, 0.5651723
6: -9.7743282, -8.4967098, -9.7687531, -8.5056849, -0.6308296, 0.6230233
7: -3.9488580, -2.9603193, -3.9562368, -2.9605885, -0.4134623, 0.4105866
8: -3.1395082, -2.1766868, -3.1211157, -2.1711450, -0.4148897, 0.4088671
9: -11.8493681, -10.8437166, -11.8581381, -10.8309879, -0.4619821, 0.4632308

Time for backsubstitution: 22.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922824, upper bound: 0.2973070
time: 3.41 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922824, upper bound: 0.2973073
time: 4.07 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.6605358, -7.4285870, -8.6605711, -7.4274554, -0.7474513, 0.7300544
1: 2.8281693, 3.8036952, 2.8091722, 3.7979853, -0.5339377, 0.5493617
2: -6.2990847, -5.2844038, -6.2743988, -5.2724957, -0.5337473, 0.5216904
3: -11.2086163, -9.9841652, -11.2493896, -9.9844151, -0.5845408, 0.5848638
4: -4.0002007, -3.0333931, -4.0314865, -3.0078344, -0.6103311, 0.6232152
5: -12.1398869, -10.9841766, -12.1415749, -11.0006075, -0.5704055, 0.5707371
6: -9.7779341, -8.4962730, -9.7829247, -8.4781876, -0.6389709, 0.6358173
7: -3.9506273, -2.9589608, -3.9637184, -2.9470537, -0.4181416, 0.4172587
8: -3.1411972, -2.1734524, -3.1464853, -2.1603160, -0.4228570, 0.4155991
9: -11.8497276, -10.8408184, -11.8873444, -10.8231049, -0.4682437, 0.4721836

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922824, upper bound: 0.2973185
time: 3.53 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922824, upper bound: 0.2973175
time: 3.80 seconds

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

Time for backsubstitution: 22.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2962264, upper bound: 0.2914908
time: 5.76 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964888, upper bound: 0.2914910
time: 3.79 seconds

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

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2962264, upper bound: 0.2922924
time: 5.57 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964888, upper bound: 0.2922929
time: 4.06 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.6745024, -7.4182296, -8.6357613, -7.4445033, -0.7347198, 0.7278695
1: 2.8096299, 3.8256395, 2.8374407, 3.7664747, -0.5374480, 0.5299358
2: -6.3101840, -5.2777863, -6.2354803, -5.2926507, -0.5215081, 0.5162338
3: -11.2451277, -9.9589386, -11.1992006, -10.0514584, -0.5628873, 0.5838276
4: -4.0281887, -2.9934192, -3.9924386, -3.0592680, -0.6175355, 0.6019421
5: -12.1396694, -10.9844990, -12.1353226, -11.0262203, -0.5572671, 0.5671293
6: -9.7874527, -8.4818840, -9.7554846, -8.5205097, -0.6378810, 0.6160676
7: -3.9666855, -2.9472570, -3.9384122, -2.9736433, -0.4097276, 0.4143164
8: -3.1558447, -2.1649561, -3.1047831, -2.1828766, -0.4117463, 0.4120383
9: -11.8808670, -10.8204527, -11.8266392, -10.8542719, -0.4549228, 0.4704280

Time for backsubstitution: 22.28 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.90 + 552.19 = 609.09 seconds
