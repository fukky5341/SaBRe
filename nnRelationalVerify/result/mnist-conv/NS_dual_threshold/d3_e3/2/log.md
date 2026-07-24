## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.281490264


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1638608, 1.1638610)
1: (3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5360075, 0.5360075)
2: (-4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5615702, 0.5615700)
3: (-12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.8178110, 0.8178110)
4: (-2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7643485, 0.7643486)
5: (-9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5886670, 0.5886672)
6: (-7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8692248, 0.8692250)
7: (-2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3831897, 0.3831897)
8: (-3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6626787, 0.6626787)
9: (-12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7445683, 0.7445687)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.87 + 34.04 = 56.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.2843336, upper bound: 0.2843319

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5815
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5815

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2840776, upper bound: 0.2791966
time: 3.88 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2843287, upper bound: 0.2843281
time: 3.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.33 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.33
Output dim: 1, lower bound: -0.2840776, upper bound: 0.2791966
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.33
Output dim: 1, lower bound: -0.2843287, upper bound: 0.2843281

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -12.1896667, -10.7201614, -12.2553892, -10.6901646, -1.0623960, 1.1020713
1: 3.4026384, 4.2648273, 3.3842697, 4.2731686, -0.5092255, 0.5169574
2: -4.7339611, -3.9685204, -4.7578964, -3.9574156, -0.5266166, 0.5386496
3: -12.5080194, -11.2808743, -12.5385303, -11.2235260, -0.7550774, 0.7206509
4: -2.1697109, -1.1134877, -2.1788936, -1.1090724, -0.7504606, 0.7514219
5: -9.8784008, -8.8916817, -9.8870106, -8.8736668, -0.5698094, 0.5621703
6: -7.8049998, -6.6457176, -7.8500810, -6.6278405, -0.8022180, 0.8259497
7: -2.6601472, -2.0528200, -2.6610959, -2.0501370, -0.3786205, 0.3776418
8: -3.6220856, -2.6548557, -3.6509314, -2.6387558, -0.6183960, 0.6311795
9: -12.2878628, -11.2276583, -12.2979374, -11.2135477, -0.7245100, 0.7187507

Time for backsubstitution: 21.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2840278, upper bound: 0.2758130
time: 3.89 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2840693, upper bound: 0.2791892
time: 3.48 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -12.2602673, -10.6615887, -12.2602692, -10.6615725, -1.1185076, 1.1295078
1: 3.3816423, 4.2802610, 3.3816414, 4.2802696, -0.5270977, 0.5287813
2: -4.7594147, -3.9468060, -4.7594166, -3.9467974, -0.5488039, 0.5475727
3: -12.5689430, -11.2200089, -12.5689688, -11.2200079, -0.7504498, 0.7699842
4: -2.1814761, -1.1082799, -2.1814768, -1.1082799, -0.7630789, 0.7637405
5: -9.8950357, -8.8726816, -9.8950434, -8.8726807, -0.5780897, 0.5810763
6: -7.8550754, -6.6118484, -7.8550768, -6.6118383, -0.8513980, 0.8586516
7: -2.6614094, -2.0481157, -2.6614099, -2.0481150, -0.3828671, 0.3829144
8: -3.6533604, -2.6237764, -3.6533632, -2.6237659, -0.6434140, 0.6511225
9: -12.3033466, -11.2095785, -12.3033543, -11.2095757, -0.7440338, 0.7461543

Time for backsubstitution: 21.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2809258, upper bound: 0.2842745
time: 3.36 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2843186, upper bound: 0.2843198
time: 3.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.60 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.60
Output dim: 1, lower bound: -0.2840278, upper bound: 0.2758130
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.60
Output dim: 1, lower bound: -0.2840693, upper bound: 0.2791892
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 28.60
Output dim: 1, lower bound: -0.2809258, upper bound: 0.2842745
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 28.60
Output dim: 1, lower bound: -0.2843186, upper bound: 0.2843198

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -12.1878099, -10.7224007, -12.2517338, -10.6946659, -1.0543485, 1.0956264
1: 3.4088454, 4.2642565, 3.3968010, 4.2719736, -0.4999349, 0.5032533
2: -4.7327061, -3.9715309, -4.7555079, -3.9634695, -0.5193377, 0.5342877
3: -12.5072165, -11.2916002, -12.5368519, -11.2434378, -0.7359343, 0.7077968
4: -2.1681695, -1.1218863, -2.1760983, -1.1259422, -0.7324989, 0.7409920
5: -9.8584661, -8.8929510, -9.8468704, -8.8754568, -0.5484744, 0.5222908
6: -7.7897081, -6.6463513, -7.8192291, -6.6291666, -0.7836859, 0.7939922
7: -2.6601472, -2.0571985, -2.6610959, -2.0586338, -0.3666295, 0.3690907
8: -3.6204443, -2.6705756, -3.6478987, -2.6704264, -0.5852172, 0.6117899
9: -12.2787485, -11.2287388, -12.2797785, -11.2152147, -0.7020097, 0.6906588

Time for backsubstitution: 22.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2836863, upper bound: 0.2715559
time: 3.43 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2840237, upper bound: 0.2758090
time: 4.35 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -12.1896648, -10.7201672, -12.2583752, -10.6897182, -1.0615613, 1.1062658
1: 3.4026499, 4.2648292, 3.3828955, 4.2843399, -0.5099988, 0.5114723
2: -4.7339602, -3.9685259, -4.7623839, -3.9558229, -0.5276712, 0.5438180
3: -12.5080175, -11.2808914, -12.5556955, -11.2228880, -0.7486691, 0.7222140
4: -2.1697092, -1.1134973, -2.1958218, -1.1087556, -0.7431192, 0.7632868
5: -9.8783884, -8.8916836, -9.8890390, -8.8325434, -0.5754423, 0.5398194
6: -7.8049889, -6.6457181, -7.8518467, -6.6005287, -0.8077741, 0.8108070
7: -2.6601472, -2.0528264, -2.6675804, -2.0493186, -0.3758363, 0.3807703
8: -3.6220851, -2.6548738, -3.6838260, -2.6387863, -0.6034980, 0.6377091
9: -12.2878571, -11.2276592, -12.3001442, -11.1999998, -0.7281857, 0.7145025

Time for backsubstitution: 21.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2837314, upper bound: 0.2749487
time: 3.48 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2840666, upper bound: 0.2791836
time: 3.96 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -12.2566080, -10.6660843, -12.2584105, -10.6638117, -1.1111836, 1.1214731
1: 3.3941898, 4.2790651, 3.3878927, 4.2796965, -0.5133584, 0.5189395
2: -4.7570276, -3.9528489, -4.7581620, -3.9497952, -0.5435634, 0.5402981
3: -12.5672646, -11.2399197, -12.5681667, -11.2307320, -0.7375827, 0.7508308
4: -2.1786344, -1.1251500, -2.1798718, -1.1166775, -0.7526031, 0.7457161
5: -9.8548946, -8.8744822, -9.8751106, -8.8739595, -0.5381935, 0.5545886
6: -7.8242092, -6.6131763, -7.8397269, -6.6124725, -0.8193583, 0.8355901
7: -2.6614094, -2.0566208, -2.6614099, -2.0524938, -0.3743212, 0.3709511
8: -3.6503291, -2.6554446, -3.6517234, -2.6394858, -0.6204295, 0.6179429
9: -12.2851868, -11.2112608, -12.2942381, -11.2106705, -0.7159369, 0.7236654

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2807227, upper bound: 0.2801554
time: 3.62 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2809248, upper bound: 0.2842732
time: 3.57 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -12.2632513, -10.6611443, -12.2602682, -10.6615772, -1.1196437, 1.1286740
1: 3.3802838, 4.2914314, 3.3816528, 4.2802687, -0.5216486, 0.5290189
2: -4.7639036, -3.9452162, -4.7594156, -3.9468031, -0.5504084, 0.5486234
3: -12.5861092, -11.2193708, -12.5689688, -11.2200251, -0.7520136, 0.7635917
4: -2.1983888, -1.1079631, -2.1814754, -1.1082895, -0.7710283, 0.7563999
5: -9.8970547, -8.8315611, -9.8950310, -8.8726816, -0.5557435, 0.5815490
6: -7.8568106, -6.5845385, -7.8550673, -6.6118402, -0.8363307, 0.8579116
7: -2.6678946, -2.0473046, -2.6614099, -2.0481217, -0.3859955, 0.3801438
8: -3.6862578, -2.6238065, -3.6533628, -2.6237841, -0.6463482, 0.6362245
9: -12.3055534, -11.1960316, -12.3033476, -11.2095785, -0.7397861, 0.7440538

Time for backsubstitution: 22.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2841155, upper bound: 0.2802059
time: 3.92 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2843162, upper bound: 0.2843170
time: 3.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.45 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.45
Output dim: 1, lower bound: -0.2836863, upper bound: 0.2715559
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.45
Output dim: 1, lower bound: -0.2840237, upper bound: 0.2758090
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.45
Output dim: 1, lower bound: -0.2837314, upper bound: 0.2749487
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.45
Output dim: 1, lower bound: -0.2840666, upper bound: 0.2791836
NS_A2_A1_A1, status: Status.VERIFIED, split count: 3, time: 30.45
Output dim: 1, lower bound: -0.2807227, upper bound: 0.2801554
NS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 30.45
Output dim: 1, lower bound: -0.2809248, upper bound: 0.2842732
NS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 30.45
Output dim: 1, lower bound: -0.2841155, upper bound: 0.2802059
NS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 30.45
Output dim: 1, lower bound: -0.2843162, upper bound: 0.2843170

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -12.1139164, -10.7958794, -12.2430706, -10.7308273, -0.7266929, 1.0173581
1: 3.4255977, 4.2473812, 3.3996143, 4.2639065, -0.4574174, 0.4913988
2: -4.6984801, -4.0049658, -4.7532158, -3.9803894, -0.4137496, 0.4987881
3: -12.4705381, -11.3292208, -12.5187159, -11.2490683, -0.6971374, 0.4862549
4: -2.1436648, -1.1410406, -2.1721296, -1.1347959, -0.6840594, 0.7158257
5: -9.8486919, -8.9024277, -9.8422899, -8.8765297, -0.5393503, 0.4778287
6: -7.7624292, -6.6744337, -7.8140645, -6.6429114, -0.7652428, 0.7809284
7: -2.6488917, -2.0731769, -2.6603112, -2.0670853, -0.3351822, 0.3575202
8: -3.6059823, -2.6868544, -3.6442852, -2.6766577, -0.4280226, 0.5876664
9: -12.2536736, -11.2498646, -12.2766914, -11.2280188, -0.6160963, 0.7064931

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2836650, upper bound: 0.2701082
time: 3.38 seconds

## Relational analysis of NS_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2836806, upper bound: 0.2715487
time: 3.38 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12.1878052, -10.7224722, -12.2517338, -10.6947021, -1.0039666, 1.0604856
1: 3.4088478, 4.2642498, 3.3968048, 4.2719698, -0.4904872, 0.4950845
2: -4.7327046, -3.9715374, -4.7555079, -3.9634724, -0.4960897, 0.5086454
3: -12.5072012, -11.2916050, -12.5368462, -11.2434387, -0.7195463, 0.6890516
4: -2.1681662, -1.1218953, -2.1760960, -1.1259496, -0.7293630, 0.7368917
5: -9.8584633, -8.8929520, -9.8468666, -8.8754578, -0.5433753, 0.5232779
6: -7.7897000, -6.6463723, -7.8192253, -6.6291761, -0.7700398, 0.7837776
7: -2.6601465, -2.0572062, -2.6610947, -2.0586381, -0.3646934, 0.3533508
8: -3.6204433, -2.6705856, -3.6478963, -2.6704321, -0.5796303, 0.5904334
9: -12.2787476, -11.2287598, -12.2797794, -11.2152262, -0.7020011, 0.6834872

Time for backsubstitution: 21.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2825737, upper bound: 0.2757801
time: 3.58 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2840164, upper bound: 0.2758019
time: 4.99 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -12.1157694, -10.7936630, -12.2497082, -10.7258949, -0.7339171, 1.0279868
1: 3.4194040, 4.2479515, 3.3857260, 4.2762709, -0.4706367, 0.4996090
2: -4.6997337, -4.0019684, -4.7600908, -3.9727421, -0.4220914, 0.5083156
3: -12.4713392, -11.3185081, -12.5375595, -11.2285156, -0.7098551, 0.5006738
4: -2.1452112, -1.1326528, -2.1918650, -1.1176085, -0.6946096, 0.7380345
5: -9.8686113, -8.9011555, -9.8844633, -8.8336105, -0.5663128, 0.4954574
6: -7.7777271, -6.6738014, -7.8466558, -6.6142740, -0.7919440, 0.7978156
7: -2.6488917, -2.0688152, -2.6667967, -2.0577617, -0.3444036, 0.3679038
8: -3.6076226, -2.6711512, -3.6802139, -2.6450181, -0.4463028, 0.6135887
9: -12.2627811, -11.2487993, -12.2970572, -11.2128115, -0.6424398, 0.7303095

Time for backsubstitution: 21.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2837087, upper bound: 0.2735010
time: 3.54 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2837230, upper bound: 0.2749413
time: 3.59 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12.1896610, -10.7202396, -12.2583742, -10.6897526, -1.0111799, 1.0689466
1: 3.4026546, 4.2648211, 3.3828979, 4.2843361, -0.5005616, 0.5033033
2: -4.7339582, -3.9685333, -4.7623825, -3.9558258, -0.5044305, 0.5154991
3: -12.5080051, -11.2808971, -12.5556889, -11.2228889, -0.7323161, 0.7034690
4: -2.1697049, -1.1135068, -2.1958206, -1.1087627, -0.7400820, 0.7540686
5: -9.8783855, -8.8916836, -9.8890362, -8.8325443, -0.5703455, 0.5409315
6: -7.8049812, -6.6457391, -7.8518429, -6.6005387, -0.7923181, 0.8005921
7: -2.6601465, -2.0528347, -2.6675799, -2.0493228, -0.3720895, 0.3650302
8: -3.6220818, -2.6548839, -3.6838241, -2.6387930, -0.5979112, 0.6163540
9: -12.2878561, -11.2276821, -12.3001461, -11.2000132, -0.7195337, 0.7073314

Time for backsubstitution: 22.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2825991, upper bound: 0.2791531
time: 4.20 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2840581, upper bound: 0.2791766
time: 4.37 seconds

## BFS NS instance: NS_A2_A1_A2

### Backsubstitution after applying NS history:
0: -12.2566051, -10.6661549, -12.2584114, -10.6638432, -1.0644572, 1.0850046
1: 3.3941913, 4.2790585, 3.3878937, 4.2796946, -0.5050460, 0.5083162
2: -4.7570262, -3.9528551, -4.7581615, -3.9497995, -0.5195295, 0.5106049
3: -12.5672483, -11.2399216, -12.5681610, -11.2307329, -0.7197280, 0.7337148
4: -2.1786308, -1.1251636, -2.1798701, -1.1166854, -0.7450757, 0.7393470
5: -9.8548927, -8.8744831, -9.8751097, -8.8739595, -0.5339811, 0.5533514
6: -7.8242035, -6.6131930, -7.8397245, -6.6124821, -0.8064742, 0.8207603
7: -2.6614094, -2.0566285, -2.6614101, -2.0524971, -0.3705465, 0.3553587
8: -3.6503267, -2.6554565, -3.6517220, -2.6394911, -0.6167912, 0.5969734
9: -12.2851849, -11.2112856, -12.2942390, -11.2106838, -0.7130272, 0.7146440

Time for backsubstitution: 21.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of NS_A2_A1_A2_B1

### Relational analysis result of NS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2794725, upper bound: 0.2842522
time: 3.81 seconds

## Relational analysis of NS_A2_A1_A2_B2

### Relational analysis result of NS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2809161, upper bound: 0.2842661
time: 3.68 seconds

## BFS NS instance: NS_A2_A2_A1

### Backsubstitution after applying NS history:
0: -12.1866245, -10.7345533, -12.2520771, -10.6977425, -0.9992239, 1.0491903
1: 3.3988495, 4.2742343, 3.3842854, 4.2721243, -0.4911129, 0.5062306
2: -4.7288723, -3.9790814, -4.7572727, -3.9638100, -0.4921839, 0.5121574
3: -12.5495729, -11.2586641, -12.5508242, -11.2253799, -0.7124580, 0.6960104
4: -2.1719832, -1.1271424, -2.1774120, -1.1171498, -0.7250583, 0.7289462
5: -9.8867159, -8.8416224, -9.8902464, -8.8737116, -0.5430648, 0.5641828
6: -7.8259196, -6.6135402, -7.8502817, -6.6256852, -0.7807240, 0.8141868
7: -2.6567492, -2.0650978, -2.6606452, -2.0565522, -0.3626738, 0.3605559
8: -3.6701851, -2.6396933, -3.6499910, -2.6300211, -0.6159339, 0.6114779
9: -12.2791710, -11.2217903, -12.3001022, -11.2224054, -0.7006896, 0.7138393

Time for backsubstitution: 22.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of NS_A2_A2_A1_B1

### Relational analysis result of NS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2826639, upper bound: 0.2801845
time: 3.79 seconds

## Relational analysis of NS_A2_A2_A1_B2

### Relational analysis result of NS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2841082, upper bound: 0.2801988
time: 3.74 seconds

## BFS NS instance: NS_A2_A2_A2

### Backsubstitution after applying NS history:
0: -12.2632465, -10.6612148, -12.2602673, -10.6616116, -1.0729222, 1.0922065
1: 3.3802862, 4.2914257, 3.3816547, 4.2802658, -0.5133455, 0.5184017
2: -4.7639022, -3.9452236, -4.7594147, -3.9468064, -0.5263743, 0.5189302
3: -12.5860939, -11.2193737, -12.5689621, -11.2200270, -0.7341595, 0.7464806
4: -2.1983857, -1.1079762, -2.1814742, -1.1082969, -0.7622201, 0.7501032
5: -9.8970528, -8.8315620, -9.8950291, -8.8726826, -0.5516498, 0.5803168
6: -7.8568048, -6.5845556, -7.8550653, -6.6118484, -0.8234661, 0.8430929
7: -2.6678944, -2.0473125, -2.6614101, -2.0481243, -0.3773714, 0.3645513
8: -3.6862569, -2.6238165, -3.6533608, -2.6237888, -0.6427114, 0.6152555
9: -12.3055515, -11.1960583, -12.3033476, -11.2095909, -0.7318957, 0.7320440

Time for backsubstitution: 22.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of NS_A2_A2_A2_B1

### Relational analysis result of NS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2828653, upper bound: 0.2842960
time: 3.74 seconds

## Relational analysis of NS_A2_A2_A2_B2

### Relational analysis result of NS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2843089, upper bound: 0.2843099
time: 3.50 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.04 seconds
NS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 30.04
Output dim: 1, lower bound: -0.2836650, upper bound: 0.2701082
NS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 30.04
Output dim: 1, lower bound: -0.2836806, upper bound: 0.2715487
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.04
Output dim: 1, lower bound: -0.2825737, upper bound: 0.2757801
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.04
Output dim: 1, lower bound: -0.2840164, upper bound: 0.2758019
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 30.04
Output dim: 1, lower bound: -0.2837087, upper bound: 0.2735010
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 30.04
Output dim: 1, lower bound: -0.2837230, upper bound: 0.2749413
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.04
Output dim: 1, lower bound: -0.2825991, upper bound: 0.2791531
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.04
Output dim: 1, lower bound: -0.2840581, upper bound: 0.2791766
NS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.04
Output dim: 1, lower bound: -0.2794725, upper bound: 0.2842522
NS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.04
Output dim: 1, lower bound: -0.2809161, upper bound: 0.2842661
NS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.04
Output dim: 1, lower bound: -0.2826639, upper bound: 0.2801845
NS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.04
Output dim: 1, lower bound: -0.2841082, upper bound: 0.2801988
NS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.04
Output dim: 1, lower bound: -0.2828653, upper bound: 0.2842960
NS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.04
Output dim: 1, lower bound: -0.2843089, upper bound: 0.2843099

## BFS NS instance: NS_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -12.1090069, -10.7996826, -12.2425022, -10.7322159, -0.7198827, 1.0131168
1: 3.4319305, 4.2388573, 3.4000664, 4.2601566, -0.4470374, 0.4825011
2: -4.6789622, -4.0187311, -4.7437520, -3.9811699, -0.3934482, 0.4750534
3: -12.4672499, -11.3320637, -12.5183420, -11.2503624, -0.6913185, 0.4813432
4: -2.1271625, -1.1501653, -2.1647129, -1.1348729, -0.6675541, 0.6985457
5: -9.8422823, -8.9081907, -9.8409100, -8.8793392, -0.5292689, 0.4708087
6: -7.7501574, -6.6894541, -7.8132806, -6.6503530, -0.7455571, 0.7650015
7: -2.6484711, -2.0767257, -2.6603112, -2.0679057, -0.3301179, 0.3513557
8: -3.6033297, -2.6881995, -3.6437473, -2.6770811, -0.4232069, 0.5833017
9: -12.2453108, -11.2563610, -12.2731819, -11.2287426, -0.6073368, 0.6945040

Time for backsubstitution: 22.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of NS_A1_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2803185, upper bound: 0.2701083
time: 3.50 seconds

## Relational analysis of NS_A1_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2803185, upper bound: 0.2701075
time: 3.41 seconds

## BFS NS instance: NS_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -12.1139183, -10.7958813, -12.2430696, -10.7308273, -0.7243577, 1.0172050
1: 3.4255991, 4.2473650, 3.3996143, 4.2639003, -0.4555509, 0.4868398
2: -4.6984453, -4.0049696, -4.7531986, -3.9803908, -0.3921340, 0.4911894
3: -12.4705381, -11.3292236, -12.5187149, -11.2490711, -0.6943984, 0.4845737
4: -2.1436343, -1.1410401, -2.1721153, -1.1347964, -0.6748097, 0.7122421
5: -9.8486881, -8.9024343, -9.8422880, -8.8765326, -0.5363278, 0.4720076
6: -7.7624283, -6.6744490, -7.8140631, -6.6429181, -0.7603757, 0.7639413
7: -2.6488917, -2.0731788, -2.6603112, -2.0670860, -0.3374442, 0.3531132
8: -3.6059809, -2.6868548, -3.6442847, -2.6766572, -0.4236213, 0.5867217
9: -12.2536602, -11.2498665, -12.2766867, -11.2280197, -0.6157033, 0.7081158

Time for backsubstitution: 22.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of NS_A1_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2803314, upper bound: 0.2715486
time: 3.51 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2803314, upper bound: 0.2715486
time: 3.51 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.1872091, -10.7238445, -12.2468929, -10.6984854, -0.9996731, 1.0539584
1: 3.4093008, 4.2605152, 3.4031558, 4.2634115, -0.4816022, 0.4833524
2: -4.7232547, -3.9723082, -4.7359405, -3.9772396, -0.4685919, 0.4884169
3: -12.5067978, -11.2928963, -12.5335979, -11.2461472, -0.7147117, 0.6832855
4: -2.1607447, -1.1219764, -2.1596735, -1.1350403, -0.7098382, 0.7200401
5: -9.8570833, -8.8957624, -9.8404655, -8.8811321, -0.5364301, 0.5131774
6: -7.7889152, -6.6538053, -7.8069315, -6.6442485, -0.7540278, 0.7640326
7: -2.6601465, -2.0580397, -2.6606748, -2.0621917, -0.3585265, 0.3482515
8: -3.6198511, -2.6710062, -3.6454215, -2.6717682, -0.5751908, 0.5858189
9: -12.2752867, -11.2294865, -12.2713881, -11.2216787, -0.6896546, 0.6747497

Time for backsubstitution: 23.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 4599

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2780106, upper bound: 0.2757800
time: 3.99 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2780106, upper bound: 0.2757800
time: 3.43 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.1878033, -10.7224741, -12.2517290, -10.6947031, -1.0022528, 1.0582199
1: 3.4088488, 4.2642431, 3.3968034, 4.2719507, -0.4842232, 0.4899789
2: -4.7326880, -3.9715395, -4.7554722, -3.9634764, -0.4841206, 0.4870303
3: -12.5072002, -11.2916050, -12.5368443, -11.2434416, -0.7177948, 0.6844860
4: -2.1681528, -1.1218956, -2.1760662, -1.1259496, -0.7207606, 0.7271714
5: -9.8584604, -8.8929539, -9.8468628, -8.8754635, -0.5374017, 0.5202870
6: -7.7897005, -6.6463780, -7.8192239, -6.6291914, -0.7521882, 0.7799897
7: -2.6601465, -2.0572071, -2.6610947, -2.0586405, -0.3588504, 0.3555446
8: -3.6204405, -2.6705875, -3.6478958, -2.6704335, -0.5785521, 0.5860286
9: -12.2787418, -11.2287617, -12.2797680, -11.2152300, -0.6984110, 0.6830935

Time for backsubstitution: 22.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 4599

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of NS_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2794682, upper bound: 0.2758016
time: 4.45 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2794682, upper bound: 0.2758032
time: 3.55 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -12.1108150, -10.7974644, -12.2490749, -10.7272778, -0.7270010, 1.0236859
1: 3.4257536, 4.2394638, 3.3861656, 4.2725153, -0.4588610, 0.4907517
2: -4.6802855, -4.0157199, -4.7506227, -3.9735010, -0.4018162, 0.4825469
3: -12.4679890, -11.3213387, -12.5371380, -11.2298069, -0.7039781, 0.4957523
4: -2.1287403, -1.1418054, -2.1844225, -1.1176829, -0.6781297, 0.7184322
5: -9.8622236, -8.9069080, -9.8830996, -8.8364182, -0.5562564, 0.4884689
6: -7.7654419, -6.6887755, -7.8458891, -6.6217194, -0.7691848, 0.7819490
7: -2.6484711, -2.0724070, -2.6667967, -2.0585780, -0.3393161, 0.3617317
8: -3.6048403, -2.6724963, -3.6795902, -2.6454411, -0.4413785, 0.6091937
9: -12.2544870, -11.2552938, -12.2936506, -11.2135410, -0.6336820, 0.7183084

Time for backsubstitution: 22.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of NS_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2791596, upper bound: 0.2735011
time: 3.37 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2791583, upper bound: 0.2735009
time: 3.54 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -12.1157694, -10.7936659, -12.2497063, -10.7258940, -0.7315263, 1.0267050
1: 3.4194064, 4.2479343, 3.3857269, 4.2762632, -0.4655274, 0.4950503
2: -4.6996984, -4.0019708, -4.7600746, -3.9727430, -0.4004815, 0.4980367
3: -12.4713392, -11.3185129, -12.5375595, -11.2285156, -0.7071707, 0.4989033
4: -2.1451814, -1.1326523, -2.1918507, -1.1176085, -0.6854315, 0.7293777
5: -9.8686047, -8.9011602, -9.8844585, -8.8336124, -0.5632858, 0.4896361
6: -7.7777247, -6.6738148, -7.8466554, -6.6142807, -0.7824767, 0.7808282
7: -2.6488917, -2.0688174, -2.6667967, -2.0577624, -0.3466046, 0.3620607
8: -3.6076207, -2.6711526, -3.6802125, -2.6450171, -0.4419013, 0.6125021
9: -12.2627697, -11.2488012, -12.2970524, -11.2128115, -0.6411561, 0.7319310

Time for backsubstitution: 22.80 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.91 + 565.49 = 622.40 seconds
