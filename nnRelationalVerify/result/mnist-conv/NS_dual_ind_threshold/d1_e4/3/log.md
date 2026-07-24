## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.10417916599999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4604456, 0.4604454)
1: (-8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2606165, 0.2606165)
2: (-4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2777116, 0.2777117)
3: (-4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3350914, 0.3350914)
4: (-8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3183744, 0.3183744)
5: (-15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5240293, 0.5240293)
6: (-22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4546742, 0.4546742)
7: (4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2212260, 0.2212260)
8: (-4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.3066764, 0.3066764)
9: (-4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2763367, 0.2763366)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.39 + 33.18 = 55.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.1108283, upper bound: 0.1108289

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 904

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1107001, upper bound: 0.1094643
time: 3.40 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1107001, upper bound: 0.1107008
time: 2.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.57 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.57
Output dim: 7, lower bound: -0.1107001, upper bound: 0.1094643
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.57
Output dim: 7, lower bound: -0.1107001, upper bound: 0.1107008

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -13.0324125, -12.0826511, -13.0357113, -12.0798464, -0.4532557, 0.4535351
1: -8.4532290, -7.8945198, -8.4536114, -7.8952045, -0.2597345, 0.2598493
2: -4.0297647, -3.3972161, -4.0309372, -3.3960037, -0.2763381, 0.2770736
3: -4.8028970, -4.2072115, -4.8039036, -4.2058506, -0.3326650, 0.3323746
4: -8.3131504, -7.6476402, -8.3141441, -7.6465797, -0.3198252, 0.3176122
5: -15.6990194, -14.9548092, -15.6995096, -14.9546223, -0.5211120, 0.5236111
6: -22.8590050, -21.7859859, -22.8616600, -21.7833405, -0.4487371, 0.4486139
7: 4.4987459, 4.8805375, 4.4976039, 4.8814354, -0.2197959, 0.2203978
8: -4.7669106, -4.1273522, -4.7679429, -4.1263285, -0.3055356, 0.3049474
9: -4.2528625, -3.6958575, -4.2535019, -3.6953635, -0.2754242, 0.2755851

Time for backsubstitution: 20.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 904

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 904

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1094635, upper bound: 0.1094643
time: 3.02 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1094635, upper bound: 0.1094643
time: 3.03 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -13.0361252, -12.0760212, -13.0361261, -12.0760164, -0.4604437, 0.4553049
1: -8.4536018, -7.8950758, -8.4536905, -7.8950763, -0.2602055, 0.2604696
2: -4.0309591, -3.3945796, -4.0309601, -3.3945782, -0.2776659, 0.2771459
3: -4.8052750, -4.2055774, -4.8052750, -4.2055769, -0.3336754, 0.3350902
4: -8.3141785, -7.6451302, -8.3141785, -7.6451292, -0.3175130, 0.3214939
5: -15.7004156, -14.9545279, -15.7004166, -14.9545145, -0.5269356, 0.5234733
6: -22.8620071, -21.7799358, -22.8620071, -21.7799339, -0.4546719, 0.4499736
7: 4.4975181, 4.8824654, 4.4975195, 4.8824663, -0.2211027, 0.2207065
8: -4.7690182, -4.1261396, -4.7690187, -4.1261411, -0.3058214, 0.3066750
9: -4.2537293, -3.6950598, -4.2537303, -3.6950593, -0.2762537, 0.2762381

Time for backsubstitution: 20.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 904

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 904

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1094635, upper bound: 0.1107009
time: 3.18 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1094635, upper bound: 0.1107009
time: 2.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.27 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.27
Output dim: 7, lower bound: -0.1094635, upper bound: 0.1094643
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.27
Output dim: 7, lower bound: -0.1094635, upper bound: 0.1094643
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.27
Output dim: 7, lower bound: -0.1094635, upper bound: 0.1107009
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.27
Output dim: 7, lower bound: -0.1094635, upper bound: 0.1107009

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -13.0324125, -12.0826511, -13.0324125, -12.0826511, -0.4503736, 0.4503739
1: -8.4532290, -7.8945198, -8.4532290, -7.8945198, -0.2592418, 0.2592418
2: -4.0297647, -3.3972161, -4.0297647, -3.3972161, -0.2758919, 0.2758918
3: -4.8028970, -4.2072115, -4.8028970, -4.2072115, -0.3313892, 0.3313892
4: -8.3131504, -7.6476402, -8.3131504, -7.6476402, -0.3192511, 0.3192511
5: -15.6990194, -14.9548092, -15.6990194, -14.9548092, -0.5223398, 0.5223398
6: -22.8590050, -21.7859859, -22.8590050, -21.7859859, -0.4461508, 0.4461508
7: 4.4987459, 4.8805375, 4.4987459, 4.8805375, -0.2192304, 0.2192305
8: -4.7669106, -4.1273522, -4.7669106, -4.1273522, -0.3045256, 0.3045256
9: -4.2528625, -3.6958575, -4.2528625, -3.6958575, -0.2750056, 0.2750056

Time for backsubstitution: 21.69 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1832
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1070440, upper bound: 0.1050298
time: 3.23 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1073598, upper bound: 0.1062392
time: 3.23 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -13.0324125, -12.0826511, -13.0361252, -12.0760212, -0.4571977, 0.4536176
1: -8.4532290, -7.8945198, -8.4536018, -7.8950758, -0.2598813, 0.2597125
2: -4.0297647, -3.3972161, -4.0309591, -3.3945796, -0.2764637, 0.2770928
3: -4.8028970, -4.2072115, -4.8052750, -4.2055774, -0.3327575, 0.3337212
4: -8.3131504, -7.6476402, -8.3141785, -7.6451302, -0.3187034, 0.3174314
5: -15.6990194, -14.9548092, -15.7004156, -14.9545279, -0.5227017, 0.5225649
6: -22.8590050, -21.7859859, -22.8620071, -21.7799358, -0.4517689, 0.4486778
7: 4.4987459, 4.8805375, 4.4975181, 4.8824654, -0.2198724, 0.2204602
8: -4.7669106, -4.1273522, -4.7690182, -4.1261396, -0.3057010, 0.3054988
9: -4.2528625, -3.6958575, -4.2537293, -3.6950598, -0.2754245, 0.2757528

Time for backsubstitution: 21.70 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1832
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1070440, upper bound: 0.1050291
time: 4.36 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1073598, upper bound: 0.1062392
time: 3.03 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -13.0361252, -12.0760212, -13.0324125, -12.0826511, -0.4536176, 0.4571977
1: -8.4536018, -7.8950758, -8.4532290, -7.8945198, -0.2597125, 0.2598813
2: -4.0309591, -3.3945796, -4.0297647, -3.3972161, -0.2770928, 0.2764637
3: -4.8052750, -4.2055774, -4.8028970, -4.2072115, -0.3337212, 0.3327575
4: -8.3141785, -7.6451302, -8.3131504, -7.6476402, -0.3174314, 0.3187034
5: -15.7004156, -14.9545279, -15.6990194, -14.9548092, -0.5225649, 0.5227017
6: -22.8620071, -21.7799358, -22.8590050, -21.7859859, -0.4486775, 0.4517689
7: 4.4975181, 4.8824654, 4.4987459, 4.8805375, -0.2204602, 0.2198722
8: -4.7690182, -4.1261396, -4.7669106, -4.1273522, -0.3054988, 0.3057010
9: -4.2537293, -3.6950598, -4.2528625, -3.6958575, -0.2757525, 0.2754244

Time for backsubstitution: 21.74 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1832
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2459
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1059130, upper bound: 0.1061458
time: 4.39 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1062387, upper bound: 0.1075904
time: 3.13 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -13.0361252, -12.0760212, -13.0361252, -12.0760212, -0.4553056, 0.4553056
1: -8.4536018, -7.8950758, -8.4536018, -7.8950758, -0.2602055, 0.2602055
2: -4.0309591, -3.3945796, -4.0309591, -3.3945796, -0.2771460, 0.2771460
3: -4.8052750, -4.2055774, -4.8052750, -4.2055774, -0.3336754, 0.3336754
4: -8.3141785, -7.6451302, -8.3141785, -7.6451302, -0.3214936, 0.3214936
5: -15.7004156, -14.9545279, -15.7004156, -14.9545279, -0.5269351, 0.5269351
6: -22.8620071, -21.7799358, -22.8620071, -21.7799358, -0.4499731, 0.4499731
7: 4.4975181, 4.8824654, 4.4975181, 4.8824654, -0.2206823, 0.2206824
8: -4.7690182, -4.1261396, -4.7690182, -4.1261396, -0.3058217, 0.3058217
9: -4.2537293, -3.6950598, -4.2537293, -3.6950598, -0.2762377, 0.2762377

Time for backsubstitution: 21.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1832
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2459
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1059130, upper bound: 0.1061467
time: 3.45 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1062387, upper bound: 0.1075906
time: 3.08 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.56 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1070440, upper bound: 0.1050298
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1073598, upper bound: 0.1062392
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1070440, upper bound: 0.1050291
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1073598, upper bound: 0.1062392
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1059130, upper bound: 0.1061458
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1062387, upper bound: 0.1075904
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1059130, upper bound: 0.1061467
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1062387, upper bound: 0.1075906

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -13.0277786, -12.0707388, -13.0291252, -12.0826759, -0.4413831, 0.4511294
1: -8.4468346, -7.8886213, -8.4492884, -7.8946977, -0.2540601, 0.2615929
2: -4.0260944, -3.3935127, -4.0276346, -3.3977482, -0.2698603, 0.2736516
3: -4.7944183, -4.2070084, -4.7981024, -4.2076893, -0.3209543, 0.3246450
4: -8.2992392, -7.6290102, -8.3049078, -7.6481628, -0.2967777, 0.3021507
5: -15.6913891, -14.9457169, -15.6945972, -14.9549446, -0.5125475, 0.5161414
6: -22.8546581, -21.8061256, -22.8584061, -21.7972374, -0.4356360, 0.4283681
7: 4.4920907, 4.8558517, 4.5005121, 4.8657112, -0.1814547, 0.1847641
8: -4.7552099, -4.1214147, -4.7598205, -4.1274519, -0.2859809, 0.2892828
9: -4.2426348, -3.6859026, -4.2459435, -3.6974721, -0.2503979, 0.2512246

Time for backsubstitution: 20.97 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 900

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 1096

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1050203, upper bound: 0.1039252
time: 3.61 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1050282, upper bound: 0.1039185
time: 3.61 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -13.0266285, -12.0826645, -13.0299358, -12.0826569, -0.4474826, 0.4472675
1: -8.4491444, -7.8945880, -8.4514732, -7.8945494, -0.2602876, 0.2573904
2: -4.0271688, -3.3975377, -4.0286541, -3.3973565, -0.2740088, 0.2737510
3: -4.8013000, -4.2074032, -4.8022137, -4.2072940, -0.3221192, 0.3309166
4: -8.3044701, -7.6478500, -8.3093853, -7.6477299, -0.2924488, 0.3176346
5: -15.6949501, -14.9548721, -15.6972752, -14.9548359, -0.5131655, 0.5187554
6: -22.8587761, -21.7879257, -22.8589077, -21.7868137, -0.4454315, 0.4325950
7: 4.4998198, 4.8752055, 4.4992113, 4.8781052, -0.2161709, 0.1759447
8: -4.7625809, -4.1273894, -4.7650514, -4.1273699, -0.2838581, 0.3037181
9: -4.2462454, -3.6968398, -4.2500305, -3.6962826, -0.2395949, 0.2717936

Time for backsubstitution: 21.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 1984

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 766

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1059664, upper bound: 0.1071522
time: 3.55 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1059664, upper bound: 0.1074539
time: 3.29 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -13.0277786, -12.0707388, -13.0328350, -12.0760460, -0.4482071, 0.4543657
1: -8.4468346, -7.8886213, -8.4496565, -7.8952551, -0.2547035, 0.2620374
2: -4.0260944, -3.3935127, -4.0288320, -3.3951061, -0.2704871, 0.2748586
3: -4.7944183, -4.2070084, -4.8004818, -4.2060466, -0.3223252, 0.3269815
4: -8.2992392, -7.6290102, -8.3059502, -7.6456532, -0.2962325, 0.3008482
5: -15.6913891, -14.9457169, -15.6960049, -14.9546642, -0.5129080, 0.5163722
6: -22.8546581, -21.8061256, -22.8614197, -21.7911739, -0.4414899, 0.4308968
7: 4.4920907, 4.8558517, 4.4992876, 4.8676300, -0.1813698, 0.1859865
8: -4.7552099, -4.1214147, -4.7619395, -4.1262350, -0.2871578, 0.2899854
9: -4.2426348, -3.6859026, -4.2468300, -3.6966512, -0.2508049, 0.2519712

Time for backsubstitution: 21.72 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 900

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 1096

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051331, upper bound: 0.1029115
time: 3.34 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051381, upper bound: 0.1029032
time: 6.00 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -13.0266285, -12.0826645, -13.0336466, -12.0760269, -0.4543035, 0.4505107
1: -8.4491444, -7.8945880, -8.4518423, -7.8951054, -0.2609217, 0.2579732
2: -4.0271688, -3.3975377, -4.0298462, -3.3947177, -0.2750890, 0.2749510
3: -4.8013000, -4.2074032, -4.8045907, -4.2056570, -0.3234882, 0.3332496
4: -8.3044701, -7.6478500, -8.3104076, -7.6452203, -0.2919176, 0.3158169
5: -15.6949501, -14.9548721, -15.6986675, -14.9545555, -0.5135527, 0.5189776
6: -22.8587761, -21.7879257, -22.8619137, -21.7807655, -0.4510155, 0.4350910
7: 4.4998198, 4.8752055, 4.4979877, 4.8800340, -0.2170478, 0.1771755
8: -4.7625809, -4.1273894, -4.7671633, -4.1261554, -0.2850223, 0.3048275
9: -4.2462454, -3.6968398, -4.2508945, -3.6954799, -0.2400159, 0.2725351

Time for backsubstitution: 21.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1984

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 766

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1060722, upper bound: 0.1059137
time: 3.49 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1060722, upper bound: 0.1062394
time: 3.47 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -13.0314970, -12.0641174, -13.0291271, -12.0826740, -0.4446082, 0.4579787
1: -8.4471798, -7.8891735, -8.4492931, -7.8946981, -0.2544925, 0.2622250
2: -4.0272875, -3.3908558, -4.0276370, -3.3977470, -0.2710650, 0.2748357
3: -4.7967834, -4.2053642, -4.7981100, -4.2076902, -0.3232741, 0.3260086
4: -8.3002672, -7.6264501, -8.3049183, -7.6481619, -0.2960179, 0.3016202
5: -15.6927719, -14.9454441, -15.6946011, -14.9549437, -0.5127611, 0.5165458
6: -22.8576717, -21.8000908, -22.8584080, -21.7972260, -0.4381366, 0.4340801
7: 4.4908924, 4.8576980, 4.5005102, 4.8657298, -0.1826631, 0.1845970
8: -4.7573090, -4.1202059, -4.7598290, -4.1274524, -0.2864280, 0.2904491
9: -4.2435203, -3.6850355, -4.2459531, -3.6974697, -0.2511134, 0.2516267

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 900

Time for candidate selection: 0.44 seconds

### Candidate
type: B, layer: 3, pos: 1096

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1037820, upper bound: 0.1040262
time: 3.56 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1037878, upper bound: 0.1040180
time: 4.30 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -13.0303297, -12.0760355, -13.0299482, -12.0826569, -0.4506946, 0.4541035
1: -8.4495087, -7.8951445, -8.4514828, -7.8945484, -0.2607194, 0.2580403
2: -4.0283575, -3.3948984, -4.0286598, -3.3973546, -0.2752075, 0.2743949
3: -4.8036776, -4.2057662, -4.8022156, -4.2072930, -0.3244507, 0.3322883
4: -8.3054790, -7.6453404, -8.3094044, -7.6477308, -0.2915671, 0.3170955
5: -15.6963329, -14.9545918, -15.6972847, -14.9548349, -0.5133781, 0.5191259
6: -22.8617802, -21.7818718, -22.8589077, -21.7868118, -0.4479625, 0.4384346
7: 4.4985991, 4.8771300, 4.4992075, 4.8781157, -0.2173883, 0.1757460
8: -4.7646918, -4.1261773, -4.7650623, -4.1273680, -0.2843163, 0.3048983
9: -4.2470994, -3.6960316, -4.2500443, -3.6962810, -0.2403009, 0.2721996

Time for backsubstitution: 21.46 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 1984

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 766

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1048728, upper bound: 0.1072643
time: 3.58 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1048728, upper bound: 0.1075906
time: 3.43 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -13.0314970, -12.0641174, -13.0328312, -12.0760450, -0.4462931, 0.4560640
1: -8.4471798, -7.8891735, -8.4496479, -7.8952551, -0.2550168, 0.2625405
2: -4.0272875, -3.3908558, -4.0288281, -3.3951063, -0.2711142, 0.2749069
3: -4.7967834, -4.2053642, -4.8004723, -4.2060471, -0.3232303, 0.3269114
4: -8.3002672, -7.6264501, -8.3059349, -7.6456532, -0.2989862, 0.3043694
5: -15.6927719, -14.9454441, -15.6959953, -14.9546652, -0.5171251, 0.5207658
6: -22.8576717, -21.8000908, -22.8614178, -21.7911968, -0.4394057, 0.4321704
7: 4.4908924, 4.8576980, 4.4992914, 4.8676000, -0.1829020, 0.1861506
8: -4.7573090, -4.1202059, -4.7619243, -4.1262374, -0.2872677, 0.2905530
9: -4.2435203, -3.6850355, -4.2468176, -3.6966543, -0.2515817, 0.2523925

Time for backsubstitution: 20.89 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 900

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 1096

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1037820, upper bound: 0.1040253
time: 3.98 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1037878, upper bound: 0.1040180
time: 4.42 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -13.0303297, -12.0760355, -13.0336437, -12.0760269, -0.4524121, 0.4521940
1: -8.4495087, -7.8951445, -8.4518394, -7.8951049, -0.2612323, 0.2583544
2: -4.0283575, -3.3948984, -4.0298443, -3.3947182, -0.2752628, 0.2749993
3: -4.8036776, -4.2057662, -4.8045902, -4.2056580, -0.3243935, 0.3332038
4: -8.3054790, -7.6453404, -8.3104019, -7.6452179, -0.2946527, 0.3198774
5: -15.6963329, -14.9545918, -15.6986647, -14.9545555, -0.5178051, 0.5233445
6: -22.8617802, -21.7818718, -22.8619118, -21.7807655, -0.4492550, 0.4363596
7: 4.4985991, 4.8771300, 4.4979877, 4.8800302, -0.2176002, 0.1773173
8: -4.7646918, -4.1261773, -4.7671609, -4.1261549, -0.2851353, 0.3050075
9: -4.2470994, -3.6960316, -4.2508912, -3.6954813, -0.2407757, 0.2730030

Time for backsubstitution: 20.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1984

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 766

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1048728, upper bound: 0.1072645
time: 3.31 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1048728, upper bound: 0.1072646
time: 3.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.06 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1050203, upper bound: 0.1039252
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1050282, upper bound: 0.1039185
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1059664, upper bound: 0.1071522
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1059664, upper bound: 0.1074539
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1051331, upper bound: 0.1029115
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1051381, upper bound: 0.1029032
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1060722, upper bound: 0.1059137
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1060722, upper bound: 0.1062394
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1037820, upper bound: 0.1040262
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1037878, upper bound: 0.1040180
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1048728, upper bound: 0.1072643
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1048728, upper bound: 0.1075906
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1037820, upper bound: 0.1040253
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1037878, upper bound: 0.1040180
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1048728, upper bound: 0.1072645
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.06
Output dim: 7, lower bound: -0.1048728, upper bound: 0.1072646

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -13.0272846, -12.0729942, -13.0313635, -12.0875549, -0.4373744, 0.4538200
1: -8.4468241, -7.8916531, -8.4532070, -7.9011135, -0.2430853, 0.2585270
2: -4.0256391, -3.3935285, -4.0288153, -3.3972526, -0.2702742, 0.2740446
3: -4.7926779, -4.2074337, -4.7991147, -4.2080998, -0.3188741, 0.3265781
4: -8.2954025, -7.6292086, -8.3048840, -7.6480608, -0.2922938, 0.3075819
5: -15.6907845, -14.9556885, -15.6977482, -14.9756718, -0.4860692, 0.5083752
6: -22.8543129, -21.8143139, -22.8582783, -21.8031273, -0.4263716, 0.4184155
7: 4.4945030, 4.8557038, 4.5039883, 4.8802123, -0.2005496, 0.1790911
8: -4.7511730, -4.1214566, -4.7582998, -4.1274405, -0.2817421, 0.2912800
9: -4.2397366, -3.6859298, -4.2465768, -3.6959133, -0.2492590, 0.2613249

Time for backsubstitution: 20.86 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1832
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2459
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1984
type: A, layer: 3, pos: 558

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1050203, upper bound: 0.1039252
time: 3.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1050203, upper bound: 0.1038493
time: 3.58 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -13.0266638, -12.0741100, -13.0354424, -12.0886946, -0.3298811, 0.4568932
1: -8.4467983, -7.9064798, -8.4480305, -7.9267502, -0.2398763, 0.2865157
2: -4.0231705, -3.3935513, -4.0245767, -3.3980608, -0.2710370, 0.2721344
3: -4.7926154, -4.2080402, -4.7995372, -4.2045565, -0.3216558, 0.2561104
4: -8.2940063, -7.6292443, -8.3036480, -7.6374984, -0.3066368, 0.2518694
5: -15.6906786, -14.9684525, -15.7195530, -14.9952841, -0.2521392, 0.5559387
6: -22.8540154, -21.8173847, -22.8817978, -21.8053017, -0.3372947, 0.4527662
7: 4.4971619, 4.8556051, 4.5081797, 4.8847647, -0.2125293, 0.1503048
8: -4.7502432, -4.1214628, -4.7578616, -4.1183472, -0.2930470, 0.2807204
9: -4.2392220, -3.6860018, -4.2464414, -3.6880698, -0.2404184, 0.2636167

Time for backsubstitution: 20.86 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1832
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2459
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1984
type: A, layer: 3, pos: 558

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1050282, upper bound: 0.1039186
time: 3.35 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1050282, upper bound: 0.1038425
time: 3.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -13.0266285, -12.0826645, -13.0277786, -12.0707388, -0.4514110, 0.4417453
1: -8.4491444, -7.8945880, -8.4468346, -7.8886213, -0.2610699, 0.2541704
2: -4.0271688, -3.3975377, -4.0260944, -3.3935127, -0.2740315, 0.2703011
3: -4.8013000, -4.2074032, -4.7944183, -4.2070084, -0.3293946, 0.3210528
4: -8.3044701, -7.6478500, -8.2992392, -7.6290102, -0.3145719, 0.2969079
5: -15.6949501, -14.9548721, -15.6913891, -14.9457169, -0.5173407, 0.5120463
6: -22.8587761, -21.7879257, -22.8546581, -21.8061256, -0.4284370, 0.4446721
7: 4.4998198, 4.8752055, 4.4920907, 4.8558517, -0.1835177, 0.2033924
8: -4.7625809, -4.1273894, -4.7552099, -4.1214147, -0.2984924, 0.2860181
9: -4.2462454, -3.6968398, -4.2426348, -3.6859026, -0.2638353, 0.2491899

Time for backsubstitution: 20.86 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1832
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1984
type: A, layer: 3, pos: 900

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 1096

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1038485, upper bound: 0.1050208
time: 3.21 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1038418, upper bound: 0.1050278
time: 4.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -13.0266285, -12.0826645, -13.0266285, -12.0826645, -0.4470935, 0.4470937
1: -8.4491444, -7.8945880, -8.4491444, -7.8945880, -0.2602509, 0.2602508
2: -4.0271688, -3.3975377, -4.0271688, -3.3975377, -0.2734402, 0.2734402
3: -4.8013000, -4.2074032, -4.8013000, -4.2074032, -0.3220773, 0.3220773
4: -8.3044701, -7.6478500, -8.3044701, -7.6478500, -0.2923844, 0.2923844
5: -15.6949501, -14.9548721, -15.6949501, -14.9548721, -0.5124669, 0.5124669
6: -22.8587761, -21.7879257, -22.8587761, -21.7879257, -0.4325774, 0.4325771
7: 4.4998198, 4.8752055, 4.4998198, 4.8752055, -0.1751082, 0.1751081
8: -4.7625809, -4.1273894, -4.7625809, -4.1273894, -0.2838476, 0.2838477
9: -4.2462454, -3.6968398, -4.2462454, -3.6968398, -0.2383537, 0.2383537

Time for backsubstitution: 20.93 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1832
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1984
type: A, layer: 3, pos: 900

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1071521, upper bound: 0.1059672
time: 3.21 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1074532, upper bound: 0.1074537
time: 3.17 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -13.0272846, -12.0729942, -13.0350914, -12.0809288, -0.4441984, 0.4570646
1: -8.4468241, -7.8916531, -8.4535828, -7.9016709, -0.2436812, 0.2590013
2: -4.0256391, -3.3935285, -4.0300074, -3.3946147, -0.2708474, 0.2752458
3: -4.7926779, -4.2074337, -4.8014908, -4.2064571, -0.3202446, 0.3289099
4: -8.2954025, -7.6292086, -8.3059111, -7.6455526, -0.2917490, 0.3057611
5: -15.6907845, -14.9556885, -15.6991444, -14.9753799, -0.4864306, 0.5086012
6: -22.8543129, -21.8143139, -22.8612881, -21.7970963, -0.4322398, 0.4209435
7: 4.4945030, 4.8557038, 4.5027637, 4.8821421, -0.2011806, 0.1803187
8: -4.7511730, -4.1214566, -4.7604041, -4.1262283, -0.2829185, 0.2922697
9: -4.2397366, -3.6859298, -4.2474456, -3.6951144, -0.2496786, 0.2620564

Time for backsubstitution: 20.88 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.56 + 545.79 = 601.35 seconds
