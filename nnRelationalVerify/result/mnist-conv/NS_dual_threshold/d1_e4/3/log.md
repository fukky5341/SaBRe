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
execution time: IAR + RelationalAnalysis = 22.60 + 32.67 = 55.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.1108283, upper bound: 0.1108289

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1107001, upper bound: 0.1094643
time: 3.14 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1107001, upper bound: 0.1107008
time: 2.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.27 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.27
Output dim: 7, lower bound: -0.1107001, upper bound: 0.1094643
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.27
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

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1739
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1832
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1984
type: A, layer: 3, pos: 1984
type: A, layer: 3, pos: 1794
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558

Time for candidate selection: 0.47 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1072638, upper bound: 0.1050291
time: 4.22 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1075900, upper bound: 0.1062392
time: 3.17 seconds

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

Time for backsubstitution: 21.77 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1984
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 1794
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1072638, upper bound: 0.1061456
time: 4.20 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1075900, upper bound: 0.1075906
time: 3.16 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.49 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 29.49
Output dim: 7, lower bound: -0.1072638, upper bound: 0.1050291
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 29.49
Output dim: 7, lower bound: -0.1075900, upper bound: 0.1062392
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 29.49
Output dim: 7, lower bound: -0.1072638, upper bound: 0.1061456
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 29.49
Output dim: 7, lower bound: -0.1075900, upper bound: 0.1075906

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -13.0277786, -12.0707388, -13.0324211, -12.0798721, -0.4442649, 0.4542842
1: -8.4468346, -7.8886213, -8.4496689, -7.8953862, -0.2545549, 0.2622075
2: -4.0260944, -3.3935127, -4.0288057, -3.3965323, -0.2703074, 0.2748373
3: -4.7944183, -4.2070084, -4.7991080, -4.2063236, -0.3222318, 0.3256304
4: -8.2992392, -7.6290102, -8.3059092, -7.6471033, -0.2973528, 0.3011751
5: -15.6913891, -14.9457169, -15.6950731, -14.9547558, -0.5113192, 0.5174170
6: -22.8546581, -21.8061256, -22.8610630, -21.7945862, -0.4382200, 0.4308317
7: 4.4920907, 4.8558517, 4.4993672, 4.8666000, -0.1820438, 0.1859336
8: -4.7552099, -4.1214147, -4.7608585, -4.1264238, -0.2869911, 0.2897208
9: -4.2426348, -3.6859026, -4.2465916, -3.6969614, -0.2508070, 0.2518083

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1739
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1984
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 1984
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 1096

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1029115
time: 3.46 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1029034
time: 3.62 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -13.0266285, -12.0826645, -13.0332346, -12.0798550, -0.4503584, 0.4504292
1: -8.4491444, -7.8945880, -8.4518547, -7.8952351, -0.2607780, 0.2580032
2: -4.0271688, -3.3975377, -4.0298252, -3.3961415, -0.2744607, 0.2749320
3: -4.8013000, -4.2074032, -4.8032207, -4.2059317, -0.3233962, 0.3319023
4: -8.3044701, -7.6478500, -8.3103781, -7.6466708, -0.2930095, 0.3157990
5: -15.6949501, -14.9548721, -15.6977625, -14.9546490, -0.5119290, 0.5200257
6: -22.8587761, -21.7879257, -22.8615646, -21.7841682, -0.4480178, 0.4350460
7: 4.4998198, 4.8752055, 4.4980683, 4.8790050, -0.2167314, 0.1771409
8: -4.7625809, -4.1273894, -4.7660861, -4.1263442, -0.2848618, 0.3041394
9: -4.2462454, -3.6968398, -4.2506695, -3.6957850, -0.2399943, 0.2723694

Time for backsubstitution: 21.72 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 1096

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1054576, upper bound: 0.1041182
time: 3.20 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1054663, upper bound: 0.1041160
time: 3.26 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -13.0314970, -12.0641174, -13.0328417, -12.0760422, -0.4514346, 0.4560840
1: -8.4471798, -7.8891735, -8.4497509, -7.8952546, -0.2550173, 0.2628160
2: -4.0272875, -3.3908558, -4.0288353, -3.3951023, -0.2716931, 0.2749169
3: -4.7967834, -4.2053642, -4.8004899, -4.2060461, -0.3232310, 0.3283486
4: -8.3002672, -7.6264501, -8.3059616, -7.6456499, -0.2961028, 0.3044229
5: -15.6927719, -14.9454441, -15.6960125, -14.9546490, -0.5171309, 0.5173259
6: -22.8576717, -21.8000908, -22.8614216, -21.7911510, -0.4441390, 0.4321702
7: 4.4908924, 4.8576980, 4.4992828, 4.8676534, -0.1829982, 0.1861839
8: -4.7573090, -4.1202059, -4.7619500, -4.1262345, -0.2872682, 0.2911565
9: -4.2435203, -3.6850355, -4.2468400, -3.6966491, -0.2516026, 0.2524440

Time for backsubstitution: 21.71 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 1739
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1984
type: A, layer: 3, pos: 1984
type: A, layer: 3, pos: 1794
type: B, layer: 3, pos: 1794
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 1096

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1040264
time: 3.17 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1040178
time: 4.15 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -13.0303297, -12.0760355, -13.0336428, -12.0760231, -0.4575169, 0.4521937
1: -8.4495087, -7.8951445, -8.4519281, -7.8951054, -0.2612324, 0.2586176
2: -4.0283575, -3.3948984, -4.0298438, -3.3947165, -0.2762861, 0.2749991
3: -4.8036776, -4.2057662, -4.8045912, -4.2056589, -0.3243933, 0.3346186
4: -8.3054790, -7.6453404, -8.3104019, -7.6452198, -0.2916572, 0.3198776
5: -15.6963329, -14.9545918, -15.6986647, -14.9545383, -0.5178065, 0.5198832
6: -22.8617802, -21.7818718, -22.8619099, -21.7807674, -0.4539530, 0.4363599
7: 4.4985991, 4.8771300, 4.4979882, 4.8800302, -0.2182645, 0.1773533
8: -4.7646918, -4.1261773, -4.7671604, -4.1261568, -0.2851357, 0.3060029
9: -4.2470994, -3.6960316, -4.2508912, -3.6954806, -0.2407897, 0.2730029

Time for backsubstitution: 21.71 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1739
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1794
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 1096

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1054576, upper bound: 0.1054704
time: 3.33 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1054663, upper bound: 0.1054667
time: 3.12 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.33 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 28.33
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1029115
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 28.33
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1029034
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 28.33
Output dim: 7, lower bound: -0.1054576, upper bound: 0.1041182
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 28.33
Output dim: 7, lower bound: -0.1054663, upper bound: 0.1041160
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 28.33
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1040264
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 28.33
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1040178
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 28.33
Output dim: 7, lower bound: -0.1054576, upper bound: 0.1054704
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 28.33
Output dim: 7, lower bound: -0.1054663, upper bound: 0.1054667

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -13.0272846, -12.0729942, -13.0346718, -12.0847530, -0.4402561, 0.4569824
1: -8.4468241, -7.8916531, -8.4535885, -7.9018025, -0.2435583, 0.2591343
2: -4.0256391, -3.3935285, -4.0299854, -3.3960385, -0.2707205, 0.2752259
3: -4.7926779, -4.2074337, -4.8001213, -4.2067342, -0.3201513, 0.3275635
4: -8.2954025, -7.6292086, -8.3058767, -7.6470022, -0.2928689, 0.3059146
5: -15.6907845, -14.9556885, -15.6982212, -14.9754801, -0.4848418, 0.5096474
6: -22.8543129, -21.8143139, -22.8609333, -21.8004856, -0.4289303, 0.4208786
7: 4.4945030, 4.8557038, 4.5028467, 4.8811107, -0.2011158, 0.1802523
8: -4.7511730, -4.1214566, -4.7593288, -4.1264167, -0.2827520, 0.2916911
9: -4.2397366, -3.6859298, -4.2472181, -3.6954184, -0.2496805, 0.2618968

Time for backsubstitution: 21.38 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1794
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1794
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_A1_B1_A1

### Relational analysis result of NS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1029115
time: 3.39 seconds

## Relational analysis of NS_A1_A1_B1_A2

### Relational analysis result of NS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1027552
time: 3.59 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -13.0266638, -12.0741100, -13.0387344, -12.0858889, -0.3311214, 0.4600408
1: -8.4467983, -7.9064798, -8.4484119, -7.9274354, -0.2403482, 0.2872669
2: -4.0231705, -3.3935513, -4.0257473, -3.3968487, -0.2714804, 0.2733160
3: -4.7926154, -4.2080402, -4.8005443, -4.2032003, -0.3229313, 0.2570949
4: -8.2940063, -7.6292443, -8.3046398, -7.6364412, -0.3072081, 0.2528633
5: -15.6906786, -14.9684525, -15.7198172, -14.9950991, -0.2523811, 0.5572777
6: -22.8540154, -21.8173847, -22.8844261, -21.8026485, -0.3397311, 0.4552526
7: 4.4971619, 4.8556051, 4.5070386, 4.8856621, -0.2130808, 0.1514702
8: -4.7502432, -4.1214628, -4.7588940, -4.1173258, -0.2940664, 0.2812384
9: -4.2392220, -3.6860018, -4.2470827, -3.6875744, -0.2410984, 0.2641878

Time for backsubstitution: 20.86 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1794
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1794
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_A1_B2_A1

### Relational analysis result of NS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1029033
time: 3.64 seconds

## Relational analysis of NS_A1_A1_B2_A2

### Relational analysis result of NS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1027471
time: 3.66 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -13.0261307, -12.0849228, -13.0346718, -12.0847530, -0.4454923, 0.4497368
1: -8.4491348, -7.8976212, -8.4535885, -7.9018025, -0.2496371, 0.2533281
2: -4.0267119, -3.3975523, -4.0299854, -3.3960385, -0.2737267, 0.2736177
3: -4.7995601, -4.2078266, -4.8001213, -4.2067342, -0.3211739, 0.3285632
4: -8.3006220, -7.6480436, -8.3058767, -7.6470022, -0.2883315, 0.3080230
5: -15.6943684, -14.9648399, -15.6982212, -14.9754801, -0.4848495, 0.5084853
6: -22.8584309, -21.7961140, -22.8609333, -21.8004856, -0.4288962, 0.4249940
7: 4.5022306, 4.8750534, 4.5028467, 4.8811107, -0.2134956, 0.1705790
8: -4.7585449, -4.1274300, -4.7593288, -4.1264167, -0.2805843, 0.2959616
9: -4.2433457, -3.6968658, -4.2472181, -3.6954184, -0.2384379, 0.2690142

Time for backsubstitution: 20.87 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051334, upper bound: 0.1027552
time: 3.38 seconds

## Relational analysis of NS_A1_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1054576, upper bound: 0.1041182
time: 3.22 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -13.0254412, -12.0860338, -13.0387344, -12.0858889, -0.4443576, 0.4527950
1: -8.4491110, -7.9124660, -8.4484119, -7.9274354, -0.2464107, 0.2748797
2: -4.0242500, -3.3975782, -4.0257473, -3.3968487, -0.2744753, 0.2722490
3: -4.7994971, -4.2084837, -4.8005443, -4.2032003, -0.3239529, 0.3277993
4: -8.2992096, -7.6481338, -8.3046398, -7.6364412, -0.3026679, 0.3084114
5: -15.6941032, -14.9775944, -15.7198172, -14.9950991, -0.4782352, 0.5560646
6: -22.8580627, -21.7992115, -22.8844261, -21.8026485, -0.4283147, 0.4593732
7: 4.5048828, 4.8749213, 4.5070386, 4.8856621, -0.2253302, 0.1703071
8: -4.7576165, -4.1274486, -4.7588940, -4.1173258, -0.2918861, 0.2966132
9: -4.2427979, -3.6969352, -4.2470827, -3.6875744, -0.2348350, 0.2713143

Time for backsubstitution: 21.04 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1027469
time: 3.76 seconds

## Relational analysis of NS_A1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1054663, upper bound: 0.1041160
time: 3.30 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -13.0310078, -12.0663719, -13.0350924, -12.0809250, -0.4474242, 0.4587722
1: -8.4471703, -7.8922076, -8.4536686, -7.9016714, -0.2439944, 0.2597250
2: -4.0268340, -3.3908737, -4.0300064, -3.3946133, -0.2720509, 0.2752998
3: -4.7950435, -4.2057848, -4.8014922, -4.2064576, -0.3211501, 0.3302679
4: -8.2964277, -7.6266513, -8.3059111, -7.6455507, -0.2916062, 0.3098242
5: -15.6921673, -14.9554157, -15.6991425, -14.9753685, -0.4906521, 0.5095468
6: -22.8573284, -21.8082809, -22.8612881, -21.7970943, -0.4348052, 0.4222043
7: 4.4933052, 4.8575497, 4.5027623, 4.8821425, -0.2023699, 0.1805079
8: -4.7532730, -4.1202474, -4.7604041, -4.1262255, -0.2830284, 0.2934244
9: -4.2406230, -3.6850615, -4.2474470, -3.6951132, -0.2504694, 0.2625066

Time for backsubstitution: 20.91 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1794
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1040264
time: 3.18 seconds

## Relational analysis of NS_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1039530
time: 3.19 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -13.0303202, -12.0674877, -13.0391388, -12.0820627, -0.4462974, 0.4618235
1: -8.4471436, -7.9070544, -8.4484863, -7.9273038, -0.2407798, 0.2812659
2: -4.0243645, -3.3908980, -4.0257711, -3.3954222, -0.2728169, 0.2739278
3: -4.7949810, -4.2064362, -4.8019147, -4.2029371, -0.3239169, 0.3294775
4: -8.2950325, -7.6267409, -8.3046732, -7.6349893, -0.3059158, 0.3102658
5: -15.6918955, -14.9681797, -15.7204437, -14.9949913, -0.4839878, 0.5572944
6: -22.8569660, -21.8113518, -22.8847408, -21.7992344, -0.4342101, 0.4566300
7: 4.4959626, 4.8574209, 4.5069551, 4.8866935, -0.2143605, 0.1802264
8: -4.7523451, -4.1202660, -4.7599711, -4.1171460, -0.2943599, 0.2941324
9: -4.2400732, -3.6851330, -4.2473140, -3.6872711, -0.2468686, 0.2647954

Time for backsubstitution: 20.96 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1794
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1794
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1040178
time: 4.50 seconds

## Relational analysis of NS_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1039453
time: 3.71 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -13.0298414, -12.0782890, -13.0350924, -12.0809250, -0.4526515, 0.4515066
1: -8.4494991, -7.8981781, -8.4536686, -7.9016714, -0.2500669, 0.2539321
2: -4.0279026, -3.3949142, -4.0300064, -3.3946133, -0.2755237, 0.2736876
3: -4.8019395, -4.2061863, -4.8014922, -4.2064576, -0.3221717, 0.3312812
4: -8.3016310, -7.6455350, -8.3059111, -7.6455507, -0.2869682, 0.3119342
5: -15.6957502, -14.9645586, -15.6991425, -14.9753685, -0.4907265, 0.5083489
6: -22.8614464, -21.7900620, -22.8612881, -21.7970943, -0.4348106, 0.4262955
7: 4.5010099, 4.8769789, 4.5027623, 4.8821425, -0.2147751, 0.1707977
8: -4.7606564, -4.1262174, -4.7604041, -4.1262255, -0.2808580, 0.2977173
9: -4.2442007, -3.6960568, -4.2474470, -3.6951132, -0.2392249, 0.2696424

Time for backsubstitution: 20.98 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1794
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1039530
time: 3.21 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1054576, upper bound: 0.1054704
time: 3.35 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -13.0291519, -12.0794058, -13.0391388, -12.0820627, -0.4515166, 0.4545584
1: -8.4494753, -7.9130263, -8.4484863, -7.9273038, -0.2468389, 0.2755237
2: -4.0254364, -3.3949385, -4.0257711, -3.3954222, -0.2762787, 0.2723174
3: -4.8018746, -4.2068434, -4.8019147, -4.2029371, -0.3249388, 0.3305178
4: -8.3002205, -7.6456242, -8.3046732, -7.6349893, -0.3012738, 0.3123763
5: -15.6954889, -14.9773178, -15.7204437, -14.9949913, -0.4841132, 0.5560451
6: -22.8610744, -21.7931595, -22.8847408, -21.7992344, -0.4342294, 0.4607260
7: 4.5036626, 4.8768473, 4.5069551, 4.8866935, -0.2266364, 0.1705248
8: -4.7597265, -4.1262331, -4.7599711, -4.1171460, -0.2921772, 0.2984231
9: -4.2436523, -3.6961260, -4.2473140, -3.6872711, -0.2356527, 0.2719405

Time for backsubstitution: 21.01 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1039453
time: 3.69 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1054663, upper bound: 0.1054667
time: 3.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 27.96 seconds
NS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1029115
NS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1027552
NS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1029033
NS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1027471
NS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1051334, upper bound: 0.1027552
NS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1054576, upper bound: 0.1041182
NS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1027469
NS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1054663, upper bound: 0.1041160
NS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1040264
NS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1039530
NS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1040178
NS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1039453
NS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1039530
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1054576, upper bound: 0.1054704
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1039453
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 27.96
Output dim: 7, lower bound: -0.1054663, upper bound: 0.1054667

## BFS NS instance: NS_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -13.0277786, -12.0707388, -13.0313826, -12.0847788, -0.4395068, 0.4534662
1: -8.4468346, -7.8886213, -8.4496460, -7.9019837, -0.2436709, 0.2616669
2: -4.0260944, -3.3935127, -4.0278540, -3.3965683, -0.2702510, 0.2725500
3: -4.7944183, -4.2070084, -4.7953262, -4.2072105, -0.3215468, 0.3222258
4: -8.2992392, -7.6290102, -8.2976379, -7.6475315, -0.2972064, 0.2917089
5: -15.6913891, -14.9457169, -15.6937561, -14.9756155, -0.4837008, 0.5173707
6: -22.8546581, -21.8061256, -22.8603382, -21.8117924, -0.4186182, 0.4304090
7: 4.4920907, 4.8558517, 4.5046115, 4.8662744, -0.1818674, 0.1788932
8: -4.7552099, -4.1214147, -4.7522345, -4.1265135, -0.2869368, 0.2807472
9: -4.2426348, -3.6859026, -4.2403035, -3.6970181, -0.2500689, 0.2481835

Time for backsubstitution: 21.01 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1739
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1984
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 1984
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1096

## Relational analysis of NS_A1_A1_B1_A1_A1

### Relational analysis result of NS_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1029106
time: 4.96 seconds

## Relational analysis of NS_A1_A1_B1_A1_A2

### Relational analysis result of NS_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1029115
time: 3.43 seconds

## BFS NS instance: NS_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -13.0266285, -12.0826645, -13.0321960, -12.0847626, -0.4456000, 0.4496028
1: -8.4491444, -7.8945880, -8.4518328, -7.9018326, -0.2498645, 0.2574614
2: -4.0271688, -3.3975377, -4.0288715, -3.3961766, -0.2744021, 0.2726493
3: -4.8013000, -4.2074032, -4.7994375, -4.2068157, -0.3227088, 0.3284972
4: -8.3044701, -7.6478500, -8.3021088, -7.6470933, -0.2928588, 0.3063836
5: -15.6949501, -14.9548721, -15.6964798, -14.9755077, -0.4843287, 0.5199757
6: -22.8587761, -21.7879257, -22.8608322, -21.8013325, -0.4284227, 0.4346292
7: 4.4998198, 4.8752055, 4.5033131, 4.8786798, -0.2165447, 0.1700189
8: -4.7625809, -4.1273894, -4.7574677, -4.1264315, -0.2848191, 0.2951958
9: -4.2462454, -3.6968398, -4.2443848, -3.6958399, -0.2392633, 0.2687221

Time for backsubstitution: 21.03 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 1096

## Relational analysis of NS_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1027552
time: 3.48 seconds

## Relational analysis of NS_A1_A1_B1_A2_B2

### Relational analysis result of NS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1027474
time: 3.60 seconds

## BFS NS instance: NS_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -13.0275059, -12.0707388, -13.0354652, -12.0859194, -0.3327353, 0.4576306
1: -8.4468346, -7.8886991, -8.4444313, -7.9276228, -0.2403975, 0.2891964
2: -4.0260944, -3.3935218, -4.0236101, -3.3973804, -0.2714332, 0.2706997
3: -4.7944183, -4.2072043, -4.7957482, -4.2036557, -0.3247199, 0.2526330
4: -8.2992392, -7.6292405, -8.2963905, -7.6369915, -0.3105485, 0.2404349
5: -15.6906834, -14.9457169, -15.7157917, -14.9952288, -0.2523031, 0.5644612
6: -22.8543949, -21.8061256, -22.8839016, -21.8138714, -0.3283753, 0.4611630
7: 4.4920907, 4.8557205, 4.5088072, 4.8708305, -0.1926171, 0.1499236
8: -4.7552099, -4.1214628, -4.7518072, -4.1174169, -0.2979763, 0.2702130
9: -4.2424874, -3.6859026, -4.2401676, -3.6891732, -0.2431722, 0.2503787

Time for backsubstitution: 20.95 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1739
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1984
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 1984
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1794
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1096

## Relational analysis of NS_A1_A1_B2_A1_A1

### Relational analysis result of NS_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1028989
time: 3.38 seconds

## Relational analysis of NS_A1_A1_B2_A1_A2

### Relational analysis result of NS_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1029038
time: 3.30 seconds

## BFS NS instance: NS_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -13.0255508, -12.0826645, -13.0362597, -12.0858974, -0.3350596, 0.4537616
1: -8.4491444, -7.8949051, -8.4466457, -7.9274654, -0.2465885, 0.2866344
2: -4.0271688, -3.3975749, -4.0246305, -3.3969879, -0.2755806, 0.2708173
3: -4.8013000, -4.2081633, -4.7998605, -4.2032800, -0.3258755, 0.2595956
4: -8.3044701, -7.6487513, -8.3008709, -7.6365342, -0.3062041, 0.2537471
5: -15.6922417, -14.9548721, -15.7181416, -14.9951210, -0.2503278, 0.5671840
6: -22.8577480, -21.7879257, -22.8843422, -21.8034725, -0.3446126, 0.4653916
7: 4.4998198, 4.8746724, 4.5075045, 4.8832235, -0.2272182, 0.1391834
8: -4.7625809, -4.1275806, -4.7570376, -4.1173401, -0.2958806, 0.2853433
9: -4.2456651, -3.6968398, -4.2442513, -3.6879964, -0.2324194, 0.2709122

Time for backsubstitution: 21.03 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1984
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1984

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1096

## Relational analysis of NS_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1027474
time: 3.58 seconds

## Relational analysis of NS_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051383, upper bound: 0.1027470
time: 4.10 seconds

## BFS NS instance: NS_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -13.0277786, -12.0707388, -13.0313826, -12.0847788, -0.4395068, 0.4534662
1: -8.4468346, -7.8886213, -8.4496460, -7.9019837, -0.2436709, 0.2616669
2: -4.0260944, -3.3935127, -4.0278540, -3.3965683, -0.2702510, 0.2725500
3: -4.7944183, -4.2070084, -4.7953262, -4.2072105, -0.3215468, 0.3222258
4: -8.2992392, -7.6290102, -8.2976379, -7.6475315, -0.2972064, 0.2917089
5: -15.6913891, -14.9457169, -15.6937561, -14.9756155, -0.4837008, 0.5173707
6: -22.8546581, -21.8061256, -22.8603382, -21.8117924, -0.4186182, 0.4304090
7: 4.4920907, 4.8558517, 4.5046115, 4.8662744, -0.1818674, 0.1788932
8: -4.7552099, -4.1214147, -4.7522345, -4.1265135, -0.2869368, 0.2807472
9: -4.2426348, -3.6859026, -4.2403035, -3.6970181, -0.2500689, 0.2481835

Time for backsubstitution: 20.95 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1739
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1832
type: B, layer: 3, pos: 1832
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1984
type: A, layer: 3, pos: 1794
type: A, layer: 3, pos: 1984
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1794
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1096

## Relational analysis of NS_A1_A2_B1_A1_A1

### Relational analysis result of NS_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1029106
time: 4.86 seconds

## Relational analysis of NS_A1_A2_B1_A1_A2

### Relational analysis result of NS_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1051333, upper bound: 0.1029115
time: 3.41 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.26 + 549.56 = 604.83 seconds
