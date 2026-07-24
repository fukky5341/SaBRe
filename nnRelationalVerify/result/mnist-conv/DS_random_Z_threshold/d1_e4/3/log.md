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
execution time: IAR + RelationalAnalysis = 24.12 + 33.15 = 57.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.1108283, upper bound: 0.1108289

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 904

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1107001, upper bound: 0.1094642
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1094635, upper bound: 0.1107009
time: 3.08 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.20 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.20
Output dim: 7, lower bound: -0.1107001, upper bound: 0.1094642
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.20
Output dim: 7, lower bound: -0.1094635, upper bound: 0.1107009

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4553072, 0.4545736
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2603244, 0.2602063
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2771467, 0.2767643
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3334744, 0.3336766
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3214948, 0.3209825
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5264907, 0.5269365
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4499753, 0.4493043
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2207309, 0.2203935
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.3056686, 0.3059134
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2762387, 0.2761720

Time for backsubstitution: 22.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 558

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1479

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1106032, upper bound: 0.1093624
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1105984, upper bound: 0.1093671
time: 2.89 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4545739, 0.4553072
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2602063, 0.2603244
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2767643, 0.2771466
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3336766, 0.3334744
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3209825, 0.3214948
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5269365, 0.5264907
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4493043, 0.4499755
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2203935, 0.2207309
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.3059132, 0.3056684
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2761722, 0.2762386

Time for backsubstitution: 22.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 558

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1984

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1094585, upper bound: 0.1098813
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1086471, upper bound: 0.1106955
time: 3.02 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.65 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.65
Output dim: 7, lower bound: -0.1106032, upper bound: 0.1093624
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.65
Output dim: 7, lower bound: -0.1105984, upper bound: 0.1093671
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.65
Output dim: 7, lower bound: -0.1094585, upper bound: 0.1098813
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.65
Output dim: 7, lower bound: -0.1086471, upper bound: 0.1106955

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4552004, 0.4544821
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2572541, 0.2561668
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2737874, 0.2746753
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3324397, 0.3313990
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3212404, 0.3207448
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5257807, 0.5249119
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4464722, 0.4446905
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2204101, 0.2201476
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.3043487, 0.3047163
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2760735, 0.2756990

Time for backsubstitution: 22.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 570

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 310

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1101662, upper bound: 0.1072186
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1076576, upper bound: 0.1088846
time: 3.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4552157, 0.4544668
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2562847, 0.2571361
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2750577, 0.2734050
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3311970, 0.3326418
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3212571, 0.3207281
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5244660, 0.5262265
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4453616, 0.4458010
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2204850, 0.2200727
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.3044715, 0.3045936
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2757654, 0.2760068

Time for backsubstitution: 22.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 310

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 766

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1074883, upper bound: 0.1051734
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1063700, upper bound: 0.1061424
time: 3.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4513624, 0.4527819
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2546877, 0.2535852
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2725971, 0.2735503
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3336818, 0.3334527
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3203778, 0.3209355
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5266366, 0.5262117
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4466460, 0.4477241
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2204270, 0.2206610
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2991219, 0.3000820
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2761099, 0.2761965

Time for backsubstitution: 23.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 206

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 766

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1062334, upper bound: 0.1056385
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1052649, upper bound: 0.1069431
time: 3.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4520485, 0.4520960
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2534665, 0.2548056
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2731686, 0.2729795
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3336549, 0.3334796
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3204231, 0.3208904
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5266576, 0.5261912
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4470537, 0.4473169
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2203236, 0.2207645
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.3003268, 0.2988780
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2761300, 0.2761763

Time for backsubstitution: 23.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2622

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1074732, upper bound: 0.1101758
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1081231, upper bound: 0.1093840
time: 3.02 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.05 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 7, lower bound: -0.1101662, upper bound: 0.1072186
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 7, lower bound: -0.1076576, upper bound: 0.1088846
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 7, lower bound: -0.1074883, upper bound: 0.1051734
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 7, lower bound: -0.1063700, upper bound: 0.1061424
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 7, lower bound: -0.1062334, upper bound: 0.1056385
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 7, lower bound: -0.1052649, upper bound: 0.1069431
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 7, lower bound: -0.1074732, upper bound: 0.1101758
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 7, lower bound: -0.1081231, upper bound: 0.1093840

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4475806, 0.4475782
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2563877, 0.2565197
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2747576, 0.2735808
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3287866, 0.3281207
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3185139, 0.3190072
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5237441, 0.5239739
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4470046, 0.4466331
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2127525, 0.2116565
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2960522, 0.2984707
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2695625, 0.2700281

Time for backsubstitution: 23.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 2622

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1500

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1099714, upper bound: 0.1064427
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1093901, upper bound: 0.1070252
time: 3.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4483120, 0.4468470
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2566376, 0.2562696
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2739632, 0.2743753
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3279185, 0.3289886
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3195195, 0.3180015
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5235281, 0.5241899
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4473040, 0.4463334
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2119938, 0.2124152
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2982259, 0.2962973
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2700946, 0.2694960

Time for backsubstitution: 23.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 766

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1069851, upper bound: 0.1083971
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1070812, upper bound: 0.1083624
time: 2.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4546239, 0.4520495
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2627122, 0.2630442
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2758958, 0.2753407
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3242369, 0.3246915
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.2972190, 0.2947235
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5192184, 0.5184493
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4371531, 0.4343119
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.1780663, 0.1762502
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2858028, 0.2861824
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2452888, 0.2417591

Time for backsubstitution: 23.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 570

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1739

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1069636, upper bound: 0.1035695
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1064069, upper bound: 0.1047009
time: 3.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4527791, 0.4538898
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2631645, 0.2625942
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2757208, 0.2755135
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3244874, 0.3244390
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.2952294, 0.2967067
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5179996, 0.5196643
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4349830, 0.4364781
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.1765876, 0.1777198
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2859340, 0.2860478
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2418196, 0.2452224

Time for backsubstitution: 22.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 558

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1479

### Candidate
type: DSZ, layer: 3, pos: 1500

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1061666, upper bound: 0.1057766
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1052719, upper bound: 0.1059916
time: 3.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4538901, 0.4527791
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2625942, 0.2631645
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2755136, 0.2757208
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3244390, 0.3244874
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.2967067, 0.2952294
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5196643, 0.5179996
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4364781, 0.4349830
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.1777198, 0.1765876
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2860479, 0.2859339
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2452226, 0.2418197

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1096

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1061174, upper bound: 0.1048868
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1054776, upper bound: 0.1055174
time: 2.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4520495, 0.4546237
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2630444, 0.2627122
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2753406, 0.2758958
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3246915, 0.3242369
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.2947235, 0.2972190
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5184493, 0.5192184
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4343119, 0.4371531
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.1762501, 0.1780663
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2861824, 0.2858028
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2417591, 0.2452890

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 1739

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1255

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1044621, upper bound: 0.1056097
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1039252, upper bound: 0.1060869
time: 5.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4250016, 0.4266243
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2081801, 0.2101129
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2313979, 0.2264263
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3347299, 0.3344631
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3235817, 0.3234904
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5271716, 0.5267453
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4072282, 0.4081776
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2202573, 0.2206343
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2600894, 0.2594326
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2305572, 0.2313432

Time for backsubstitution: 22.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1832

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 570

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1062534, upper bound: 0.1099821
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1072711, upper bound: 0.1092313
time: 4.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4258907, 0.4257588
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2099948, 0.2082414
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2260441, 0.2318240
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3346653, 0.3345294
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3229778, 0.3241084
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5271921, 0.5267258
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4075065, 0.4079354
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2202971, 0.2205955
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2597189, 0.2598445
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2313186, 0.2306236

Time for backsubstitution: 21.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 206

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 570

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1068994, upper bound: 0.1091980
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1079177, upper bound: 0.1086350
time: 3.15 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.57 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1099714, upper bound: 0.1064427
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1093901, upper bound: 0.1070252
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1069851, upper bound: 0.1083971
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1070812, upper bound: 0.1083624
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1069636, upper bound: 0.1035695
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1064069, upper bound: 0.1047009
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1061666, upper bound: 0.1057766
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1052719, upper bound: 0.1059916
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1061174, upper bound: 0.1048868
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1054776, upper bound: 0.1055174
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1044621, upper bound: 0.1056097
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1039252, upper bound: 0.1060869
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1062534, upper bound: 0.1099821
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1072711, upper bound: 0.1092313
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1068994, upper bound: 0.1091980
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.57
Output dim: 7, lower bound: -0.1079177, upper bound: 0.1086350

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4528365, 0.4520843
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2611538, 0.2614051
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2767359, 0.2762539
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3326483, 0.3328776
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3204517, 0.3198481
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5243149, 0.5241604
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4493659, 0.4486082
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2206057, 0.2201848
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.3055797, 0.3058221
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2715855, 0.2718111

Time for backsubstitution: 21.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1479

### Candidate
type: DSZ, layer: 3, pos: 310

### Candidate
type: DSZ, layer: 3, pos: 570

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1090394, upper bound: 0.1062468
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1097470, upper bound: 0.1059343
time: 3.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4528179, 0.4521029
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2615231, 0.2610358
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2766362, 0.2763535
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3326755, 0.3328505
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3203604, 0.3199394
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5237150, 0.5247607
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4492791, 0.4486947
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2205223, 0.2202682
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.3055773, 0.3058245
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2718778, 0.2715189

Time for backsubstitution: 21.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 2459

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2622

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1080796, upper bound: 0.1065070
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1088703, upper bound: 0.1056986
time: 3.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4387183, 0.4381781
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2592797, 0.2596507
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2771733, 0.2765025
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3269184, 0.3274369
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3216429, 0.3211517
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5235200, 0.5239182
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4503052, 0.4496574
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2147529, 0.2146741
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.3019977, 0.3023486
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2742131, 0.2745006

Time for backsubstitution: 21.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1397

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1832

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1066270, upper bound: 0.1071242
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1057120, upper bound: 0.1080343
time: 3.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4389117, 0.4379845
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2597687, 0.2591617
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2768848, 0.2767909
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3272350, 0.3271203
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3216641, 0.3211305
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5234723, 0.5239658
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4503286, 0.4496341
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2150114, 0.2144156
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.3021038, 0.3022425
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2745671, 0.2741466

Time for backsubstitution: 21.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2622

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1500

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1068900, upper bound: 0.1075852
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1063023, upper bound: 0.1081729
time: 2.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4543011, 0.4535069
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2593068, 0.2591535
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2755190, 0.2750149
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3326929, 0.3332355
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3189542, 0.3182530
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5233712, 0.5243535
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4490397, 0.4485681
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2203583, 0.2197481
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.3039074, 0.3045216
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2747450, 0.2747523

Time for backsubstitution: 21.93 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.27 + 547.37 = 604.65 seconds
