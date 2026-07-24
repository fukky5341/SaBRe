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
execution time: IAR + RelationalAnalysis = 22.89 + 33.11 = 56.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.1108283, upper bound: 0.1108289

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 904

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1107001, upper bound: 0.1094642
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1094635, upper bound: 0.1107009
time: 3.05 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.31 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.31
Output dim: 7, lower bound: -0.1107001, upper bound: 0.1094642
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.31
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

Time for backsubstitution: 20.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.39 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1105846, upper bound: 0.1087034
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1099431, upper bound: 0.1093478
time: 3.04 seconds

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

Time for backsubstitution: 21.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1093470, upper bound: 0.1099438
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1087026, upper bound: 0.1105850
time: 3.14 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.02 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.02
Output dim: 7, lower bound: -0.1105846, upper bound: 0.1087034
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.02
Output dim: 7, lower bound: -0.1099431, upper bound: 0.1093478
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.02
Output dim: 7, lower bound: -0.1093470, upper bound: 0.1099438
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.02
Output dim: 7, lower bound: -0.1087026, upper bound: 0.1105850

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4613295, 0.4624741
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2607683, 0.2610005
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2523444, 0.2493802
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3017356, 0.3013532
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.2990866, 0.3005407
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5124292, 0.5139084
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4345021, 0.4344916
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2184643, 0.2179643
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2867951, 0.2905984
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2668278, 0.2680849

Time for backsubstitution: 20.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2622

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1092719, upper bound: 0.1081929
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1100638, upper bound: 0.1075370
time: 3.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4632077, 0.4605956
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2611185, 0.2606503
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2497625, 0.2519618
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3011510, 0.3019378
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3010530, 0.2985742
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5134625, 0.5128751
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4351625, 0.4338310
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2183017, 0.2181270
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2903535, 0.2870401
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2681515, 0.2667614

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 2622

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1087899, upper bound: 0.1088310
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1094295, upper bound: 0.1080200
time: 2.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4605956, 0.4632077
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2606503, 0.2611185
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2519619, 0.2497625
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3019378, 0.3011510
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.2985742, 0.3010530
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5128751, 0.5134625
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4338310, 0.4351628
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2181269, 0.2183017
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2870400, 0.2903535
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2667613, 0.2681514

Time for backsubstitution: 21.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 2622

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1080192, upper bound: 0.1094303
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1088303, upper bound: 0.1087906
time: 3.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4624739, 0.4613292
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2610005, 0.2607683
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2493804, 0.2523441
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3013532, 0.3017356
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3005407, 0.2990866
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5139084, 0.5124292
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4344916, 0.4345021
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2179643, 0.2184644
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2905984, 0.2867951
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2680850, 0.2668278

Time for backsubstitution: 22.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2622

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1075369, upper bound: 0.1100645
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1081921, upper bound: 0.1092726
time: 3.03 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.56 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1092719, upper bound: 0.1081929
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1100638, upper bound: 0.1075370
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1087899, upper bound: 0.1088310
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1094295, upper bound: 0.1080200
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1080192, upper bound: 0.1094303
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1088303, upper bound: 0.1087906
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1075369, upper bound: 0.1100645
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 7, lower bound: -0.1081921, upper bound: 0.1092726

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4257588, 0.4258907
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2082413, 0.2099949
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2318240, 0.2260440
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3345294, 0.3346653
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3241084, 0.3229778
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5267258, 0.5271921
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4079354, 0.4075065
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2205954, 0.2202970
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2598445, 0.2597189
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2306236, 0.2313186

Time for backsubstitution: 20.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Candidate
type: DSZ, layer: 3, pos: 766

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1063487, upper bound: 0.1040640
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1045671, upper bound: 0.1049712
time: 2.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4266243, 0.4250016
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2101128, 0.2081800
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2264264, 0.2313979
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3344631, 0.3347299
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3234904, 0.3235817
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5267453, 0.5271716
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4081776, 0.4072282
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2206345, 0.2202573
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2594326, 0.2600894
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2313432, 0.2305572

Time for backsubstitution: 21.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Candidate
type: DSZ, layer: 3, pos: 766

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1069515, upper bound: 0.1030519
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1059095, upper bound: 0.1044336
time: 3.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4257588, 0.4258907
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2082413, 0.2099949
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2318240, 0.2260440
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3345294, 0.3346653
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3241084, 0.3229778
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5267258, 0.5271921
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4079354, 0.4075065
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2205954, 0.2202970
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2598445, 0.2597189
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2306236, 0.2313186

Time for backsubstitution: 21.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Candidate
type: DSZ, layer: 3, pos: 766

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1057949, upper bound: 0.1046960
time: 2.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1041545, upper bound: 0.1056042
time: 3.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0361261, -12.0760136, -13.0361261, -12.0760136, -0.4266243, 0.4250016
1: -8.4537792, -7.8950758, -8.4537792, -7.8950758, -0.2101128, 0.2081800
2: -4.0309587, -3.3945775, -4.0309587, -3.3945775, -0.2264264, 0.2313979
3: -4.8052759, -4.2055769, -4.8052759, -4.2055769, -0.3344631, 0.3347299
4: -8.3141775, -7.6451287, -8.3141775, -7.6451287, -0.3234904, 0.3235817
5: -15.7004185, -14.9545002, -15.7004185, -14.9545002, -0.5267453, 0.5271716
6: -22.8620071, -21.7799339, -22.8620071, -21.7799339, -0.4081776, 0.4072282
7: 4.4975195, 4.8824663, 4.4975195, 4.8824663, -0.2206345, 0.2202573
8: -4.7690191, -4.1261396, -4.7690191, -4.1261396, -0.2594326, 0.2600894
9: -4.2537298, -3.6950593, -4.2537298, -3.6950593, -0.2313432, 0.2305572

Time for backsubstitution: 22.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Candidate
type: DSZ, layer: 3, pos: 766

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1063196, upper bound: 0.1034777
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1052854, upper bound: 0.1049894
time: 3.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Candidate
type: DSZ, layer: 3, pos: 766

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1049887, upper bound: 0.1052862
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1034770, upper bound: 0.1063199
time: 3.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 21.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Candidate
type: DSZ, layer: 3, pos: 766

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1056034, upper bound: 0.1041553
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1046952, upper bound: 0.1057957
time: 3.42 seconds

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

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Candidate
type: DSZ, layer: 3, pos: 766

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1044329, upper bound: 0.1059104
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1030511, upper bound: 0.1069523
time: 3.09 seconds

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

Time for backsubstitution: 21.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Candidate
type: DSZ, layer: 3, pos: 766

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1049705, upper bound: 0.1045679
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1040633, upper bound: 0.1063496
time: 3.25 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.34 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1063487, upper bound: 0.1040640
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1045671, upper bound: 0.1049712
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1069515, upper bound: 0.1030519
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1059095, upper bound: 0.1044336
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1057949, upper bound: 0.1046960
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1041545, upper bound: 0.1056042
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1063196, upper bound: 0.1034777
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1052854, upper bound: 0.1049894
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1049887, upper bound: 0.1052862
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1034770, upper bound: 0.1063199
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1056034, upper bound: 0.1041553
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1046952, upper bound: 0.1057957
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1044329, upper bound: 0.1059104
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1030511, upper bound: 0.1069523
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1049705, upper bound: 0.1045679
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.34
Output dim: 7, lower bound: -0.1040633, upper bound: 0.1063496

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 21.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Candidate
type: DSZ, layer: 3, pos: 2622

### Candidate
type: DSZ, layer: 3, pos: 1397

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1036848, upper bound: 0.1010601
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1036848, upper bound: 0.1010601
time: 3.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 21.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Candidate
type: DSZ, layer: 3, pos: 2622

### Candidate
type: DSZ, layer: 3, pos: 1397

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1018486, upper bound: 0.1019168
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1018486, upper bound: 0.1019168
time: 3.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 21.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Candidate
type: DSZ, layer: 3, pos: 2622

### Candidate
type: DSZ, layer: 3, pos: 1397

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1038244, upper bound: 0.1002170
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1038244, upper bound: 0.1002170
time: 3.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 21.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Candidate
type: DSZ, layer: 3, pos: 2622

### Candidate
type: DSZ, layer: 3, pos: 1397

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1028190, upper bound: 0.1017413
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1028190, upper bound: 0.1017413
time: 3.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2459
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Candidate
type: DSZ, layer: 3, pos: 2622

### Candidate
type: DSZ, layer: 3, pos: 1397

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1030814, upper bound: 0.1016259
time: 3.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1030814, upper bound: 0.1016259
time: 3.88 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 29.61 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.61
Output dim: 7, lower bound: -0.1036848, upper bound: 0.1010601
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.61
Output dim: 7, lower bound: -0.1036848, upper bound: 0.1010601
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.61
Output dim: 7, lower bound: -0.1018486, upper bound: 0.1019168
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.61
Output dim: 7, lower bound: -0.1018486, upper bound: 0.1019168
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.61
Output dim: 7, lower bound: -0.1038244, upper bound: 0.1002170
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.61
Output dim: 7, lower bound: -0.1038244, upper bound: 0.1002170
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.61
Output dim: 7, lower bound: -0.1028190, upper bound: 0.1017413
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.61
Output dim: 7, lower bound: -0.1028190, upper bound: 0.1017413
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.61
Output dim: 7, lower bound: -0.1030814, upper bound: 0.1016259
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.61
Output dim: 7, lower bound: -0.1030814, upper bound: 0.1016259
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 7, lower bound: -0.1041545, upper bound: 0.1056042
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 7, lower bound: -0.1063196, upper bound: 0.1034777
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 7, lower bound: -0.1052854, upper bound: 0.1049894
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 7, lower bound: -0.1049887, upper bound: 0.1052862
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 7, lower bound: -0.1034770, upper bound: 0.1063199
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 7, lower bound: -0.1056034, upper bound: 0.1041553
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 7, lower bound: -0.1046952, upper bound: 0.1057957
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 7, lower bound: -0.1044329, upper bound: 0.1059104
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 7, lower bound: -0.1030511, upper bound: 0.1069523
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 7, lower bound: -0.1049705, upper bound: 0.1045679
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 7, lower bound: -0.1040633, upper bound: 0.1063496

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.00 + 545.24 = 601.24 seconds
