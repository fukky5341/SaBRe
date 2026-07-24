## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.06767026200000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2808142, 0.2808142)
1: (-6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2129111, 0.2129111)
2: (-8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2251027, 0.2251027)
3: (-2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2969913, 0.2969913)
4: (-7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2891202, 0.2891202)
5: (-8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4648643, 0.4648643)
6: (-13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3328521, 0.3328521)
7: (5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1187847, 0.1187847)
8: (-2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2629046, 0.2629046)
9: (-2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1667712, 0.1667712)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.45 + 33.70 = 57.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0683536, upper bound: 0.0683538

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 568

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676406, upper bound: 0.0683533
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683531, upper bound: 0.0676408
time: 2.81 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.10 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.10
Output dim: 7, lower bound: -0.0676406, upper bound: 0.0683533
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.10
Output dim: 7, lower bound: -0.0683531, upper bound: 0.0676408

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2805092, 0.2798781
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2114091, 0.2124207
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2247924, 0.2241499
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2967365, 0.2962086
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2885702, 0.2889410
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4646735, 0.4642792
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3325689, 0.3319843
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1177459, 0.1184461
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2615802, 0.2588389
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1663677, 0.1666399

Time for backsubstitution: 21.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676131, upper bound: 0.0683478
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676350, upper bound: 0.0683258
time: 2.64 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2798784, 0.2805092
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2124205, 0.2114092
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2241501, 0.2247925
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2962089, 0.2967365
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2889409, 0.2885700
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4642792, 0.4646735
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3319843, 0.3325689
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1184461, 0.1177459
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2588389, 0.2615802
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1666399, 0.1663676

Time for backsubstitution: 21.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683256, upper bound: 0.0676353
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683475, upper bound: 0.0676134
time: 3.10 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.52 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.52
Output dim: 7, lower bound: -0.0676131, upper bound: 0.0683478
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.52
Output dim: 7, lower bound: -0.0676350, upper bound: 0.0683258
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.52
Output dim: 7, lower bound: -0.0683256, upper bound: 0.0676353
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.52
Output dim: 7, lower bound: -0.0683475, upper bound: 0.0676134

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2805138, 0.2798836
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2114105, 0.2124223
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2247977, 0.2241563
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2967362, 0.2962084
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2885697, 0.2889408
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4646711, 0.4642763
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3325686, 0.3319840
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1177460, 0.1184462
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2615812, 0.2588396
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1663671, 0.1666393

Time for backsubstitution: 21.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676129, upper bound: 0.0680151
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0672805, upper bound: 0.0683476
time: 2.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2805145, 0.2798829
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2114110, 0.2124220
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2247987, 0.2241553
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2967360, 0.2962084
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2885699, 0.2889407
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4646711, 0.4642763
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3325684, 0.3319840
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1177460, 0.1184462
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2615812, 0.2588398
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1663670, 0.1666394

Time for backsubstitution: 22.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676349, upper bound: 0.0679931
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673024, upper bound: 0.0683257
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2798827, 0.2805145
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2124219, 0.2114109
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2241553, 0.2247987
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2962084, 0.2967362
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2889407, 0.2885698
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4642763, 0.4646711
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3319840, 0.3325684
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1184462, 0.1177460
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2588398, 0.2615809
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1666394, 0.1663671

Time for backsubstitution: 22.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683255, upper bound: 0.0673026
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679929, upper bound: 0.0676351
time: 2.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2798836, 0.2805138
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2124224, 0.2114105
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2241563, 0.2247977
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2962084, 0.2967362
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2889407, 0.2885698
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4642763, 0.4646711
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3319840, 0.3325686
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1184462, 0.1177460
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2588398, 0.2615812
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1666394, 0.1663671

Time for backsubstitution: 22.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683474, upper bound: 0.0672807
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680148, upper bound: 0.0676132
time: 2.70 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 27.89 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.89
Output dim: 7, lower bound: -0.0676129, upper bound: 0.0680151
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.89
Output dim: 7, lower bound: -0.0672805, upper bound: 0.0683476
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.89
Output dim: 7, lower bound: -0.0676349, upper bound: 0.0679931
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.89
Output dim: 7, lower bound: -0.0673024, upper bound: 0.0683257
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.89
Output dim: 7, lower bound: -0.0683255, upper bound: 0.0673026
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.89
Output dim: 7, lower bound: -0.0679929, upper bound: 0.0676351
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.89
Output dim: 7, lower bound: -0.0683474, upper bound: 0.0672807
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.89
Output dim: 7, lower bound: -0.0680148, upper bound: 0.0676132

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2804317, 0.2798259
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2114654, 0.2124197
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2247967, 0.2241825
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2967284, 0.2961974
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2884072, 0.2887900
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4646244, 0.4642167
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3325605, 0.3319836
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1177834, 0.1184440
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2615788, 0.2588992
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1660139, 0.1663902

Time for backsubstitution: 21.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2562

Time for candidate selection: 0.38 seconds

### Candidate
type: DSZ, layer: 3, pos: 908

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673606, upper bound: 0.0678994
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0674972, upper bound: 0.0677627
time: 2.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2804558, 0.2798018
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2114081, 0.2124768
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2248240, 0.2241552
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2967250, 0.2962005
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2884191, 0.2887783
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4646111, 0.4642301
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3325682, 0.3319759
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1177437, 0.1184836
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2616408, 0.2588370
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1661180, 0.1662861

Time for backsubstitution: 21.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2562

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 908

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670281, upper bound: 0.0682319
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0671648, upper bound: 0.0680952
time: 2.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2804327, 0.2798250
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2114654, 0.2124194
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2247976, 0.2241815
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2967284, 0.2961974
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2884072, 0.2887900
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4646249, 0.4642162
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3325605, 0.3319838
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1177834, 0.1184440
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2615783, 0.2588992
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1660138, 0.1663903

Time for backsubstitution: 21.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2562

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 908

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673825, upper bound: 0.0678774
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675192, upper bound: 0.0677408
time: 2.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2804568, 0.2798009
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2114081, 0.2124766
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2248249, 0.2241542
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2967250, 0.2962005
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2884191, 0.2887782
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4646115, 0.4642296
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3325682, 0.3319762
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1177438, 0.1184836
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2616403, 0.2588372
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1661178, 0.1662862

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2562

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 3, pos: 908

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670500, upper bound: 0.0682100
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0671867, upper bound: 0.0680733
time: 2.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2798009, 0.2804568
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2124767, 0.2114083
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2241542, 0.2248249
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2962005, 0.2967250
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2887782, 0.2884190
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4642296, 0.4646115
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3319762, 0.3325682
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1184835, 0.1177438
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2588370, 0.2616405
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1662862, 0.1661179

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2562

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 3, pos: 908

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680731, upper bound: 0.0671869
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682098, upper bound: 0.0670502
time: 2.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2798250, 0.2804327
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2124195, 0.2114655
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2241815, 0.2247976
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2961974, 0.2967284
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2887899, 0.2884073
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4642162, 0.4646249
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3319838, 0.3325605
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1184440, 0.1177834
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2588995, 0.2615786
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1663904, 0.1660138

Time for backsubstitution: 21.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2562

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 908

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0677406, upper bound: 0.0675194
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678772, upper bound: 0.0673827
time: 2.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2798018, 0.2804558
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2124767, 0.2114080
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2241552, 0.2248240
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2962005, 0.2967250
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2887782, 0.2884190
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4642301, 0.4646111
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3319759, 0.3325682
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1184836, 0.1177438
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2588370, 0.2616405
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1662862, 0.1661180

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2562

Time for candidate selection: 0.37 seconds

### Candidate
type: DSZ, layer: 3, pos: 908

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680950, upper bound: 0.0671650
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682317, upper bound: 0.0670283
time: 2.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2798257, 0.2804317
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2124195, 0.2114651
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2241824, 0.2247967
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2961972, 0.2967284
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2887901, 0.2884073
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4642167, 0.4646244
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3319836, 0.3325605
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1184440, 0.1177834
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2588990, 0.2615786
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1663902, 0.1660139

Time for backsubstitution: 21.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2562

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 908

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0677625, upper bound: 0.0674975
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678991, upper bound: 0.0673608
time: 2.72 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 27.58 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0673606, upper bound: 0.0678994
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0674972, upper bound: 0.0677627
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0670281, upper bound: 0.0682319
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0671648, upper bound: 0.0680952
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0673825, upper bound: 0.0678774
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0675192, upper bound: 0.0677408
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0670500, upper bound: 0.0682100
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0671867, upper bound: 0.0680733
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0680731, upper bound: 0.0671869
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0682098, upper bound: 0.0670502
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0677406, upper bound: 0.0675194
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0678772, upper bound: 0.0673827
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0680950, upper bound: 0.0671650
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0682317, upper bound: 0.0670283
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0677625, upper bound: 0.0674975
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 7, lower bound: -0.0678991, upper bound: 0.0673608

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2762651, 0.2761213
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2204410, 0.2198768
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2273471, 0.2272946
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2876263, 0.2862759
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2822707, 0.2832708
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4803128, 0.4789844
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3368845, 0.3358812
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1184738, 0.1193341
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2478608, 0.2421782
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1334256, 0.1361374

Time for backsubstitution: 22.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2562

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 67

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0669370, upper bound: 0.0638414
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0633027, upper bound: 0.0674741
time: 2.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2767272, 0.2756592
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2189223, 0.2213957
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2279087, 0.2267330
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2868068, 0.2870953
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2828882, 0.2826533
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4793921, 0.4799051
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3364582, 0.3363075
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1186734, 0.1191345
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2448577, 0.2451813
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1357611, 0.1338019

Time for backsubstitution: 21.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2562

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 67

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0670719, upper bound: 0.0637048
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0634393, upper bound: 0.0673392
time: 2.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2762895, 0.2760972
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2203838, 0.2199339
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2273744, 0.2272673
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2876232, 0.2862790
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2822824, 0.2832592
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4802995, 0.4789977
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3368921, 0.3358736
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1184343, 0.1193737
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2479228, 0.2421162
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1335297, 0.1360333

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2562

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 67

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0666045, upper bound: 0.0641739
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0629702, upper bound: 0.0678063
time: 3.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2767510, 0.2756352
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2188650, 0.2214530
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2279360, 0.2267057
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2868035, 0.2870986
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2828999, 0.2826416
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4793787, 0.4799180
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3364658, 0.3362999
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1186339, 0.1191741
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2449199, 0.2451193
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1358652, 0.1336978

Time for backsubstitution: 22.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2562

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 67

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0667394, upper bound: 0.0640373
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0631068, upper bound: 0.0676713
time: 3.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2762661, 0.2761203
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2204415, 0.2198764
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2273481, 0.2272936
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2876263, 0.2862759
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2822707, 0.2832708
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4803128, 0.4789839
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3368843, 0.3358815
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1184739, 0.1193341
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2478606, 0.2421784
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1334255, 0.1361375

Time for backsubstitution: 21.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2562

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 67

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0669590, upper bound: 0.0638195
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0633246, upper bound: 0.0674521
time: 2.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2767282, 0.2756585
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2189223, 0.2213954
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2279097, 0.2267320
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2868068, 0.2870953
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2828882, 0.2826533
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4793925, 0.4799047
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3364580, 0.3363075
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1186735, 0.1191345
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2448577, 0.2451816
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1357610, 0.1338020

Time for backsubstitution: 21.75 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.15 + 553.62 = 610.77 seconds
