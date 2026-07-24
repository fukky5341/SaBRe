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
execution time: IAR + RelationalAnalysis = 23.61 + 33.19 = 56.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0683536, upper bound: 0.0683538

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683534, upper bound: 0.0680211
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680209, upper bound: 0.0683536
time: 2.67 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.28 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.28
Output dim: 7, lower bound: -0.0683534, upper bound: 0.0680211
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.28
Output dim: 7, lower bound: -0.0680209, upper bound: 0.0683536

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2807324, 0.2807562
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2129657, 0.2129085
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2251017, 0.2251290
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2969835, 0.2969801
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2889576, 0.2889693
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4648170, 0.4648037
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3328440, 0.3328519
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1188220, 0.1187825
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2629018, 0.2629640
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1664180, 0.1665220

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 568

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676405, upper bound: 0.0680206
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683529, upper bound: 0.0673081
time: 2.68 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2807562, 0.2807324
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2129085, 0.2129657
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2251289, 0.2251017
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2969801, 0.2969835
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2889693, 0.2889576
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4648037, 0.4648170
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3328519, 0.3328440
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1187825, 0.1188221
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2629642, 0.2629020
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1665220, 0.1664180

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 568

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679934, upper bound: 0.0683481
time: 2.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680154, upper bound: 0.0683262
time: 2.54 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.11 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.11
Output dim: 7, lower bound: -0.0676405, upper bound: 0.0680206
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.11
Output dim: 7, lower bound: -0.0683529, upper bound: 0.0673081
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.11
Output dim: 7, lower bound: -0.0679934, upper bound: 0.0683481
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.11
Output dim: 7, lower bound: -0.0680154, upper bound: 0.0683262

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2804272, 0.2798204
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2114639, 0.2124181
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2247915, 0.2241763
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2967288, 0.2961977
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2884073, 0.2887902
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4646263, 0.4642181
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3325608, 0.3319838
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1177833, 0.1184439
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2615776, 0.2588985
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1660142, 0.1663906

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676129, upper bound: 0.0680151
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676349, upper bound: 0.0679931
time: 2.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2797964, 0.2804513
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2124753, 0.2114066
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2241490, 0.2248187
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2962010, 0.2967255
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2887783, 0.2884192
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4642315, 0.4646130
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3319762, 0.3325684
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1184835, 0.1177437
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2588363, 0.2616398
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1662866, 0.1661183

Time for backsubstitution: 23.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683255, upper bound: 0.0673026
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683474, upper bound: 0.0672807
time: 2.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2807608, 0.2807376
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2129102, 0.2129676
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2251343, 0.2251079
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2969799, 0.2969830
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2889689, 0.2889572
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4648018, 0.4648156
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3328519, 0.3328440
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1187825, 0.1188222
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2629652, 0.2629032
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1665216, 0.1664174

Time for backsubstitution: 23.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 568

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 568

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0672805, upper bound: 0.0683476
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679929, upper bound: 0.0676351
time: 2.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2807617, 0.2807367
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2129104, 0.2129674
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2251352, 0.2251070
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2969799, 0.2969830
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2889692, 0.2889572
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4648023, 0.4648151
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3328519, 0.3328443
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1187825, 0.1188221
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2629652, 0.2629032
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1665215, 0.1664175

Time for backsubstitution: 23.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 568

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 568

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673024, upper bound: 0.0683257
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680148, upper bound: 0.0676132
time: 2.58 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.36 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.36
Output dim: 7, lower bound: -0.0676129, upper bound: 0.0680151
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.36
Output dim: 7, lower bound: -0.0676349, upper bound: 0.0679931
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.36
Output dim: 7, lower bound: -0.0683255, upper bound: 0.0673026
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.36
Output dim: 7, lower bound: -0.0683474, upper bound: 0.0672807
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.36
Output dim: 7, lower bound: -0.0672805, upper bound: 0.0683476
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.36
Output dim: 7, lower bound: -0.0679929, upper bound: 0.0676351
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.36
Output dim: 7, lower bound: -0.0673024, upper bound: 0.0683257
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.36
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

Time for backsubstitution: 22.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2562
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1926

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2635

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0670901, upper bound: 0.0666622
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0661515, upper bound: 0.0674431
time: 2.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 23.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 2562
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 908

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1839

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0673327, upper bound: 0.0655038
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651455, upper bound: 0.0676910
time: 2.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 23.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 2562
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 1222

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1839

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680233, upper bound: 0.0648132
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0658361, upper bound: 0.0670005
time: 2.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 23.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 2562
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 908

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680950, upper bound: 0.0671650
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682317, upper bound: 0.0670283
time: 2.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2562
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1926

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 908

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670281, upper bound: 0.0682319
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0671648, upper bound: 0.0680952
time: 2.65 seconds

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

Time for backsubstitution: 22.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 2562
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2579

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1501

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0677765, upper bound: 0.0673440
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0675477, upper bound: 0.0674339
time: 2.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 22.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2562

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1222

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0665322, upper bound: 0.0683179
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0672954, upper bound: 0.0669853
time: 2.64 seconds

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

Time for backsubstitution: 22.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 2562
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 3125

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1501

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678137, upper bound: 0.0671680
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0677236, upper bound: 0.0673968
time: 2.66 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 27.63 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0670901, upper bound: 0.0666622
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0661515, upper bound: 0.0674431
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0673327, upper bound: 0.0655038
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0651455, upper bound: 0.0676910
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0680233, upper bound: 0.0648132
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0658361, upper bound: 0.0670005
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0680950, upper bound: 0.0671650
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0682317, upper bound: 0.0670283
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0670281, upper bound: 0.0682319
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0671648, upper bound: 0.0680952
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0677765, upper bound: 0.0673440
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0675477, upper bound: 0.0674339
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0665322, upper bound: 0.0683179
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0672954, upper bound: 0.0669853
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0678137, upper bound: 0.0671680
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 7, lower bound: -0.0677236, upper bound: 0.0673968

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2747045, 0.2745070
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2076551, 0.2061332
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2192914, 0.2196296
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2968609, 0.2962468
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2831783, 0.2847867
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4647145, 0.4642997
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3335371, 0.3300822
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1164791, 0.1172778
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2481649, 0.2467787
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1658021, 0.1661978

Time for backsubstitution: 22.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2562
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1222

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2579

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0630301, upper bound: 0.0655814
time: 2.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0630364, upper bound: 0.0655749
time: 3.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2744832, 0.2747284
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2061903, 0.2075980
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2196023, 0.2193186
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2962499, 0.2968576
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2847750, 0.2831900
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4643130, 0.4647012
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3300743, 0.3335450
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1173174, 0.1164395
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2467167, 0.2482269
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1660937, 0.1659062

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2562
type: DSZ, layer: 3, pos: 908
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1222

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2586

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0662944, upper bound: 0.0631108
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0663628, upper bound: 0.0630580
time: 2.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2756352, 0.2767513
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2214528, 0.2188649
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2267057, 0.2279360
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2870986, 0.2868037
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2826416, 0.2828999
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4799180, 0.4793787
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3362999, 0.3364658
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1191741, 0.1186339
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2451192, 0.2449198
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1336978, 0.1358652

Time for backsubstitution: 22.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 2562
type: DSZ, layer: 3, pos: 1492

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 67

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676715, upper bound: 0.0631070
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0640371, upper bound: 0.0667397
time: 2.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2760973, 0.2762893
1: -6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2199341, 0.2203840
2: -8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2272673, 0.2273744
3: -2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2862790, 0.2876232
4: -7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2832592, 0.2822824
5: -8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4789977, 0.4802995
6: -13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3358736, 0.3368921
7: 5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1193737, 0.1184343
8: -2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2421161, 0.2479229
9: -2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1360332, 0.1335297

Time for backsubstitution: 22.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2562
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1926

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1222

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0668914, upper bound: 0.0670213
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682239, upper bound: 0.0662581
time: 2.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 22.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2562
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1926
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 2635
type: DSZ, layer: 3, pos: 3125

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1839

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0667259, upper bound: 0.0657583
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0645286, upper bound: 0.0679298
time: 2.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 22.39 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.80 + 558.48 = 615.28 seconds
