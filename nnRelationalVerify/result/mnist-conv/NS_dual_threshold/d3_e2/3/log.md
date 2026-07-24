## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.36109017


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7289498, 0.7289498)
1: (-10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5894637, 0.5894637)
2: (-4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4779406, 0.4779406)
3: (-3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7333860, 0.7333860)
4: (-3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5551434, 0.5551434)
5: (-9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4376069, 0.4376069)
6: (-14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5916382, 0.5916383)
7: (3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7084303, 0.7084298)
8: (-6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5950758, 0.5950758)
9: (-1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4374404, 0.4374403)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.13 + 34.48 = 57.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.4012108, upper bound: 0.4012113

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962

Time for candidate selection: 0.59 seconds

### Candidate
type: A, layer: 3, pos: 2818

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3817583, upper bound: 0.3893787
time: 3.81 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3861259, upper bound: 0.3861264
time: 3.44 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.85 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.85
Output dim: 7, lower bound: -0.3817583, upper bound: 0.3893787
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.85
Output dim: 7, lower bound: -0.3861259, upper bound: 0.3861264

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -15.1882420, -13.9735336, -15.1927280, -13.9735336, -0.7251005, 0.7289336
1: -10.1070337, -9.0974617, -10.1100082, -9.0974617, -0.5873301, 0.5894308
2: -4.2032676, -3.3283267, -4.2049417, -3.3283174, -0.4763052, 0.4779359
3: -3.1392529, -1.9299777, -3.1393237, -1.9232740, -0.7332292, 0.7248981
4: -3.6569180, -2.8445487, -3.6613157, -2.8445070, -0.5501511, 0.5550253
5: -9.2376385, -8.4675541, -9.2376575, -8.4610548, -0.4375170, 0.4302403
6: -14.7871857, -13.7760391, -14.7872343, -13.7731915, -0.5914344, 0.5903987
7: 3.0898681, 3.8798275, 3.0774527, 3.8798275, -0.6978903, 0.7082648
8: -6.7023869, -5.8084626, -6.7023869, -5.8010044, -0.5950401, 0.5841651
9: -1.3172565, -0.6008897, -1.3182242, -0.6008139, -0.4360985, 0.4373089

Time for backsubstitution: 8.73 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 2818

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3815192, upper bound: 0.3815199
time: 4.08 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3815192, upper bound: 0.3861274
time: 3.76 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -15.1723232, -13.9523754, -15.1890841, -13.9735336, -0.7299006, 0.7500467
1: -10.0923443, -9.0854988, -10.1071033, -9.0974617, -0.5908093, 0.6004858
2: -4.2004495, -3.3220515, -4.2042627, -3.3283234, -0.4776177, 0.4849483
3: -3.1783366, -1.9449842, -3.1393046, -1.9274828, -0.7829959, 0.7329955
4: -3.6441889, -2.8236494, -3.6583443, -2.8445191, -0.5533183, 0.5769804
5: -9.2672348, -8.4868479, -9.2376509, -8.4651861, -0.4763145, 0.4339985
6: -14.7993088, -13.7818928, -14.7872200, -13.7745266, -0.5986378, 0.5943592
7: 3.0985899, 3.9374599, 3.0816374, 3.8798275, -0.7312059, 0.7577577
8: -6.7380075, -5.8296738, -6.7023869, -5.8066292, -0.6522427, 0.5877047
9: -1.3114789, -0.5978093, -1.3172626, -0.6008382, -0.4346439, 0.4425542

Time for backsubstitution: 8.47 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 2818

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3861265, upper bound: 0.3815187
time: 3.47 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3861265, upper bound: 0.3861264
time: 3.41 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 15.61 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.61
Output dim: 7, lower bound: -0.3815192, upper bound: 0.3815199
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.61
Output dim: 7, lower bound: -0.3815192, upper bound: 0.3861274
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.61
Output dim: 7, lower bound: -0.3861265, upper bound: 0.3815187
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.61
Output dim: 7, lower bound: -0.3861265, upper bound: 0.3861264

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -15.1882420, -13.9735336, -15.1882420, -13.9735336, -0.7250843, 0.7250844
1: -10.1070337, -9.0974617, -10.1070337, -9.0974617, -0.5872972, 0.5872972
2: -4.2032676, -3.3283267, -4.2032676, -3.3283267, -0.4763005, 0.4763006
3: -3.1392529, -1.9299777, -3.1392529, -1.9299777, -0.7247415, 0.7247412
4: -3.6569180, -2.8445487, -3.6569180, -2.8445487, -0.5500331, 0.5500331
5: -9.2376385, -8.4675541, -9.2376385, -8.4675541, -0.4301505, 0.4301505
6: -14.7871857, -13.7760391, -14.7871857, -13.7760391, -0.5901943, 0.5901946
7: 3.0898681, 3.8798275, 3.0898681, 3.8798275, -0.6977248, 0.6977248
8: -6.7023869, -5.8084626, -6.7023869, -5.8084626, -0.5841293, 0.5841293
9: -1.3172565, -0.6008897, -1.3172565, -0.6008897, -0.4359670, 0.4359670

Time for backsubstitution: 9.14 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 892

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3734007, upper bound: 0.3776909
time: 3.26 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3682870, upper bound: 0.3743267
time: 3.21 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -15.1882420, -13.9735336, -15.1723232, -13.9523754, -0.7498419, 0.7151599
1: -10.1070337, -9.0974617, -10.0923443, -9.0854988, -0.6016991, 0.5755692
2: -4.2032676, -3.3283267, -4.2004495, -3.3220515, -0.4838445, 0.4754239
3: -3.1392529, -1.9299777, -3.1783366, -1.9449842, -0.7217751, 0.7763138
4: -3.6569180, -2.8445487, -3.6441889, -2.8236494, -0.5745857, 0.5415385
5: -9.2376385, -8.4675541, -9.2672348, -8.4868479, -0.4263943, 0.4713502
6: -14.7871857, -13.7760391, -14.7993088, -13.7818928, -0.5808386, 0.5996865
7: 3.0898681, 3.8798275, 3.0985899, 3.9374599, -0.7524061, 0.6871543
8: -6.7023869, -5.8084626, -6.7380075, -5.8296738, -0.5893822, 0.6426244
9: -1.3172565, -0.6008897, -1.3114789, -0.5978093, -0.4419247, 0.4330646

Time for backsubstitution: 9.15 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 962

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 892

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3734007, upper bound: 0.3795904
time: 3.32 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3682870, upper bound: 0.3762265
time: 3.82 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -15.1723232, -13.9523754, -15.1882420, -13.9735336, -0.7151599, 0.7498419
1: -10.0923443, -9.0854988, -10.1070337, -9.0974617, -0.5755694, 0.6016994
2: -4.2004495, -3.3220515, -4.2032676, -3.3283267, -0.4754241, 0.4838448
3: -3.1783366, -1.9449842, -3.1392529, -1.9299777, -0.7763135, 0.7217751
4: -3.6441889, -2.8236494, -3.6569180, -2.8445487, -0.5415385, 0.5745857
5: -9.2672348, -8.4868479, -9.2376385, -8.4675541, -0.4713501, 0.4263942
6: -14.7993088, -13.7818928, -14.7871857, -13.7760391, -0.5996863, 0.5808384
7: 3.0985899, 3.9374599, 3.0898681, 3.8798275, -0.6871543, 0.7524056
8: -6.7380075, -5.8296738, -6.7023869, -5.8084626, -0.6426244, 0.5893822
9: -1.3114789, -0.5978093, -1.3172565, -0.6008897, -0.4330646, 0.4419246

Time for backsubstitution: 9.14 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 892

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3753647, upper bound: 0.3727337
time: 3.12 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3720002, upper bound: 0.3676160
time: 3.18 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -15.1723232, -13.9523754, -15.1723232, -13.9523754, -0.7298632, 0.7298630
1: -10.0923443, -9.0854988, -10.0923443, -9.0854988, -0.5907590, 0.5907590
2: -4.2004495, -3.3220515, -4.2004495, -3.3220515, -0.4776742, 0.4776742
3: -3.1783366, -1.9449842, -3.1783366, -1.9449842, -0.7326505, 0.7326505
4: -3.6441889, -2.8236494, -3.6441889, -2.8236494, -0.5530047, 0.5530047
5: -9.2672348, -8.4868479, -9.2672348, -8.4868479, -0.4338642, 0.4338642
6: -14.7993088, -13.7818928, -14.7993088, -13.7818928, -0.5941610, 0.5941610
7: 3.0985899, 3.9374599, 3.0985899, 3.9374599, -0.7303591, 0.7303588
8: -6.7380075, -5.8296738, -6.7380075, -5.8296738, -0.5875387, 0.5875386
9: -1.3114789, -0.5978093, -1.3114789, -0.5978093, -0.4343759, 0.4343759

Time for backsubstitution: 9.21 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 892

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3753650, upper bound: 0.3727337
time: 3.17 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3720007, upper bound: 0.3676160
time: 3.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 16.03 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 16.03
Output dim: 7, lower bound: -0.3734007, upper bound: 0.3776909
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 16.03
Output dim: 7, lower bound: -0.3682870, upper bound: 0.3743267
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 16.03
Output dim: 7, lower bound: -0.3734007, upper bound: 0.3795904
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 16.03
Output dim: 7, lower bound: -0.3682870, upper bound: 0.3762265
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.03
Output dim: 7, lower bound: -0.3753647, upper bound: 0.3727337
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.03
Output dim: 7, lower bound: -0.3720002, upper bound: 0.3676160
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.03
Output dim: 7, lower bound: -0.3753650, upper bound: 0.3727337
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.03
Output dim: 7, lower bound: -0.3720007, upper bound: 0.3676160

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -15.1882420, -13.9735336, -15.1878805, -13.9751263, -0.7237897, 0.7249299
1: -10.1070337, -9.0974617, -10.1063538, -9.0974617, -0.5872517, 0.5864911
2: -4.2032676, -3.3283267, -4.2030454, -3.3293409, -0.4754817, 0.4762267
3: -3.1392529, -1.9299777, -3.1376338, -1.9299858, -0.7247353, 0.7234197
4: -3.6569180, -2.8445487, -3.6562452, -2.8446603, -0.5499706, 0.5495808
5: -9.2376385, -8.4675541, -9.2375546, -8.4675598, -0.4301406, 0.4300704
6: -14.7871857, -13.7760391, -14.7849245, -13.7760658, -0.5901791, 0.5880926
7: 3.0898681, 3.8798275, 3.0901742, 3.8798275, -0.6973805, 0.6967459
8: -6.7023869, -5.8084626, -6.7023869, -5.8087626, -0.5827503, 0.5840960
9: -1.3172565, -0.6008897, -1.3172455, -0.6013479, -0.4357717, 0.4359603

Time for backsubstitution: 9.15 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3797867, upper bound: 0.3766354
time: 3.81 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3797867, upper bound: 0.3786676
time: 4.06 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -15.1878281, -13.9738941, -15.2499533, -13.9773932, -0.7304928, 0.7971790
1: -10.1054583, -9.0974617, -10.1120815, -9.0747585, -0.6223624, 0.6338973
2: -4.2030153, -3.3284559, -4.2446742, -3.3296411, -0.4804736, 0.5178137
3: -3.1389446, -1.9299932, -3.1432817, -1.8858368, -0.7574499, 0.7584178
4: -3.6556697, -2.8446758, -3.6437488, -2.8184557, -0.5797772, 0.5526755
5: -9.2375441, -8.4675674, -9.2410984, -8.4667749, -0.4314781, 0.4346318
6: -14.7845240, -13.7760916, -14.7773533, -13.7111483, -0.6647160, 0.6179340
7: 3.0903273, 3.8798275, 3.0809069, 3.8872199, -0.7068567, 0.7177029
8: -6.7023869, -5.8102121, -6.7039843, -5.8178005, -0.5952203, 0.6105742
9: -1.3172357, -0.6013088, -1.3337414, -0.5974953, -0.4567974, 0.4470474

Time for backsubstitution: 9.13 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 962

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3753523, upper bound: 0.3727828
time: 4.04 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3753523, upper bound: 0.3753522
time: 4.22 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -15.1882420, -13.9735336, -15.1719723, -13.9539700, -0.7485476, 0.7150065
1: -10.1070337, -9.0974617, -10.0916691, -9.0854988, -0.6016538, 0.5747602
2: -4.2032676, -3.3283267, -4.2002311, -3.3230653, -0.4830236, 0.4753590
3: -3.1392529, -1.9299777, -3.1767054, -1.9449930, -0.7217691, 0.7749908
4: -3.6569180, -2.8445487, -3.6435151, -2.8237615, -0.5745182, 0.5410874
5: -9.2376385, -8.4675541, -9.2671547, -8.4868565, -0.4263854, 0.4712662
6: -14.7871857, -13.7760391, -14.7970486, -13.7819204, -0.5808195, 0.5975820
7: 3.0898681, 3.8798275, 3.0989032, 3.9374599, -0.7520614, 0.6861720
8: -6.7023869, -5.8084626, -6.7380075, -5.8299685, -0.5880055, 0.6425912
9: -1.3172565, -0.6008897, -1.3114724, -0.5982690, -0.4417228, 0.4330571

Time for backsubstitution: 9.08 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 962

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3676929, upper bound: 0.3756457
time: 5.10 seconds

## Relational analysis of NS_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3695537, upper bound: 0.3755634
time: 3.68 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -15.1878281, -13.9738941, -15.2341366, -13.9562359, -0.7552514, 0.7871192
1: -10.1054583, -9.0974617, -10.0977745, -9.0627956, -0.6367643, 0.6222172
2: -4.2030153, -3.3284559, -4.2418194, -3.3233647, -0.4879334, 0.5171235
3: -3.1389446, -1.9299932, -3.1821885, -1.9008501, -0.7544708, 0.8098056
4: -3.6556697, -2.8446758, -3.6310153, -2.7976670, -0.6040952, 0.5441942
5: -9.2375441, -8.4675674, -9.2706776, -8.4860973, -0.4277509, 0.4757518
6: -14.7845240, -13.7760916, -14.7894983, -13.7170115, -0.6552403, 0.6272978
7: 3.0903273, 3.8798275, 3.0896993, 3.9448504, -0.7615356, 0.7070417
8: -6.7023869, -5.8102121, -6.7396064, -5.8389492, -0.6004894, 0.6690691
9: -1.3172357, -0.6013088, -1.3279679, -0.5943921, -0.4623182, 0.4441270

Time for backsubstitution: 9.15 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 962

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3644131, upper bound: 0.3704304
time: 3.29 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3636819, upper bound: 0.3716043
time: 3.24 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -15.1719723, -13.9539700, -15.1882420, -13.9735336, -0.7150066, 0.7485476
1: -10.0916691, -9.0854988, -10.1070337, -9.0974617, -0.5747602, 0.6016536
2: -4.2002311, -3.3230653, -4.2032676, -3.3283267, -0.4753591, 0.4830235
3: -3.1767054, -1.9449930, -3.1392529, -1.9299777, -0.7749906, 0.7217691
4: -3.6435151, -2.8237615, -3.6569180, -2.8445487, -0.5410874, 0.5745182
5: -9.2671547, -8.4868565, -9.2376385, -8.4675541, -0.4712662, 0.4263854
6: -14.7970486, -13.7819204, -14.7871857, -13.7760391, -0.5975820, 0.5808195
7: 3.0989032, 3.9374599, 3.0898681, 3.8798275, -0.6861720, 0.7520618
8: -6.7380075, -5.8299685, -6.7023869, -5.8084626, -0.6425915, 0.5880055
9: -1.3114724, -0.5982690, -1.3172565, -0.6008897, -0.4330571, 0.4417228

Time for backsubstitution: 9.27 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756456, upper bound: 0.3676929
time: 3.22 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3755628, upper bound: 0.3695536
time: 3.28 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -15.2341366, -13.9562359, -15.1878281, -13.9738941, -0.7871194, 0.7552514
1: -10.0977745, -9.0627956, -10.1054583, -9.0974617, -0.6222172, 0.6367643
2: -4.2418194, -3.3233647, -4.2030153, -3.3284559, -0.5171236, 0.4879334
3: -3.1821885, -1.9008501, -3.1389446, -1.9299932, -0.8098056, 0.7544706
4: -3.6310153, -2.7976670, -3.6556697, -2.8446758, -0.5441942, 0.6040952
5: -9.2706776, -8.4860973, -9.2375441, -8.4675674, -0.4757518, 0.4277511
6: -14.7894983, -13.7170115, -14.7845240, -13.7760916, -0.6272978, 0.6552403
7: 3.0896993, 3.9448504, 3.0903273, 3.8798275, -0.7070417, 0.7615361
8: -6.7396064, -5.8389492, -6.7023869, -5.8102121, -0.6690688, 0.6004894
9: -1.3279679, -0.5943921, -1.3172357, -0.6013088, -0.4441268, 0.4623183

Time for backsubstitution: 9.15 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 962

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3704306, upper bound: 0.3644129
time: 3.34 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3716045, upper bound: 0.3636820
time: 3.49 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -15.1719723, -13.9539700, -15.1723232, -13.9523754, -0.7297084, 0.7285683
1: -10.0916691, -9.0854988, -10.0923443, -9.0854988, -0.5899498, 0.5907142
2: -4.2002311, -3.3230653, -4.2004495, -3.3220515, -0.4775981, 0.4768554
3: -3.1767054, -1.9449930, -3.1783366, -1.9449842, -0.7313294, 0.7326448
4: -3.6435151, -2.8237615, -3.6441889, -2.8236494, -0.5525527, 0.5529423
5: -9.2671547, -8.4868565, -9.2672348, -8.4868479, -0.4337840, 0.4338543
6: -14.7970486, -13.7819204, -14.7993088, -13.7818928, -0.5920575, 0.5941459
7: 3.0989032, 3.9374599, 3.0985899, 3.9374599, -0.7293825, 0.7300155
8: -6.7380075, -5.8299685, -6.7380075, -5.8296738, -0.5875056, 0.5861609
9: -1.3114724, -0.5982690, -1.3114789, -0.5978093, -0.4343688, 0.4341813

Time for backsubstitution: 9.25 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 962

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3701602, upper bound: 0.3693436
time: 3.35 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3713500, upper bound: 0.3688487
time: 3.25 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -15.2341366, -13.9562359, -15.1719189, -13.9527369, -0.8017449, 0.7352720
1: -10.0977745, -9.0627956, -10.0907660, -9.0854988, -0.6371305, 0.6258304
2: -4.2418194, -3.3233647, -4.2002001, -3.3221817, -0.5191842, 0.4818424
3: -3.1821885, -1.9008501, -3.1780224, -1.9450018, -0.7663667, 0.7653553
4: -3.6310153, -2.7976670, -3.6429400, -2.8237765, -0.5556476, 0.5827076
5: -9.2706776, -8.4860973, -9.2671394, -8.4868641, -0.4383625, 0.4351919
6: -14.7894983, -13.7170115, -14.7966480, -13.7819462, -0.6218524, 0.6686621
7: 3.0896993, 3.9448504, 3.0990677, 3.9374599, -0.7502885, 0.7394924
8: -6.7396064, -5.8389492, -6.7380075, -5.8314295, -0.6139815, 0.5985589
9: -1.3279679, -0.5943921, -1.3114607, -0.5982313, -0.4454572, 0.4551600

Time for backsubstitution: 9.26 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 962

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3663382, upper bound: 0.3636242
time: 3.33 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3673853, upper bound: 0.3629779
time: 3.45 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 16.30 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3797867, upper bound: 0.3766354
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3797867, upper bound: 0.3786676
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3753523, upper bound: 0.3727828
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3753523, upper bound: 0.3753522
NS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3676929, upper bound: 0.3756457
NS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3695537, upper bound: 0.3755634
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3644131, upper bound: 0.3704304
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3636819, upper bound: 0.3716043
NS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3756456, upper bound: 0.3676929
NS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3755628, upper bound: 0.3695536
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3704306, upper bound: 0.3644129
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3716045, upper bound: 0.3636820
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3701602, upper bound: 0.3693436
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3713500, upper bound: 0.3688487
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3663382, upper bound: 0.3636242
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.30
Output dim: 7, lower bound: -0.3673853, upper bound: 0.3629779

## BFS NS instance: NS_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -15.1858892, -13.9737854, -15.1878805, -13.9751263, -0.7225692, 0.7226585
1: -10.1060572, -9.0976896, -10.1063538, -9.0974617, -0.5862916, 0.5836849
2: -4.2003961, -3.3283615, -4.2030454, -3.3293409, -0.4693714, 0.4757833
3: -3.1295147, -1.9299803, -3.1376338, -1.9299858, -0.7109971, 0.7234194
4: -3.6516995, -2.8452163, -3.6562452, -2.8446603, -0.5448897, 0.5492052
5: -9.2371349, -8.4703426, -9.2375546, -8.4675598, -0.4299136, 0.4248524
6: -14.7863903, -13.7813835, -14.7849245, -13.7760658, -0.5896682, 0.5803500
7: 3.0910525, 3.8776870, 3.0901742, 3.8798275, -0.6960387, 0.6939712
8: -6.7023869, -5.8123293, -6.7023869, -5.8087626, -0.5827503, 0.5733786
9: -1.3172545, -0.6013570, -1.3172455, -0.6013479, -0.4357715, 0.4351312

Time for backsubstitution: 9.17 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 962

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3767168, upper bound: 0.3766354
time: 4.05 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3767168, upper bound: 0.3766346
time: 5.47 seconds

## BFS NS instance: NS_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -15.1950817, -13.9740372, -15.1869755, -13.9752464, -0.7369468, 0.7279911
1: -10.1227522, -9.0977125, -10.1060047, -9.0975218, -0.6060257, 0.5902669
2: -4.1988368, -3.3257303, -4.2012625, -3.3293500, -0.4738246, 0.4898436
3: -3.1182046, -1.9187458, -3.1314750, -1.9299874, -0.7120907, 0.7608771
4: -3.6530707, -2.8330226, -3.6548178, -2.8449280, -0.5464101, 0.5655673
5: -9.2404652, -8.4742565, -9.2373524, -8.4694633, -0.4451170, 0.4254618
6: -14.7959003, -13.7867746, -14.7846088, -13.7790298, -0.6126258, 0.5798513
7: 3.0781388, 3.8775482, 3.0906410, 3.8792162, -0.7092986, 0.6934285
8: -6.7127738, -5.8186502, -6.7023869, -5.8122950, -0.5950544, 0.5781102
9: -1.3178105, -0.5962877, -1.3172457, -0.6015196, -0.4394217, 0.4381276

Time for backsubstitution: 9.40 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 962

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3767168, upper bound: 0.3786677
time: 4.35 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3767168, upper bound: 0.3786677
time: 4.17 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -15.1854448, -13.9741459, -15.2499533, -13.9773932, -0.7292595, 0.7949080
1: -10.1045208, -9.0976896, -10.1120815, -9.0747585, -0.6213653, 0.6310911
2: -4.2001381, -3.3284903, -4.2446742, -3.3296411, -0.4743779, 0.5173700
3: -3.1292055, -1.9299951, -3.1432817, -1.8858368, -0.7437596, 0.7584176
4: -3.6504526, -2.8453534, -3.6437488, -2.8184557, -0.5746970, 0.5522959
5: -9.2370338, -8.4703560, -9.2410984, -8.4667749, -0.4312493, 0.4294149
6: -14.7837324, -13.7814350, -14.7773533, -13.7111483, -0.6642540, 0.6101921
7: 3.0914974, 3.8776870, 3.0809069, 3.8872199, -0.7054939, 0.7149277
8: -6.7023869, -5.8140402, -6.7039843, -5.8178005, -0.5952203, 0.5998766
9: -1.3172355, -0.6017718, -1.3337414, -0.5974953, -0.4567969, 0.4462242

Time for backsubstitution: 9.20 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 962

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3727822, upper bound: 0.3727828
time: 4.11 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3727822, upper bound: 0.3727824
time: 4.25 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -15.1945839, -13.9743977, -15.2489786, -13.9775143, -0.7436502, 0.8003178
1: -10.1213169, -9.0977125, -10.1115875, -9.0748186, -0.6413066, 0.6376312
2: -4.1985450, -3.3258636, -4.2432570, -3.3296514, -0.4795666, 0.5310704
3: -3.1178732, -1.9187608, -3.1371958, -1.8858371, -0.7448249, 0.7954817
4: -3.6518221, -2.8331718, -3.6423144, -2.8187246, -0.5761864, 0.5686774
5: -9.2403469, -8.4742680, -9.2408524, -8.4686794, -0.4464636, 0.4299966
6: -14.7932596, -13.7868242, -14.7768784, -13.7141113, -0.6875238, 0.6097062
7: 3.0784950, 3.8775482, 3.0814686, 3.8866072, -0.7188263, 0.7146134
8: -6.7127738, -5.8203864, -6.7039843, -5.8213167, -0.6077321, 0.6046696
9: -1.3177927, -0.5966816, -1.3337412, -0.5977559, -0.4604775, 0.4492012

Time for backsubstitution: 9.13 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3727822, upper bound: 0.3753531
time: 3.85 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3727822, upper bound: 0.3753521
time: 4.12 seconds

## BFS NS instance: NS_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -15.1882420, -13.9735336, -15.1696453, -13.9542217, -0.7462764, 0.7137970
1: -10.1070337, -9.0974617, -10.0907631, -9.0857277, -0.5988479, 0.5738153
2: -4.2032676, -3.3283267, -4.1973519, -3.3231006, -0.4825758, 0.4692841
3: -3.1392529, -1.9299777, -3.1669431, -1.9449942, -0.7217684, 0.7611833
4: -3.6569180, -2.8445487, -3.6383035, -2.8244429, -0.5741050, 0.5360036
5: -9.2376385, -8.4675541, -9.2666445, -8.4896526, -0.4211848, 0.4710113
6: -14.7871857, -13.7760391, -14.7962551, -13.7872620, -0.5730737, 0.5970609
7: 3.0898681, 3.8798275, 3.1000557, 3.9353185, -0.7492857, 0.6848683
8: -6.7023869, -5.8084626, -6.7380075, -5.8338156, -0.5773063, 0.6425912
9: -1.3172565, -0.6008897, -1.3114717, -0.5987275, -0.4408444, 0.4330568

Time for backsubstitution: 9.38 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 962

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A1_B2_B1_B1_A1

### Relational analysis result of NS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3676929, upper bound: 0.3742836
time: 3.89 seconds

## Relational analysis of NS_A1_B2_B1_B1_A2

### Relational analysis result of NS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3676929, upper bound: 0.3755634
time: 3.67 seconds

## BFS NS instance: NS_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -15.1873331, -13.9736557, -15.1790466, -13.9544735, -0.7516069, 0.7283969
1: -10.1066751, -9.0975218, -10.1077995, -9.0857506, -0.6054287, 0.5933232
2: -4.2014718, -3.3283339, -4.1958027, -3.3204751, -0.4965323, 0.4738606
3: -3.1330731, -1.9299798, -3.1555858, -1.9337614, -0.7592425, 0.7621286
4: -3.6554885, -2.8448181, -3.6397538, -2.8122725, -0.5903730, 0.5375485
5: -9.2374344, -8.4694576, -9.2699738, -8.4936247, -0.4220228, 0.4861505
6: -14.7868671, -13.7790070, -14.8057919, -13.7926464, -0.5725060, 0.6197731
7: 3.0903430, 3.8792162, 3.0870342, 3.9351797, -0.7487407, 0.6981883
8: -6.7023869, -5.8119984, -6.7483945, -5.8400116, -0.5820346, 0.6548848
9: -1.3172545, -0.6010666, -1.3120298, -0.5936623, -0.4436383, 0.4367052

Time for backsubstitution: 9.26 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 962

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3695537, upper bound: 0.3742832
time: 3.37 seconds

## Relational analysis of NS_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3695537, upper bound: 0.3755628
time: 3.32 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -15.1854448, -13.9741459, -15.2341366, -13.9562359, -0.7540181, 0.7848485
1: -10.1045208, -9.0976896, -10.0977745, -9.0627956, -0.6357672, 0.6194110
2: -4.2001381, -3.3284903, -4.2418194, -3.3233647, -0.4818375, 0.5166799
3: -3.1292055, -1.9299951, -3.1821885, -1.9008501, -0.7407808, 0.8098054
4: -3.6504526, -2.8453534, -3.6310153, -2.7976670, -0.5990150, 0.5438147
5: -9.2370338, -8.4703560, -9.2706776, -8.4860973, -0.4275221, 0.4705348
6: -14.7837324, -13.7814350, -14.7894983, -13.7170115, -0.6547785, 0.6195560
7: 3.0914974, 3.8776870, 3.0896993, 3.9448504, -0.7601728, 0.7042665
8: -6.7023869, -5.8140402, -6.7396064, -5.8389492, -0.6004894, 0.6583714
9: -1.3172355, -0.6017718, -1.3279679, -0.5943921, -0.4623179, 0.4433038

Time for backsubstitution: 9.14 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 962

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3626251, upper bound: 0.3704304
time: 3.40 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3626251, upper bound: 0.3704310
time: 3.81 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -15.1945839, -13.9743977, -15.2331858, -13.9563580, -0.7684083, 0.7902546
1: -10.1213169, -9.0977125, -10.0973110, -9.0628586, -0.6557090, 0.6259665
2: -4.1985450, -3.3258636, -4.2403951, -3.3233738, -0.4870249, 0.5304174
3: -3.1178732, -1.9187608, -3.1760798, -1.9008515, -0.7418461, 0.8468645
4: -3.6518221, -2.8331718, -3.6295815, -2.7979364, -0.6004941, 0.5601954
5: -9.2403469, -8.4742680, -9.2704315, -8.4880018, -0.4427404, 0.4711165
6: -14.7932596, -13.7868242, -14.7890263, -13.7199726, -0.6780472, 0.6190659
7: 3.0784950, 3.8775482, 3.0902443, 3.9442401, -0.7735071, 0.7039766
8: -6.7127738, -5.8203864, -6.7396064, -5.8424621, -0.6130226, 0.6631644
9: -1.3177927, -0.5966816, -1.3279679, -0.5946503, -0.4659832, 0.4462806

Time for backsubstitution: 9.19 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 962

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3626251, upper bound: 0.3716044
time: 3.47 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3626251, upper bound: 0.3716053
time: 4.50 seconds

## BFS NS instance: NS_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -15.1696453, -13.9542217, -15.1882420, -13.9735336, -0.7137971, 0.7462764
1: -10.0907631, -9.0857277, -10.1070337, -9.0974617, -0.5738153, 0.5988479
2: -4.1973519, -3.3231006, -4.2032676, -3.3283267, -0.4692839, 0.4825759
3: -3.1669431, -1.9449942, -3.1392529, -1.9299777, -0.7611835, 0.7217684
4: -3.6383035, -2.8244429, -3.6569180, -2.8445487, -0.5360036, 0.5741050
5: -9.2666445, -8.4896526, -9.2376385, -8.4675541, -0.4710113, 0.4211847
6: -14.7962551, -13.7872620, -14.7871857, -13.7760391, -0.5970608, 0.5730737
7: 3.1000557, 3.9353185, 3.0898681, 3.8798275, -0.6848683, 0.7492857
8: -6.7380075, -5.8338156, -6.7023869, -5.8084626, -0.6425915, 0.5773063
9: -1.3114717, -0.5987275, -1.3172565, -0.6008897, -0.4330568, 0.4408444

Time for backsubstitution: 9.16 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 962

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3742833, upper bound: 0.3676925
time: 3.80 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3742833, upper bound: 0.3676937
time: 3.89 seconds

## BFS NS instance: NS_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -15.1790466, -13.9544735, -15.1873331, -13.9736557, -0.7283969, 0.7516069
1: -10.1077995, -9.0857506, -10.1066751, -9.0975218, -0.5933235, 0.6054287
2: -4.1958027, -3.3204751, -4.2014718, -3.3283339, -0.4738606, 0.4965324
3: -3.1555858, -1.9337614, -3.1330731, -1.9299798, -0.7621288, 0.7592423
4: -3.6397538, -2.8122725, -3.6554885, -2.8448181, -0.5375483, 0.5903730
5: -9.2699738, -8.4936247, -9.2374344, -8.4694576, -0.4861506, 0.4220228
6: -14.8057919, -13.7926464, -14.7868671, -13.7790070, -0.6197731, 0.5725060
7: 3.0870342, 3.9351797, 3.0903430, 3.8792162, -0.6981883, 0.7487409
8: -6.7483945, -5.8400116, -6.7023869, -5.8119984, -0.6548848, 0.5820346
9: -1.3120298, -0.5936623, -1.3172545, -0.6010666, -0.4367054, 0.4436383

Time for backsubstitution: 9.45 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 962

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3742833, upper bound: 0.3695534
time: 3.67 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3742833, upper bound: 0.3695545
time: 4.03 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -15.2341366, -13.9562359, -15.1854448, -13.9741459, -0.7848485, 0.7540181
1: -10.0977745, -9.0627956, -10.1045208, -9.0976896, -0.6194108, 0.6357672
2: -4.2418194, -3.3233647, -4.2001381, -3.3284903, -0.5166799, 0.4818376
3: -3.1821885, -1.9008501, -3.1292055, -1.9299951, -0.8098052, 0.7407806
4: -3.6310153, -2.7976670, -3.6504526, -2.8453534, -0.5438147, 0.5990148
5: -9.2706776, -8.4860973, -9.2370338, -8.4703560, -0.4705348, 0.4275222
6: -14.7894983, -13.7170115, -14.7837324, -13.7814350, -0.6195558, 0.6547785
7: 3.0896993, 3.9448504, 3.0914974, 3.8776870, -0.7042665, 0.7601731
8: -6.7396064, -5.8389492, -6.7023869, -5.8140402, -0.6583712, 0.6004894
9: -1.3279679, -0.5943921, -1.3172355, -0.6017718, -0.4433039, 0.4623179

Time for backsubstitution: 9.30 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 962

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3704305, upper bound: 0.3626248
time: 3.49 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3704305, upper bound: 0.3636820
time: 3.44 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -15.2331858, -13.9563580, -15.1945839, -13.9743977, -0.7902546, 0.7684083
1: -10.0973110, -9.0628586, -10.1213169, -9.0977125, -0.6259665, 0.6557090
2: -4.2403951, -3.3233738, -4.1985450, -3.3258636, -0.5304176, 0.4870250
3: -3.1760798, -1.9008515, -3.1178732, -1.9187608, -0.8468647, 0.7418458
4: -3.6295815, -2.7979364, -3.6518221, -2.8331718, -0.5601954, 0.6004941
5: -9.2704315, -8.4880018, -9.2403469, -8.4742680, -0.4711165, 0.4427404
6: -14.7890263, -13.7199726, -14.7932596, -13.7868242, -0.6190658, 0.6780472
7: 3.0902443, 3.9442401, 3.0784950, 3.8775482, -0.7039766, 0.7735069
8: -6.7396064, -5.8424621, -6.7127738, -5.8203864, -0.6631646, 0.6130226
9: -1.3279679, -0.5946503, -1.3177927, -0.5966816, -0.4462806, 0.4659832

Time for backsubstitution: 9.21 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3716046, upper bound: 0.3626247
time: 3.43 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3716046, upper bound: 0.3636819
time: 3.51 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -15.1719723, -13.9539700, -15.1700344, -13.9526291, -0.7274375, 0.7273471
1: -10.0916691, -9.0854988, -10.0914307, -9.0857277, -0.5871439, 0.5897424
2: -4.2002311, -3.3230653, -4.1975803, -3.3220847, -0.4771546, 0.4707013
3: -3.1767054, -1.9449930, -3.1685529, -1.9449844, -0.7313290, 0.7188666
4: -3.6435151, -2.8237615, -3.6389780, -2.8243198, -0.5521781, 0.5478619
5: -9.2671547, -8.4868565, -9.2667370, -8.4896431, -0.4285673, 0.4336267
6: -14.7970486, -13.7819204, -14.7985210, -13.7872362, -0.5843158, 0.5936316
7: 3.0989032, 3.9374599, 3.0997381, 3.9353185, -0.7266073, 0.7286880
8: -6.7380075, -5.8299685, -6.7380075, -5.8335090, -0.5767984, 0.5861609
9: -1.3114724, -0.5982690, -1.3114815, -0.5982704, -0.4335486, 0.4341811

Time for backsubstitution: 9.24 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 962

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3701602, upper bound: 0.3670163
time: 3.46 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3701602, upper bound: 0.3688485
time: 3.42 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -15.1710882, -13.9540901, -15.1794195, -13.9528790, -0.7327693, 0.7416418
1: -10.0913429, -9.0855579, -10.1084232, -9.0857506, -0.5937223, 0.6093831
2: -4.1984401, -3.3230753, -4.1960220, -3.3194592, -0.4912035, 0.4751678
3: -3.1705236, -1.9449935, -3.1571178, -1.9337506, -0.7688191, 0.7200596
4: -3.6420894, -2.8240299, -3.6404266, -2.8121598, -0.5684948, 0.5493836
5: -9.2669544, -8.4887619, -9.2700653, -8.4936180, -0.4291750, 0.4488136
6: -14.7967367, -13.7848873, -14.8080215, -13.7926197, -0.5838162, 0.6165254
7: 3.0993543, 3.9368467, 3.0867958, 3.9351797, -0.7260704, 0.7418256
8: -6.7380075, -5.8334813, -6.7483945, -5.8397064, -0.5814643, 0.5984683
9: -1.3114719, -0.5984397, -1.3120370, -0.5932131, -0.4365461, 0.4378334

Time for backsubstitution: 9.35 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 1859
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 962

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3713501, upper bound: 0.3670165
time: 3.41 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3713501, upper bound: 0.3688488
time: 3.52 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -15.2341366, -13.9562359, -15.1695948, -13.9529905, -0.7994742, 0.7340372
1: -10.0977745, -9.0627956, -10.0898943, -9.0857277, -0.6343246, 0.6248224
2: -4.2418194, -3.3233647, -4.1973243, -3.3222170, -0.5187405, 0.4757022
3: -3.1821885, -1.9008501, -3.1682377, -1.9450016, -0.7663667, 0.7516241
4: -3.6310153, -2.7976670, -3.6377292, -2.8244562, -0.5552683, 0.5776277
5: -9.2706776, -8.4860973, -9.2666349, -8.4896603, -0.4331468, 0.4349632
6: -14.7894983, -13.7170115, -14.7958603, -13.7872896, -0.6141115, 0.6681931
7: 3.0896993, 3.9448504, 3.1002007, 3.9353185, -0.7475133, 0.7381458
8: -6.7396064, -5.8389492, -6.7380075, -5.8352313, -0.6032944, 0.5985589
9: -1.3279679, -0.5943921, -1.3114605, -0.5986884, -0.4446374, 0.4551595

Time for backsubstitution: 9.25 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 962

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3663382, upper bound: 0.3618341
time: 3.53 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3663382, upper bound: 0.3629779
time: 3.56 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -15.2331858, -13.9563580, -15.1789303, -13.9532394, -0.8048713, 0.7483445
1: -10.0973110, -9.0628586, -10.1069860, -9.0857506, -0.6408572, 0.6446593
2: -4.2403951, -3.3233738, -4.1957350, -3.3195915, -0.5324688, 0.4809151
3: -3.1760798, -1.9008515, -3.1567807, -1.9337673, -0.8034461, 0.7527959
4: -3.6295815, -2.7979364, -3.6391778, -2.8123081, -0.5716047, 0.5791168
5: -9.2704315, -8.4880018, -9.2699490, -8.4936314, -0.4337285, 0.4501604
6: -14.7890263, -13.7199726, -14.8053751, -13.7926731, -0.6136284, 0.6914105
7: 3.0902443, 3.9442401, 3.0871630, 3.9351797, -0.7472029, 0.7513554
8: -6.7396064, -5.8424621, -6.7483945, -5.8414469, -0.6080246, 0.6110796
9: -1.3279679, -0.5946503, -1.3120198, -0.5936093, -0.4476212, 0.4588406

Time for backsubstitution: 9.25 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 892
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 962

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3673854, upper bound: 0.3618341
time: 3.46 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3673854, upper bound: 0.3629779
time: 3.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 16.41 seconds
NS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3767168, upper bound: 0.3766354
NS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3767168, upper bound: 0.3766346
NS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3767168, upper bound: 0.3786677
NS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3767168, upper bound: 0.3786677
NS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3727822, upper bound: 0.3727828
NS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3727822, upper bound: 0.3727824
NS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3727822, upper bound: 0.3753531
NS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3727822, upper bound: 0.3753521
NS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3676929, upper bound: 0.3742836
NS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3676929, upper bound: 0.3755634
NS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3695537, upper bound: 0.3742832
NS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3695537, upper bound: 0.3755628
NS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3626251, upper bound: 0.3704304
NS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3626251, upper bound: 0.3704310
NS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3626251, upper bound: 0.3716044
NS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3626251, upper bound: 0.3716053
NS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3742833, upper bound: 0.3676925
NS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3742833, upper bound: 0.3676937
NS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3742833, upper bound: 0.3695534
NS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3742833, upper bound: 0.3695545
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3704305, upper bound: 0.3626248
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3704305, upper bound: 0.3636820
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3716046, upper bound: 0.3626247
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3716046, upper bound: 0.3636819
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3701602, upper bound: 0.3670163
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3701602, upper bound: 0.3688485
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3713501, upper bound: 0.3670165
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3713501, upper bound: 0.3688488
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3663382, upper bound: 0.3618341
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3663382, upper bound: 0.3629779
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3673854, upper bound: 0.3618341
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.41
Output dim: 7, lower bound: -0.3673854, upper bound: 0.3629779

## BFS NS instance: NS_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -15.1858892, -13.9737854, -15.1854916, -13.9753819, -0.7202981, 0.7214222
1: -10.1060572, -9.0976896, -10.1053839, -9.0976896, -0.5834854, 0.5827131
2: -4.2003961, -3.3283615, -4.2001653, -3.3293757, -0.4689279, 0.4696840
3: -3.1295147, -1.9299803, -3.1279156, -1.9299846, -0.7109971, 0.7096620
4: -3.6516995, -2.8452163, -3.6510274, -2.8453383, -0.5445085, 0.5441246
5: -9.2371349, -8.4703426, -9.2370434, -8.4703484, -0.4246961, 0.4246228
6: -14.7863903, -13.7813835, -14.7841282, -13.7814083, -0.5819262, 0.5798396
7: 3.0910525, 3.8776870, 3.0913644, 3.8776870, -0.6932631, 0.6926055
8: -6.7023869, -5.8123293, -6.7023869, -5.8126369, -0.5720150, 0.5733786
9: -1.3172545, -0.6013570, -1.3172438, -0.6018095, -0.4349391, 0.4351308

Time for backsubstitution: 9.18 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 1493

## Relational analysis of NS_A1_B1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3717531, upper bound: 0.3746644
time: 3.55 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3769294, upper bound: 0.3756772
time: 3.72 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -15.1858892, -13.9737854, -15.1947012, -13.9756317, -0.7197926, 0.7372334
1: -10.1060572, -9.0976896, -10.1221237, -9.0977125, -0.5832024, 0.6030710
2: -4.2003961, -3.3283615, -4.1986141, -3.3267469, -0.4841753, 0.4727244
3: -3.1295147, -1.9299803, -3.1166828, -1.9187539, -0.7510605, 0.7185450
4: -3.6516995, -2.8452163, -3.6523976, -2.8331361, -0.5622544, 0.5463693
5: -9.2371349, -8.4703426, -9.2403784, -8.4742651, -0.4294064, 0.4402618
6: -14.7863903, -13.7813835, -14.7936764, -13.7867985, -0.5867913, 0.6043973
7: 3.0910525, 3.8776870, 3.0783749, 3.8775482, -0.6930947, 0.7066634
8: -6.7023869, -5.8123293, -6.7127738, -5.8189597, -0.5752409, 0.5880617
9: -1.3172545, -0.6013570, -1.3178015, -0.5967331, -0.4379236, 0.4390597

Time for backsubstitution: 9.12 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1493
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 2082
type: A, layer: 3, pos: 892
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 2082
type: B, layer: 3, pos: 570
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 2586
type: A, layer: 3, pos: 1859
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2586
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 962

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 1493

## Relational analysis of NS_A1_B1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3717531, upper bound: 0.3746650
time: 3.80 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3769294, upper bound: 0.3756772
time: 3.89 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.61 + 545.46 = 603.07 seconds
