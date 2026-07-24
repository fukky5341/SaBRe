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
execution time: IAR + RelationalAnalysis = 24.03 + 34.28 = 58.31 seconds
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
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 1236

Time for candidate selection: 0.58 seconds

### Candidate
type: A, layer: 3, pos: 2818

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3817583, upper bound: 0.3893787
time: 4.03 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3861259, upper bound: 0.3861264
time: 3.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.05 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.05
Output dim: 7, lower bound: -0.3817583, upper bound: 0.3893787
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.05
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

Time for backsubstitution: 8.46 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 1236

Time for candidate selection: 0.57 seconds

### Candidate
type: B, layer: 3, pos: 2818

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3815192, upper bound: 0.3815199
time: 4.28 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3815192, upper bound: 0.3861274
time: 3.97 seconds

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

Time for backsubstitution: 8.51 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 1236

Time for candidate selection: 0.59 seconds

### Candidate
type: B, layer: 3, pos: 2818

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3861265, upper bound: 0.3815187
time: 3.65 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3861265, upper bound: 0.3861264
time: 3.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 16.41 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.41
Output dim: 7, lower bound: -0.3815192, upper bound: 0.3815199
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.41
Output dim: 7, lower bound: -0.3815192, upper bound: 0.3861274
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.41
Output dim: 7, lower bound: -0.3861265, upper bound: 0.3815187
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.41
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

Time for backsubstitution: 8.99 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 1236

Time for candidate selection: 0.44 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3785997, upper bound: 0.3796272
time: 3.84 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3785997, upper bound: 0.3822798
time: 3.93 seconds

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

Time for backsubstitution: 9.19 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 1236

Time for candidate selection: 0.53 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3785997, upper bound: 0.3838545
time: 3.64 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3785997, upper bound: 0.3862037
time: 5.74 seconds

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

Time for backsubstitution: 9.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 1236

Time for candidate selection: 0.47 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3832095, upper bound: 0.3760052
time: 3.63 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3832091, upper bound: 0.3783539
time: 4.06 seconds

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

Time for backsubstitution: 9.04 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 1236

Time for candidate selection: 0.48 seconds

### Candidate
type: A, layer: 3, pos: 2809

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3832097, upper bound: 0.3760053
time: 3.87 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3832097, upper bound: 0.3783541
time: 3.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 17.12 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.12
Output dim: 7, lower bound: -0.3785997, upper bound: 0.3796272
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.12
Output dim: 7, lower bound: -0.3785997, upper bound: 0.3822798
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.12
Output dim: 7, lower bound: -0.3785997, upper bound: 0.3838545
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.12
Output dim: 7, lower bound: -0.3785997, upper bound: 0.3862037
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.12
Output dim: 7, lower bound: -0.3832095, upper bound: 0.3760052
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.12
Output dim: 7, lower bound: -0.3832091, upper bound: 0.3783539
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.12
Output dim: 7, lower bound: -0.3832097, upper bound: 0.3760053
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.12
Output dim: 7, lower bound: -0.3832097, upper bound: 0.3783541

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -15.1858892, -13.9737854, -15.1882420, -13.9735336, -0.7238641, 0.7228130
1: -10.1060572, -9.0976896, -10.1070337, -9.0974617, -0.5863371, 0.5844913
2: -4.2003961, -3.3283615, -4.2032676, -3.3283267, -0.4701902, 0.4758571
3: -3.1295147, -1.9299803, -3.1392529, -1.9299777, -0.7110031, 0.7247410
4: -3.6516995, -2.8452163, -3.6569180, -2.8445487, -0.5449522, 0.5496575
5: -9.2371349, -8.4703426, -9.2376385, -8.4675541, -0.4299235, 0.4249326
6: -14.7863903, -13.7813835, -14.7871857, -13.7760391, -0.5896837, 0.5824521
7: 3.0910525, 3.8776870, 3.0898681, 3.8798275, -0.6963825, 0.6949501
8: -6.7023869, -5.8123293, -6.7023869, -5.8084626, -0.5841293, 0.5734119
9: -1.3172545, -0.6013570, -1.3172565, -0.6008897, -0.4359666, 0.4351379

Time for backsubstitution: 9.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 1236

Time for candidate selection: 0.45 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3818124, upper bound: 0.3818123
time: 7.31 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3818124, upper bound: 0.3818129
time: 4.11 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -15.1950817, -13.9740372, -15.1873331, -13.9736557, -0.7382412, 0.7281437
1: -10.1227522, -9.0977125, -10.1066751, -9.0975218, -0.6060712, 0.5910721
2: -4.1988368, -3.3257303, -4.2014718, -3.3283339, -0.4746432, 0.4899166
3: -3.1182046, -1.9187458, -3.1330731, -1.9299798, -0.7120967, 0.7622144
4: -3.6530707, -2.8330226, -3.6554885, -2.8448181, -0.5464718, 0.5660195
5: -9.2404652, -8.4742565, -9.2374344, -8.4694576, -0.4451268, 0.4255403
6: -14.7959003, -13.7867746, -14.7868671, -13.7790070, -0.6126411, 0.5819497
7: 3.0781388, 3.8775482, 3.0903430, 3.8792162, -0.7096424, 0.6944053
8: -6.7127738, -5.8186502, -6.7023869, -5.8119984, -0.5964313, 0.5781435
9: -1.3178105, -0.5962877, -1.3172545, -0.6010666, -0.4396150, 0.4381343

Time for backsubstitution: 9.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 1236

Time for candidate selection: 0.47 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3818124, upper bound: 0.3848788
time: 3.74 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3818124, upper bound: 0.3848788
time: 4.01 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -15.1858892, -13.9737854, -15.1723232, -13.9523754, -0.7486217, 0.7128885
1: -10.1060572, -9.0976896, -10.0923443, -9.0854988, -0.6007390, 0.5727632
2: -4.2003961, -3.3283615, -4.2004495, -3.3220515, -0.4777343, 0.4749806
3: -3.1295147, -1.9299803, -3.1783366, -1.9449842, -0.7080367, 0.7763133
4: -3.6516995, -2.8452163, -3.6441889, -2.8236494, -0.5695047, 0.5411630
5: -9.2371349, -8.4703426, -9.2672348, -8.4868479, -0.4261674, 0.4661322
6: -14.7863903, -13.7813835, -14.7993088, -13.7818928, -0.5803276, 0.5919440
7: 3.0910525, 3.8776870, 3.0985899, 3.9374599, -0.7510633, 0.6843796
8: -6.7023869, -5.8123293, -6.7380075, -5.8296738, -0.5893822, 0.6319070
9: -1.3172545, -0.6013570, -1.3114789, -0.5978093, -0.4419242, 0.4322355

Time for backsubstitution: 8.43 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 1236

Time for candidate selection: 0.45 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3761616, upper bound: 0.3838539
time: 3.87 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3761616, upper bound: 0.3838527
time: 4.10 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -15.1950817, -13.9740372, -15.1714344, -13.9524956, -0.7629991, 0.7182264
1: -10.1227522, -9.0977125, -10.0920086, -9.0855579, -0.6204734, 0.5793478
2: -4.1988368, -3.3257303, -4.1986470, -3.3220587, -0.4821866, 0.4890616
3: -3.1182046, -1.9187458, -3.1721306, -1.9449854, -0.7091300, 0.8138127
4: -3.6530707, -2.8330226, -3.6427622, -2.8239202, -0.5710125, 0.5575244
5: -9.2404652, -8.4742565, -9.2670345, -8.4887514, -0.4413743, 0.4667306
6: -14.7959003, -13.7867746, -14.7989941, -13.7848587, -0.6032847, 0.5914392
7: 3.0781388, 3.8775482, 3.0990486, 3.9368467, -0.7643232, 0.6838593
8: -6.7127738, -5.8186502, -6.7380075, -5.8331885, -0.6016972, 0.6366384
9: -1.3178105, -0.5962877, -1.3114810, -0.5979836, -0.4455557, 0.4352320

Time for backsubstitution: 9.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 1236

Time for candidate selection: 0.44 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3761616, upper bound: 0.3862040
time: 3.91 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3761616, upper bound: 0.3862031
time: 4.41 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -15.1700344, -13.9526291, -15.1882420, -13.9735336, -0.7139654, 0.7475710
1: -10.0914307, -9.0857277, -10.1070337, -9.0974617, -0.5746305, 0.5988934
2: -4.1975803, -3.3220847, -4.2032676, -3.3283267, -0.4693372, 0.4833970
3: -3.1685529, -1.9449844, -3.1392529, -1.9299777, -0.7625265, 0.7217751
4: -3.6389780, -2.8243198, -3.6569180, -2.8445487, -0.5364552, 0.5741787
5: -9.2667370, -8.4896431, -9.2376385, -8.4675541, -0.4710981, 0.4211931
6: -14.7985210, -13.7872362, -14.7871857, -13.7760391, -0.5991684, 0.5730927
7: 3.0997381, 3.9353185, 3.0898681, 3.8798275, -0.6858721, 0.7496300
8: -6.7380075, -5.8335090, -6.7023869, -5.8084626, -0.6426244, 0.5787010
9: -1.3114815, -0.5982704, -1.3172565, -0.6008897, -0.4330643, 0.4410537

Time for backsubstitution: 9.26 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 1236

Time for candidate selection: 0.56 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3838532, upper bound: 0.3761623
time: 3.91 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3838532, upper bound: 0.3761613
time: 4.53 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -15.1794195, -13.9528790, -15.1873331, -13.9736557, -0.7285321, 0.7529018
1: -10.1084232, -9.0857506, -10.1066751, -9.0975218, -0.5946112, 0.6054742
2: -4.1960220, -3.3194592, -4.2014718, -3.3283339, -0.4739084, 0.4973551
3: -3.1571178, -1.9337506, -3.1330731, -1.9299798, -0.7635722, 0.7592480
4: -3.6404266, -2.8121598, -3.6554885, -2.8448181, -0.5379782, 0.5904212
5: -9.2700653, -8.4936180, -9.2374344, -8.4694576, -0.4862212, 0.4220314
6: -14.8080215, -13.7926197, -14.7868671, -13.7790070, -0.6220419, 0.5725245
7: 3.0867958, 3.9351797, 3.0903430, 3.8792162, -0.6991339, 0.7490852
8: -6.7483945, -5.8397064, -6.7023869, -5.8119984, -0.6549263, 0.5834398
9: -1.3120370, -0.5932131, -1.3172545, -0.6010666, -0.4367130, 0.4438465

Time for backsubstitution: 9.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 1236

Time for candidate selection: 0.45 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3838532, upper bound: 0.3786004
time: 3.88 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3838532, upper bound: 0.3785994
time: 5.12 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -15.1700344, -13.9526291, -15.1723232, -13.9523754, -0.7286417, 0.7275919
1: -10.0914307, -9.0857277, -10.0923443, -9.0854988, -0.5897875, 0.5879533
2: -4.1975803, -3.3220847, -4.2004495, -3.3220515, -0.4715203, 0.4772308
3: -3.1685529, -1.9449844, -3.1783366, -1.9449842, -0.7188723, 0.7326500
4: -3.6389780, -2.8243198, -3.6441889, -2.8236494, -0.5479243, 0.5526303
5: -9.2667370, -8.4896431, -9.2672348, -8.4868479, -0.4336364, 0.4286475
6: -14.7985210, -13.7872362, -14.7993088, -13.7818928, -0.5936468, 0.5864195
7: 3.0997381, 3.9353185, 3.0985899, 3.9374599, -0.7290320, 0.7275841
8: -6.7380075, -5.8335090, -6.7380075, -5.8296738, -0.5875387, 0.5768316
9: -1.3114815, -0.5982704, -1.3114789, -0.5978093, -0.4343756, 0.4335556

Time for backsubstitution: 9.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 1236

Time for candidate selection: 0.43 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3811919, upper bound: 0.3760063
time: 4.21 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3811919, upper bound: 0.3760063
time: 3.82 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -15.1794195, -13.9528790, -15.1714344, -13.9524956, -0.7429361, 0.7329220
1: -10.1084232, -9.0857506, -10.0920086, -9.0855579, -0.6094286, 0.5945289
2: -4.1960220, -3.3194592, -4.1986470, -3.3220587, -0.4759862, 0.4912789
3: -3.1571178, -1.9337506, -3.1721306, -1.9449854, -0.7200654, 0.7701542
4: -3.6404266, -2.8121598, -3.6427622, -2.8239202, -0.5494447, 0.5689471
5: -9.2700653, -8.4936180, -9.2670345, -8.4887514, -0.4488231, 0.4292543
6: -14.8080215, -13.7926197, -14.7989941, -13.7848587, -0.6165404, 0.5859165
7: 3.0867958, 3.9351797, 3.0990486, 3.9368467, -0.7421694, 0.7270446
8: -6.7483945, -5.8397064, -6.7380075, -5.8331885, -0.5998440, 0.5814974
9: -1.3120370, -0.5932131, -1.3114810, -0.5979836, -0.4380248, 0.4365529

Time for backsubstitution: 9.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 1236

Time for candidate selection: 0.52 seconds

### Candidate
type: B, layer: 3, pos: 2809

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3811919, upper bound: 0.3783553
time: 4.17 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3811919, upper bound: 0.3783549
time: 4.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 18.54 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3818124, upper bound: 0.3818123
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3818124, upper bound: 0.3818129
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3818124, upper bound: 0.3848788
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3818124, upper bound: 0.3848788
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3761616, upper bound: 0.3838539
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3761616, upper bound: 0.3838527
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3761616, upper bound: 0.3862040
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3761616, upper bound: 0.3862031
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3838532, upper bound: 0.3761623
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3838532, upper bound: 0.3761613
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3838532, upper bound: 0.3786004
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3838532, upper bound: 0.3785994
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3811919, upper bound: 0.3760063
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3811919, upper bound: 0.3760063
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3811919, upper bound: 0.3783553
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.54
Output dim: 7, lower bound: -0.3811919, upper bound: 0.3783549

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -15.1858892, -13.9737854, -15.1858892, -13.9737854, -0.7215927, 0.7215927
1: -10.1060572, -9.0976896, -10.1060572, -9.0976896, -0.5835311, 0.5835309
2: -4.2003961, -3.3283615, -4.2003961, -3.3283615, -0.4697468, 0.4697468
3: -3.1295147, -1.9299803, -3.1295147, -1.9299803, -0.7110028, 0.7110028
4: -3.6516995, -2.8452163, -3.6516995, -2.8452163, -0.5445766, 0.5445766
5: -9.2371349, -8.4703426, -9.2371349, -8.4703426, -0.4247056, 0.4247056
6: -14.7863903, -13.7813835, -14.7863903, -13.7813835, -0.5819412, 0.5819411
7: 3.0910525, 3.8776870, 3.0910525, 3.8776870, -0.6936078, 0.6936078
8: -6.7023869, -5.8123293, -6.7023869, -5.8123293, -0.5734119, 0.5734119
9: -1.3172545, -0.6013570, -1.3172545, -0.6013570, -0.4351375, 0.4351375

Time for backsubstitution: 9.52 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 1236

Time for candidate selection: 0.66 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3720917, upper bound: 0.3759694
time: 4.13 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3706694, upper bound: 0.3698899
time: 4.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -15.1858892, -13.9737854, -15.1950817, -13.9740372, -0.7210867, 0.7373686
1: -10.1060572, -9.0976896, -10.1227522, -9.0977125, -0.5832484, 0.6043591
2: -4.2003961, -3.3283615, -4.1988368, -3.3257303, -0.4849946, 0.4728281
3: -3.1295147, -1.9299803, -3.1182046, -1.9187458, -0.7510657, 0.7195628
4: -3.6516995, -2.8452163, -3.6530707, -2.8330226, -0.5622981, 0.5467489
5: -9.2371349, -8.4703426, -9.2404652, -8.4742565, -0.4294165, 0.4403323
6: -14.7863903, -13.7813835, -14.7959003, -13.7867746, -0.5868087, 0.6066644
7: 3.0910525, 3.8776870, 3.0781388, 3.8775482, -0.6934390, 0.7076049
8: -6.7023869, -5.8123293, -6.7127738, -5.8186502, -0.5766988, 0.5881031
9: -1.3172545, -0.6013570, -1.3178105, -0.5962877, -0.4381247, 0.4390664

Time for backsubstitution: 9.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 1236

Time for candidate selection: 0.56 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3720917, upper bound: 0.3759694
time: 3.99 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3706694, upper bound: 0.3698892
time: 4.42 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -15.1950817, -13.9740372, -15.1858892, -13.9737854, -0.7373686, 0.7210867
1: -10.1227522, -9.0977125, -10.1060572, -9.0976896, -0.6043589, 0.5832481
2: -4.1988368, -3.3257303, -4.2003961, -3.3283615, -0.4728283, 0.4849948
3: -3.1182046, -1.9187458, -3.1295147, -1.9299803, -0.7195628, 0.7510657
4: -3.6530707, -2.8330226, -3.6516995, -2.8452163, -0.5467489, 0.5622981
5: -9.2404652, -8.4742565, -9.2371349, -8.4703426, -0.4403323, 0.4294165
6: -14.7959003, -13.7867746, -14.7863903, -13.7813835, -0.6066643, 0.5868087
7: 3.0781388, 3.8775482, 3.0910525, 3.8776870, -0.7076044, 0.6934390
8: -6.7127738, -5.8186502, -6.7023869, -5.8123293, -0.5881031, 0.5766987
9: -1.3178105, -0.5962877, -1.3172545, -0.6013570, -0.4390665, 0.4381247

Time for backsubstitution: 8.63 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 892

Time for candidate selection: 0.55 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3713603, upper bound: 0.3786209
time: 4.14 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3698893, upper bound: 0.3722030
time: 4.26 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -15.1950817, -13.9740372, -15.1950817, -13.9740372, -0.7356682, 0.7356681
1: -10.1227522, -9.0977125, -10.1227522, -9.0977125, -0.6026790, 0.6026790
2: -4.1988368, -3.3257303, -4.1988368, -3.3257303, -0.4741808, 0.4741808
3: -3.1182046, -1.9187458, -3.1182046, -1.9187458, -0.7162349, 0.7162349
4: -3.6530707, -2.8330226, -3.6530707, -2.8330226, -0.5588608, 0.5588608
5: -9.2404652, -8.4742565, -9.2404652, -8.4742565, -0.4338115, 0.4338115
6: -14.7959003, -13.7867746, -14.7959003, -13.7867746, -0.5909905, 0.5909905
7: 3.0781388, 3.8775482, 3.0781388, 3.8775482, -0.7017179, 0.7017179
8: -6.7127738, -5.8186502, -6.7127738, -5.8186502, -0.5782483, 0.5782483
9: -1.3178105, -0.5962877, -1.3178105, -0.5962877, -0.4417195, 0.4417197

Time for backsubstitution: 8.48 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 892

Time for candidate selection: 0.51 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3713603, upper bound: 0.3786210
time: 4.08 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3698893, upper bound: 0.3722021
time: 4.04 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -15.1858892, -13.9737854, -15.1700344, -13.9526291, -0.7463508, 0.7116940
1: -10.1060572, -9.0976896, -10.0914307, -9.0857277, -0.5979333, 0.5718246
2: -4.2003961, -3.3283615, -4.1975803, -3.3220847, -0.4772865, 0.4688939
3: -3.1295147, -1.9299803, -3.1685529, -1.9449844, -0.7080369, 0.7625263
4: -3.6516995, -2.8452163, -3.6389780, -2.8243198, -0.5690980, 0.5360794
5: -9.2371349, -8.4703426, -9.2667370, -8.4896431, -0.4209663, 0.4658802
6: -14.7863903, -13.7813835, -14.7985210, -13.7872362, -0.5725818, 0.5914261
7: 3.0910525, 3.8776870, 3.0997381, 3.9353185, -0.7482877, 0.6830974
8: -6.7023869, -5.8123293, -6.7380075, -5.8335090, -0.5787010, 0.6319070
9: -1.3172545, -0.6013570, -1.3114815, -0.5982704, -0.4410533, 0.4322352

Time for backsubstitution: 9.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 1236

Time for candidate selection: 0.52 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3617973, upper bound: 0.3736268
time: 3.61 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3587664, upper bound: 0.3642738
time: 3.32 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -15.1858892, -13.9737854, -15.1794195, -13.9528790, -0.7458451, 0.7276597
1: -10.1060572, -9.0976896, -10.1084232, -9.0857506, -0.5976503, 0.5928988
2: -4.2003961, -3.3283615, -4.1960220, -3.3194592, -0.4924333, 0.4721750
3: -3.1295147, -1.9299803, -3.1571178, -1.9337506, -0.7480991, 0.7711709
4: -3.6516995, -2.8452163, -3.6404266, -2.8121598, -0.5866997, 0.5382552
5: -9.2371349, -8.4703426, -9.2700653, -8.4936180, -0.4259067, 0.4814267
6: -14.7863903, -13.7813835, -14.8080215, -13.7926197, -0.5773754, 0.6160653
7: 3.0910525, 3.8776870, 3.0867958, 3.9351797, -0.7481194, 0.6970959
8: -6.7023869, -5.8123293, -6.7483945, -5.8397064, -0.5820770, 0.6465981
9: -1.3172545, -0.6013570, -1.3120370, -0.5932131, -0.4438366, 0.4361644

Time for backsubstitution: 9.01 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 1236

Time for candidate selection: 0.44 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3617973, upper bound: 0.3736268
time: 3.49 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3587664, upper bound: 0.3642737
time: 3.52 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -15.1950817, -13.9740372, -15.1700344, -13.9526291, -0.7621267, 0.7111881
1: -10.1227522, -9.0977125, -10.0914307, -9.0857277, -0.6187613, 0.5715418
2: -4.1988368, -3.3257303, -4.1975803, -3.3220847, -0.4803678, 0.4841418
3: -3.1182046, -1.9187458, -3.1685529, -1.9449844, -0.7165966, 0.8025889
4: -3.6530707, -2.8330226, -3.6389780, -2.8243198, -0.5712702, 0.5538008
5: -9.2404652, -8.4742565, -9.2667370, -8.4896431, -0.4365927, 0.4705911
6: -14.7959003, -13.7867746, -14.7985210, -13.7872362, -0.5973051, 0.5962937
7: 3.0781388, 3.8775482, 3.0997381, 3.9353185, -0.7622848, 0.6829286
8: -6.7127738, -5.8186502, -6.7380075, -5.8335090, -0.5933919, 0.6351936
9: -1.3178105, -0.5962877, -1.3114815, -0.5982704, -0.4449822, 0.4352224

Time for backsubstitution: 9.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 892

Time for candidate selection: 0.57 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3606993, upper bound: 0.3756008
time: 3.43 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3576741, upper bound: 0.3660299
time: 3.26 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -15.1950817, -13.9740372, -15.1794195, -13.9528790, -0.7604260, 0.7259717
1: -10.1227522, -9.0977125, -10.1084232, -9.0857506, -0.6170814, 0.5912380
2: -4.1988368, -3.3257303, -4.1960220, -3.3194592, -0.4816229, 0.4734460
3: -3.1182046, -1.9187458, -3.1571178, -1.9337506, -0.7132678, 0.7677104
4: -3.6530707, -2.8330226, -3.6404266, -2.8121598, -0.5832624, 0.5503671
5: -9.2404652, -8.4742565, -9.2700653, -8.4936180, -0.4303026, 0.4749043
6: -14.7959003, -13.7867746, -14.8080215, -13.7926197, -0.5815654, 0.6003964
7: 3.0781388, 3.8775482, 3.0867958, 3.9351797, -0.7563977, 0.6913066
8: -6.7127738, -5.8186502, -6.7483945, -5.8397064, -0.5835447, 0.6367433
9: -1.3178105, -0.5962877, -1.3120370, -0.5932131, -0.4474318, 0.4388173

Time for backsubstitution: 8.98 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 892

Time for candidate selection: 0.45 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3606993, upper bound: 0.3756018
time: 4.38 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3576741, upper bound: 0.3660307
time: 3.40 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -15.1700344, -13.9526291, -15.1858892, -13.9737854, -0.7116940, 0.7463508
1: -10.0914307, -9.0857277, -10.1060572, -9.0976896, -0.5718243, 0.5979333
2: -4.1975803, -3.3220847, -4.2003961, -3.3283615, -0.4688938, 0.4772867
3: -3.1685529, -1.9449844, -3.1295147, -1.9299803, -0.7625265, 0.7080369
4: -3.6389780, -2.8243198, -3.6516995, -2.8452163, -0.5360794, 0.5690980
5: -9.2667370, -8.4896431, -9.2371349, -8.4703426, -0.4658802, 0.4209661
6: -14.7985210, -13.7872362, -14.7863903, -13.7813835, -0.5914260, 0.5725818
7: 3.0997381, 3.9353185, 3.0910525, 3.8776870, -0.6830974, 0.7482877
8: -6.7380075, -5.8335090, -6.7023869, -5.8123293, -0.6319070, 0.5787010
9: -1.3114815, -0.5982704, -1.3172545, -0.6013570, -0.4322352, 0.4410534

Time for backsubstitution: 9.29 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 1236

Time for candidate selection: 0.50 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3669146, upper bound: 0.3643992
time: 3.39 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3653399, upper bound: 0.3576740
time: 3.55 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -15.1700344, -13.9526291, -15.1950817, -13.9740372, -0.7111881, 0.7621267
1: -10.0914307, -9.0857277, -10.1227522, -9.0977125, -0.5715418, 0.6187613
2: -4.1975803, -3.3220847, -4.1988368, -3.3257303, -0.4841418, 0.4803679
3: -3.1685529, -1.9449844, -3.1182046, -1.9187458, -0.8025887, 0.7165966
4: -3.6389780, -2.8243198, -3.6530707, -2.8330226, -0.5538008, 0.5712702
5: -9.2667370, -8.4896431, -9.2404652, -8.4742565, -0.4705911, 0.4365928
6: -14.7985210, -13.7872362, -14.7959003, -13.7867746, -0.5962937, 0.5973051
7: 3.0997381, 3.9353185, 3.0781388, 3.8775482, -0.6829286, 0.7622848
8: -6.7380075, -5.8335090, -6.7127738, -5.8186502, -0.6351936, 0.5933919
9: -1.3114815, -0.5982704, -1.3178105, -0.5962877, -0.4352224, 0.4449823

Time for backsubstitution: 9.20 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 1236

Time for candidate selection: 0.57 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3669146, upper bound: 0.3644001
time: 3.61 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3653399, upper bound: 0.3576738
time: 3.82 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -15.1794195, -13.9528790, -15.1858892, -13.9737854, -0.7276595, 0.7458451
1: -10.1084232, -9.0857506, -10.1060572, -9.0976896, -0.5928991, 0.5976503
2: -4.1960220, -3.3194592, -4.2003961, -3.3283615, -0.4721751, 0.4924334
3: -3.1571178, -1.9337506, -3.1295147, -1.9299803, -0.7711711, 0.7480991
4: -3.6404266, -2.8121598, -3.6516995, -2.8452163, -0.5382552, 0.5866997
5: -9.2700653, -8.4936180, -9.2371349, -8.4703426, -0.4814267, 0.4259067
6: -14.8080215, -13.7926197, -14.7863903, -13.7813835, -0.6160654, 0.5773754
7: 3.0867958, 3.9351797, 3.0910525, 3.8776870, -0.6970959, 0.7481194
8: -6.7483945, -5.8397064, -6.7023869, -5.8123293, -0.6465983, 0.5820770
9: -1.3120370, -0.5932131, -1.3172545, -0.6013570, -0.4361645, 0.4438366

Time for backsubstitution: 9.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 892

Time for candidate selection: 0.52 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3658486, upper bound: 0.3663674
time: 3.39 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3642739, upper bound: 0.3594739
time: 3.44 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -15.1794195, -13.9528790, -15.1950817, -13.9740372, -0.7259717, 0.7604260
1: -10.1084232, -9.0857506, -10.1227522, -9.0977125, -0.5912383, 0.6170814
2: -4.1960220, -3.3194592, -4.1988368, -3.3257303, -0.4734460, 0.4816231
3: -3.1571178, -1.9337506, -3.1182046, -1.9187458, -0.7677104, 0.7132678
4: -3.6404266, -2.8121598, -3.6530707, -2.8330226, -0.5503671, 0.5832624
5: -9.2700653, -8.4936180, -9.2404652, -8.4742565, -0.4749042, 0.4303027
6: -14.8080215, -13.7926197, -14.7959003, -13.7867746, -0.6003964, 0.5815654
7: 3.0867958, 3.9351797, 3.0781388, 3.8775482, -0.6913066, 0.7563977
8: -6.7483945, -5.8397064, -6.7127738, -5.8186502, -0.6367433, 0.5835447
9: -1.3120370, -0.5932131, -1.3178105, -0.5962877, -0.4388175, 0.4474317

Time for backsubstitution: 9.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 892

Time for candidate selection: 0.50 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3658486, upper bound: 0.3663681
time: 3.56 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3642739, upper bound: 0.3594739
time: 3.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -15.1700344, -13.9526291, -15.1700344, -13.9526291, -0.7263708, 0.7263708
1: -10.0914307, -9.0857277, -10.0914307, -9.0857277, -0.5869818, 0.5869818
2: -4.1975803, -3.3220847, -4.1975803, -3.3220847, -0.4710768, 0.4710768
3: -3.1685529, -1.9449844, -3.1685529, -1.9449844, -0.7188718, 0.7188718
4: -3.6389780, -2.8243198, -3.6389780, -2.8243198, -0.5475500, 0.5475500
5: -9.2667370, -8.4896431, -9.2667370, -8.4896431, -0.4284198, 0.4284198
6: -14.7985210, -13.7872362, -14.7985210, -13.7872362, -0.5859051, 0.5859052
7: 3.0997381, 3.9353185, 3.0997381, 3.9353185, -0.7262568, 0.7262566
8: -6.7380075, -5.8335090, -6.7380075, -5.8335090, -0.5768316, 0.5768316
9: -1.3114815, -0.5982704, -1.3114815, -0.5982704, -0.4335554, 0.4335554

Time for backsubstitution: 9.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 1236

Time for candidate selection: 0.49 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3627042, upper bound: 0.3639926
time: 3.39 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3609580, upper bound: 0.3567471
time: 3.46 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -15.1700344, -13.9526291, -15.1794195, -13.9528790, -0.7258651, 0.7420638
1: -10.0914307, -9.0857277, -10.1084232, -9.0857506, -0.5866990, 0.6077166
2: -4.1975803, -3.3220847, -4.1960220, -3.3194592, -0.4863193, 0.4742526
3: -3.1685529, -1.9449844, -3.1571178, -1.9337506, -0.7589355, 0.7276652
4: -3.6389780, -2.8243198, -3.6404266, -2.8121598, -0.5652258, 0.5497222
5: -9.2667370, -8.4896431, -9.2700653, -8.4936180, -0.4331286, 0.4440295
6: -14.7985210, -13.7872362, -14.8080215, -13.7926197, -0.5907650, 0.6105647
7: 3.0997381, 3.9353185, 3.0867958, 3.9351797, -0.7260885, 0.7401309
8: -6.7380075, -5.8335090, -6.7483945, -5.8397064, -0.5801344, 0.5915232
9: -1.3114815, -0.5982704, -1.3120370, -0.5932131, -0.4365436, 0.4374844

Time for backsubstitution: 9.11 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 1236

Time for candidate selection: 0.50 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3627042, upper bound: 0.3639927
time: 3.87 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3609580, upper bound: 0.3567469
time: 3.85 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -15.1794195, -13.9528790, -15.1700344, -13.9526291, -0.7420638, 0.7258650
1: -10.1084232, -9.0857506, -10.0914307, -9.0857277, -0.6077166, 0.5866990
2: -4.1960220, -3.3194592, -4.1975803, -3.3220847, -0.4742526, 0.4863192
3: -3.1571178, -1.9337506, -3.1685529, -1.9449844, -0.7276652, 0.7589355
4: -3.6404266, -2.8121598, -3.6389780, -2.8243198, -0.5497222, 0.5652258
5: -9.2700653, -8.4936180, -9.2667370, -8.4896431, -0.4440296, 0.4331286
6: -14.8080215, -13.7926197, -14.7985210, -13.7872362, -0.6105647, 0.5907649
7: 3.0867958, 3.9351797, 3.0997381, 3.9353185, -0.7401309, 0.7260883
8: -6.7483945, -5.8397064, -6.7380075, -5.8335090, -0.5915232, 0.5801344
9: -1.3120370, -0.5932131, -1.3114815, -0.5982704, -0.4374845, 0.4365436

Time for backsubstitution: 9.16 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 892

Time for candidate selection: 0.50 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3616421, upper bound: 0.3659385
time: 3.32 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3598933, upper bound: 0.3585382
time: 3.42 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -15.1794195, -13.9528790, -15.1794195, -13.9528790, -0.7403762, 0.7403761
1: -10.1084232, -9.0857506, -10.1084232, -9.0857506, -0.6060553, 0.6060553
2: -4.1960220, -3.3194592, -4.1960220, -3.3194592, -0.4755223, 0.4755222
3: -3.1571178, -1.9337506, -3.1571178, -1.9337506, -0.7242041, 0.7242041
4: -3.6404266, -2.8121598, -3.6404266, -2.8121598, -0.5617890, 0.5617890
5: -9.2700653, -8.4936180, -9.2700653, -8.4936180, -0.4375064, 0.4375064
6: -14.8080215, -13.7926197, -14.8080215, -13.7926197, -0.5948962, 0.5948963
7: 3.0867958, 3.9351797, 3.0867958, 3.9351797, -0.7343407, 0.7343407
8: -6.7483945, -5.8397064, -6.7483945, -5.8397064, -0.5816026, 0.5816026
9: -1.3120370, -0.5932131, -1.3120370, -0.5932131, -0.4401383, 0.4401383

Time for backsubstitution: 9.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 570
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1493
type: A, layer: 3, pos: 2082
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 2586
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 1859
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 892

Time for candidate selection: 0.49 seconds

### Candidate
type: A, layer: 3, pos: 1992

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3616421, upper bound: 0.3659386
time: 3.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3598933, upper bound: 0.3585382
time: 3.67 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 16.98 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3720917, upper bound: 0.3759694
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3706694, upper bound: 0.3698899
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3720917, upper bound: 0.3759694
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3706694, upper bound: 0.3698892
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3713603, upper bound: 0.3786209
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3698893, upper bound: 0.3722030
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3713603, upper bound: 0.3786210
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3698893, upper bound: 0.3722021
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3617973, upper bound: 0.3736268
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3587664, upper bound: 0.3642738
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3617973, upper bound: 0.3736268
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3587664, upper bound: 0.3642737
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3606993, upper bound: 0.3756008
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3576741, upper bound: 0.3660299
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3606993, upper bound: 0.3756018
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3576741, upper bound: 0.3660307
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3669146, upper bound: 0.3643992
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3653399, upper bound: 0.3576740
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3669146, upper bound: 0.3644001
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3653399, upper bound: 0.3576738
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3658486, upper bound: 0.3663674
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3642739, upper bound: 0.3594739
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3658486, upper bound: 0.3663681
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3642739, upper bound: 0.3594739
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3627042, upper bound: 0.3639926
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3609580, upper bound: 0.3567471
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3627042, upper bound: 0.3639927
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3609580, upper bound: 0.3567469
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3616421, upper bound: 0.3659385
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3598933, upper bound: 0.3585382
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3616421, upper bound: 0.3659386
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 7, lower bound: -0.3598933, upper bound: 0.3585382

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -15.1857290, -13.9784737, -15.1858892, -13.9737854, -0.7215481, 0.7164488
1: -10.1051130, -9.0976896, -10.1060572, -9.0976896, -0.5824902, 0.5830951
2: -4.1999254, -3.3284073, -4.2003961, -3.3283615, -0.4694514, 0.4697169
3: -3.1284354, -1.9299777, -3.1295147, -1.9299803, -0.7098143, 0.7107778
4: -3.6505380, -2.8452175, -3.6516995, -2.8452163, -0.5431802, 0.5442722
5: -9.2371349, -8.4731712, -9.2371349, -8.4703426, -0.4245116, 0.4218490
6: -14.7863903, -13.7825613, -14.7863903, -13.7813835, -0.5817677, 0.5811162
7: 3.0935917, 3.8776870, 3.0910525, 3.8776870, -0.6920662, 0.6936078
8: -6.7023869, -5.8140078, -6.7023869, -5.8123293, -0.5734119, 0.5711008
9: -1.3172534, -0.6021686, -1.3172545, -0.6013570, -0.4349911, 0.4344108

Time for backsubstitution: 9.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 570
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 1493
type: B, layer: 3, pos: 2082
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 1243
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2867
type: B, layer: 3, pos: 2586
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 1859
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 1236

Time for candidate selection: 0.50 seconds

### Candidate
type: B, layer: 3, pos: 1992

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3707523, upper bound: 0.3707518
time: 3.99 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3707523, upper bound: 0.3707529
time: 4.15 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.31 + 549.08 = 607.39 seconds
