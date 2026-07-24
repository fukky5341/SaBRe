## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.15455141599999997


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6983652, 0.6983652)
1: (-10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5463943, 0.5463943)
2: (-8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4942775, 0.4942775)
3: (-8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4032021, 0.4032018)
4: (-3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3239938, 0.3239939)
5: (-8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4194160, 0.4194160)
6: (-13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4799125, 0.4799125)
7: (-3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4211464, 0.4211464)
8: (-0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5180578, 0.5180578)
9: (3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3047249, 0.3047249)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.45 + 34.65 = 57.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.1644164, upper bound: 0.1644164

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 538

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 538

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1625574, upper bound: 0.1644083
time: 3.45 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1644138, upper bound: 0.1644147
time: 3.44 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.05 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.05
Output dim: 9, lower bound: -0.1625574, upper bound: 0.1644083
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.05
Output dim: 9, lower bound: -0.1644138, upper bound: 0.1644147

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -12.1968737, -11.0496187, -12.1975031, -11.0425797, -0.6937537, 0.6872334
1: -10.2288456, -9.2337561, -10.2290459, -9.2314425, -0.5450673, 0.5428157
2: -8.6909456, -7.9456201, -8.6912594, -7.9438868, -0.4927468, 0.4914379
3: -8.3035488, -7.6060758, -8.3062801, -7.6055346, -0.3987594, 0.4009781
4: -3.5023153, -2.9025960, -3.5037756, -2.9025707, -0.3216826, 0.3231506
5: -8.5407772, -7.7252212, -8.5416403, -7.7246981, -0.4178367, 0.4181695
6: -13.7398052, -12.8425827, -13.7404699, -12.8359632, -0.4758518, 0.4695399
7: -3.5742784, -2.9655659, -3.5756140, -2.9654758, -0.4181237, 0.4195485
8: -0.4776235, 0.2371969, -0.4779358, 0.2433228, -0.5129828, 0.5069695
9: 3.4910545, 4.0823555, 3.4887967, 4.0824003, -0.3011358, 0.3033664

Time for backsubstitution: 21.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 538

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 538

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1625577, upper bound: 0.1625573
time: 3.49 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1625577, upper bound: 0.1644083
time: 3.78 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -12.2235670, -11.0381880, -12.1978388, -11.0387983, -0.7063498, 0.6952124
1: -10.2386217, -9.2301559, -10.2291517, -9.2301998, -0.5564127, 0.5451107
2: -8.6976194, -7.9428301, -8.6914234, -7.9429541, -0.5004220, 0.4934735
3: -8.3077869, -7.5957918, -8.3077450, -7.6052465, -0.4019578, 0.4076977
4: -3.5053174, -2.8989418, -3.5045638, -2.9025569, -0.3241903, 0.3267231
5: -8.5421095, -7.7198868, -8.5421009, -7.7244182, -0.4194393, 0.4240415
6: -13.7640152, -12.8319912, -13.7408247, -12.8324060, -0.4851284, 0.4769876
7: -3.5778308, -2.9599838, -3.5763359, -2.9654269, -0.4235435, 0.4248335
8: -0.5014777, 0.2490604, -0.4781017, 0.2466230, -0.5217519, 0.5203300
9: 3.4870310, 4.0906291, 3.4875841, 4.0824246, -0.3042520, 0.3092138

Time for backsubstitution: 20.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 538

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 538

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1644086, upper bound: 0.1625573
time: 3.56 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1644086, upper bound: 0.1625573
time: 3.28 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.48 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.48
Output dim: 9, lower bound: -0.1625577, upper bound: 0.1625573
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.48
Output dim: 9, lower bound: -0.1625577, upper bound: 0.1644083
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.48
Output dim: 9, lower bound: -0.1644086, upper bound: 0.1625573
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.48
Output dim: 9, lower bound: -0.1644086, upper bound: 0.1625573

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -12.1968737, -11.0496187, -12.1968737, -11.0496187, -0.6867023, 0.6867023
1: -10.2288456, -9.2337561, -10.2288456, -9.2337561, -0.5427608, 0.5427608
2: -8.6909456, -7.9456201, -8.6909456, -7.9456201, -0.4910421, 0.4910421
3: -8.3035488, -7.6060758, -8.3035488, -7.6060758, -0.3982623, 0.3982623
4: -3.5023153, -2.9025960, -3.5023153, -2.9025960, -0.3216580, 0.3216577
5: -8.5407772, -7.7252212, -8.5407772, -7.7252212, -0.4173210, 0.4173210
6: -13.7398052, -12.8425827, -13.7398052, -12.8425827, -0.4692199, 0.4692199
7: -3.5742784, -2.9655659, -3.5742784, -2.9655659, -0.4177237, 0.4177237
8: -0.4776235, 0.2371969, -0.4776235, 0.2371969, -0.5060854, 0.5060854
9: 3.4910545, 4.0823555, 3.4910545, 4.0823555, -0.3010600, 0.3010600

Time for backsubstitution: 21.24 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 1942
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 1920
type: A, layer: 3, pos: 3110
type: A, layer: 3, pos: 724
type: A, layer: 3, pos: 704
type: A, layer: 3, pos: 2607
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 2817
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 414

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 1689

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1504929, upper bound: 0.1590642
time: 3.76 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1590638, upper bound: 0.1590642
time: 3.71 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -12.1968737, -11.0496187, -12.2235670, -11.0381880, -0.6981583, 0.6954975
1: -10.2288456, -9.2337561, -10.2386217, -9.2301559, -0.5463643, 0.5528595
2: -8.6909456, -7.9456201, -8.6976194, -7.9428301, -0.4939179, 0.4978004
3: -8.3035488, -7.6060758, -8.3077869, -7.5957918, -0.4035141, 0.4025381
4: -3.5023153, -2.9025960, -3.5053174, -2.8989418, -0.3244424, 0.3247654
5: -8.5407772, -7.7252212, -8.5421095, -7.7198868, -0.4227376, 0.4186709
6: -13.7398052, -12.8425827, -13.7640152, -12.8319912, -0.4783454, 0.4749293
7: -3.5742784, -2.9655659, -3.5778308, -2.9599838, -0.4220231, 0.4212685
8: -0.4776235, 0.2371969, -0.5014777, 0.2490604, -0.5179896, 0.5111358
9: 3.4910545, 4.0823555, 3.4870310, 4.0906291, -0.3056636, 0.3051131

Time for backsubstitution: 21.36 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 1942
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 1920
type: A, layer: 3, pos: 3110
type: A, layer: 3, pos: 724
type: A, layer: 3, pos: 704
type: A, layer: 3, pos: 2607
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 2817
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 414

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 1689

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1504929, upper bound: 0.1609167
time: 3.89 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1590638, upper bound: 0.1609167
time: 3.60 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -12.2235670, -11.0381880, -12.1968737, -11.0496187, -0.6954975, 0.6981587
1: -10.2386217, -9.2301559, -10.2288456, -9.2337561, -0.5528595, 0.5463643
2: -8.6976194, -7.9428301, -8.6909456, -7.9456201, -0.4978004, 0.4939179
3: -8.3077869, -7.5957918, -8.3035488, -7.6060758, -0.4025381, 0.4035141
4: -3.5053174, -2.8989418, -3.5023153, -2.9025960, -0.3247656, 0.3244424
5: -8.5421095, -7.7198868, -8.5407772, -7.7252212, -0.4186709, 0.4227376
6: -13.7640152, -12.8319912, -13.7398052, -12.8425827, -0.4749293, 0.4783454
7: -3.5778308, -2.9599838, -3.5742784, -2.9655659, -0.4212685, 0.4220231
8: -0.5014777, 0.2490604, -0.4776235, 0.2371969, -0.5111356, 0.5179901
9: 3.4870310, 4.0906291, 3.4910545, 4.0823555, -0.3051131, 0.3056636

Time for backsubstitution: 21.70 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 1942
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 1920
type: A, layer: 3, pos: 724
type: A, layer: 3, pos: 3110
type: A, layer: 3, pos: 704
type: A, layer: 3, pos: 2607
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 2817
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 414

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 1689

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1523429, upper bound: 0.1590639
time: 4.46 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1609158, upper bound: 0.1590642
time: 3.99 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -12.2235670, -11.0381880, -12.2235670, -11.0381880, -0.7004147, 0.7004147
1: -10.2386217, -9.2301559, -10.2386217, -9.2301559, -0.5477142, 0.5477142
2: -8.6976194, -7.9428301, -8.6976194, -7.9428301, -0.4941525, 0.4941525
3: -8.3077869, -7.5957918, -8.3077869, -7.5957918, -0.4040492, 0.4040492
4: -3.5053174, -2.8989418, -3.5053174, -2.8989418, -0.3244145, 0.3244143
5: -8.5421095, -7.7198868, -8.5421095, -7.7198868, -0.4239495, 0.4239495
6: -13.7640152, -12.8319912, -13.7640152, -12.8319912, -0.4825308, 0.4822235
7: -3.5778308, -2.9599838, -3.5778308, -2.9599838, -0.4235940, 0.4235940
8: -0.5014777, 0.2490604, -0.5014777, 0.2490604, -0.5207200, 0.5207200
9: 3.4870310, 4.0906291, 3.4870310, 4.0906291, -0.3043411, 0.3043408

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 1942
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 1920
type: A, layer: 3, pos: 724
type: A, layer: 3, pos: 3110
type: A, layer: 3, pos: 704
type: A, layer: 3, pos: 2607
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 2817
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 414

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 1689

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1523437, upper bound: 0.1590642
time: 4.05 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1609166, upper bound: 0.1590809
time: 5.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.53 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.53
Output dim: 9, lower bound: -0.1504929, upper bound: 0.1590642
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.53
Output dim: 9, lower bound: -0.1590638, upper bound: 0.1590642
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.53
Output dim: 9, lower bound: -0.1504929, upper bound: 0.1609167
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.53
Output dim: 9, lower bound: -0.1590638, upper bound: 0.1609167
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.53
Output dim: 9, lower bound: -0.1523429, upper bound: 0.1590639
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.53
Output dim: 9, lower bound: -0.1609158, upper bound: 0.1590642
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.53
Output dim: 9, lower bound: -0.1523437, upper bound: 0.1590642
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.53
Output dim: 9, lower bound: -0.1609166, upper bound: 0.1590809

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -12.1957874, -11.0506153, -12.1965408, -11.0499287, -0.6805220, 0.6824827
1: -10.2288389, -9.2483110, -10.2288399, -9.2385464, -0.5381737, 0.5315461
2: -8.6909332, -7.9624977, -8.6909428, -7.9511108, -0.4850540, 0.4734197
3: -8.3035393, -7.6363668, -8.3035460, -7.6158800, -0.3890862, 0.3688312
4: -3.4876485, -2.9026337, -3.4977517, -2.9026079, -0.3061118, 0.3164568
5: -8.5374537, -7.7256470, -8.5397129, -7.7253513, -0.4145427, 0.4150918
6: -13.7388639, -12.8425817, -13.7395172, -12.8425808, -0.4655838, 0.4676924
7: -3.5742383, -2.9763691, -3.5742664, -2.9689274, -0.4145088, 0.4079185
8: -0.4648695, 0.2371898, -0.4736547, 0.2371941, -0.4926195, 0.5018907
9: 3.5085974, 4.0823011, 3.4967270, 4.0823393, -0.2871559, 0.2953634

Time for backsubstitution: 21.62 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1494
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 1942
type: B, layer: 3, pos: 1690
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2572
type: B, layer: 3, pos: 1920
type: B, layer: 3, pos: 3110
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 704
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 1843
type: B, layer: 3, pos: 1731
type: B, layer: 3, pos: 2817
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 414

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 1689

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1504929, upper bound: 0.1504925
time: 3.78 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1504929, upper bound: 0.1590642
time: 3.63 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12.1886177, -11.0851631, -12.1966896, -11.0666838, -0.6806836, 0.6867342
1: -10.2580509, -9.2516499, -10.2288418, -9.2420416, -0.5631766, 0.5353723
2: -8.7250900, -7.9649119, -8.6909285, -7.9548922, -0.5333514, 0.4792156
3: -8.3733511, -7.6143332, -8.3035364, -7.6101465, -0.4667363, 0.3829896
4: -3.4931655, -2.8693466, -3.4976716, -2.9026062, -0.3123589, 0.3610613
5: -8.5366611, -7.7179594, -8.5388718, -7.7252932, -0.4210641, 0.4154401
6: -13.7433090, -12.8419437, -13.7396374, -12.8425798, -0.4646835, 0.4781055
7: -3.5976233, -2.9718523, -3.5742691, -2.9689507, -0.4357486, 0.4140902
8: -0.4692717, 0.2644713, -0.4733086, 0.2371898, -0.4976616, 0.5332212
9: 3.5047998, 4.1212940, 3.4977674, 4.0823402, -0.2961714, 0.3427954

Time for backsubstitution: 21.48 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1494
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 1942
type: B, layer: 3, pos: 1690
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2572
type: B, layer: 3, pos: 1920
type: B, layer: 3, pos: 3110
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 704
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 1843
type: B, layer: 3, pos: 1731
type: B, layer: 3, pos: 2817
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 414

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 1739

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1501157, upper bound: 0.1549748
time: 5.22 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1518196, upper bound: 0.1518199
time: 3.76 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -12.1957874, -11.0506153, -12.2232323, -11.0384979, -0.6919775, 0.6885557
1: -10.2288389, -9.2483110, -10.2386217, -9.2349510, -0.5417767, 0.5408008
2: -8.6909332, -7.9624977, -8.6976128, -7.9483166, -0.4879289, 0.4801779
3: -8.3035393, -7.6363668, -8.3077831, -7.6055956, -0.3943315, 0.3731072
4: -3.4876485, -2.9026337, -3.5007544, -2.8989527, -0.3088139, 0.3195651
5: -8.5374537, -7.7256470, -8.5410490, -7.7200170, -0.4199572, 0.4164419
6: -13.7388639, -12.8425817, -13.7637177, -12.8319931, -0.4746094, 0.4733720
7: -3.5742383, -2.9763691, -3.5778193, -2.9633453, -0.4188406, 0.4114633
8: -0.4648695, 0.2371898, -0.4975088, 0.2490573, -0.5045242, 0.5069404
9: 3.5085974, 4.0823011, 3.4927006, 4.0906115, -0.2919198, 0.2994146

Time for backsubstitution: 21.52 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1494
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 1942
type: B, layer: 3, pos: 1690
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2572
type: B, layer: 3, pos: 1920
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 3110
type: B, layer: 3, pos: 704
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 1843
type: B, layer: 3, pos: 1731
type: B, layer: 3, pos: 2817
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 414

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 1689

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1504929, upper bound: 0.1523429
time: 3.44 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1504929, upper bound: 0.1609159
time: 3.81 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12.1886177, -11.0851631, -12.2233839, -11.0552521, -0.6921415, 0.6925745
1: -10.2580509, -9.2516499, -10.2386217, -9.2384453, -0.5667810, 0.5444119
2: -8.7250900, -7.9649119, -8.6975994, -7.9520969, -0.5362272, 0.4859753
3: -8.3733511, -7.6143332, -8.3077726, -7.5998626, -0.4716096, 0.3872657
4: -3.4931655, -2.8693466, -3.5006742, -2.8989530, -0.3150817, 0.3641758
5: -8.5366611, -7.7179594, -8.5402060, -7.7199602, -0.4264798, 0.4167912
6: -13.7433090, -12.8419437, -13.7638416, -12.8319931, -0.4737291, 0.4839382
7: -3.5976233, -2.9718523, -3.5778203, -2.9633694, -0.4396124, 0.4176335
8: -0.4692717, 0.2644713, -0.4971631, 0.2490525, -0.5095663, 0.5375552
9: 3.5047998, 4.1212940, 3.4937439, 4.0906143, -0.3010108, 0.3468482

Time for backsubstitution: 21.54 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1494
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 1942
type: B, layer: 3, pos: 1690
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2572
type: B, layer: 3, pos: 1920
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 3110
type: B, layer: 3, pos: 704
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 1843
type: B, layer: 3, pos: 1731
type: B, layer: 3, pos: 2817
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 414

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 1739

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1501157, upper bound: 0.1568296
time: 5.17 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1518198, upper bound: 0.1536717
time: 5.20 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12.2224712, -11.0391827, -12.1965408, -11.0499287, -0.6892471, 0.6920595
1: -10.2386198, -9.2447147, -10.2288399, -9.2385464, -0.5481598, 0.5351496
2: -8.6976032, -7.9597006, -8.6909428, -7.9511108, -0.4918063, 0.4762936
3: -8.3077774, -7.6260858, -8.3035460, -7.6158800, -0.3933625, 0.3743293
4: -3.4906495, -2.8989804, -3.4977517, -2.9026079, -0.3092191, 0.3192302
5: -8.5387878, -7.7203116, -8.5397129, -7.7253513, -0.4158933, 0.4205031
6: -13.7630463, -12.8319931, -13.7395172, -12.8425808, -0.4712353, 0.4767747
7: -3.5777929, -2.9707870, -3.5742664, -2.9689274, -0.4180555, 0.4123497
8: -0.4887209, 0.2490532, -0.4736547, 0.2371941, -0.4976599, 0.5137959
9: 3.5045724, 4.0905743, 3.4967270, 4.0823393, -0.2912111, 0.3001404

Time for backsubstitution: 21.57 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1494
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 1942
type: B, layer: 3, pos: 1690
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2572
type: B, layer: 3, pos: 1920
type: B, layer: 3, pos: 3110
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 704
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 1843
type: B, layer: 3, pos: 1731
type: B, layer: 3, pos: 2817
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 414

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 1689

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1523429, upper bound: 0.1504926
time: 3.94 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1523429, upper bound: 0.1590640
time: 4.37 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12.2152405, -11.0737333, -12.1966896, -11.0666838, -0.6913509, 0.6960778
1: -10.2678280, -9.2480555, -10.2288418, -9.2420416, -0.5732198, 0.5389762
2: -8.7317638, -7.9621158, -8.6909285, -7.9548922, -0.5387502, 0.4820919
3: -8.3775864, -7.6040654, -8.3035364, -7.6101465, -0.4710097, 0.3884556
4: -3.4961686, -2.8656979, -3.4976716, -2.9026062, -0.3154659, 0.3635969
5: -8.5379925, -7.7126427, -8.5388718, -7.7252932, -0.4224148, 0.4208257
6: -13.7673702, -12.8313560, -13.7396374, -12.8425798, -0.4704633, 0.4873435
7: -3.6011841, -2.9662700, -3.5742691, -2.9689507, -0.4393010, 0.4185634
8: -0.4931180, 0.2763331, -0.4733086, 0.2371898, -0.5028877, 0.5448046
9: 3.5007811, 4.1295614, 3.4977674, 4.0823402, -0.3002269, 0.3474302

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1494
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 1942
type: B, layer: 3, pos: 1690
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2572
type: B, layer: 3, pos: 1920
type: B, layer: 3, pos: 3110
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 704
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 1843
type: B, layer: 3, pos: 1731
type: B, layer: 3, pos: 2817
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 414

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 1739

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1519724, upper bound: 0.1549750
time: 3.85 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1518199, upper bound: 0.1518199
time: 3.83 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12.2224712, -11.0391827, -12.2232323, -11.0384979, -0.6942225, 0.6961923
1: -10.2386198, -9.2447147, -10.2386217, -9.2349510, -0.5431261, 0.5364985
2: -8.6976032, -7.9597006, -8.6976128, -7.9483166, -0.4881625, 0.4765286
3: -8.3077774, -7.6260858, -8.3077831, -7.6055956, -0.3948722, 0.3746121
4: -3.4906495, -2.8989804, -3.5007544, -2.8989527, -0.3088676, 0.3192129
5: -8.5387878, -7.7203116, -8.5410490, -7.7200170, -0.4211698, 0.4217145
6: -13.7630463, -12.8319931, -13.7637177, -12.8319931, -0.4788826, 0.4806662
7: -3.5777929, -2.9707870, -3.5778193, -2.9633453, -0.4203782, 0.4137869
8: -0.4887209, 0.2490532, -0.4975088, 0.2490573, -0.5072575, 0.5165267
9: 3.5045724, 4.0905743, 3.4927006, 4.0906115, -0.2904403, 0.2986448

Time for backsubstitution: 21.57 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1494
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 1942
type: B, layer: 3, pos: 1690
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2572
type: B, layer: 3, pos: 1920
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 3110
type: B, layer: 3, pos: 704
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 1843
type: B, layer: 3, pos: 1731
type: B, layer: 3, pos: 2817
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 414

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 1689

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1523499, upper bound: 0.1505082
time: 3.70 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1523499, upper bound: 0.1590810
time: 3.43 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -12.2152405, -11.0737333, -12.2233839, -11.0552521, -0.6943035, 0.7002945
1: -10.2678280, -9.2480555, -10.2386217, -9.2384453, -0.5681324, 0.5403261
2: -8.7317638, -7.9621158, -8.6975994, -7.9520969, -0.5364666, 0.4823270
3: -8.3775864, -7.6040654, -8.3077726, -7.5998626, -0.4725211, 0.3887579
4: -3.4961686, -2.8656979, -3.5006742, -2.8989530, -0.3151145, 0.3638172
5: -8.5379925, -7.7126427, -8.5402060, -7.7199602, -0.4276922, 0.4220381
6: -13.7673702, -12.8313560, -13.7638416, -12.8319931, -0.4779055, 0.4912329
7: -3.6011841, -2.9662700, -3.5778203, -2.9633694, -0.4416142, 0.4199581
8: -0.4931180, 0.2763331, -0.4971631, 0.2490525, -0.5123086, 0.5474844
9: 3.5007811, 4.1295614, 3.4937439, 4.0906143, -0.2994554, 0.3460820

Time for backsubstitution: 21.54 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1494
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 1942
type: B, layer: 3, pos: 1690
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2572
type: B, layer: 3, pos: 1920
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 3110
type: B, layer: 3, pos: 704
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 1843
type: B, layer: 3, pos: 1731
type: B, layer: 3, pos: 2817
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 414

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 1739

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1519807, upper bound: 0.1549956
time: 4.11 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1536798, upper bound: 0.1518385
time: 3.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.69 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1504929, upper bound: 0.1504925
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1504929, upper bound: 0.1590642
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1501157, upper bound: 0.1549748
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1518196, upper bound: 0.1518199
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1504929, upper bound: 0.1523429
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1504929, upper bound: 0.1609159
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1501157, upper bound: 0.1568296
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1518198, upper bound: 0.1536717
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1523429, upper bound: 0.1504926
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1523429, upper bound: 0.1590640
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1519724, upper bound: 0.1549750
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1518199, upper bound: 0.1518199
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1523499, upper bound: 0.1505082
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1523499, upper bound: 0.1590810
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1519807, upper bound: 0.1549956
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 29.69
Output dim: 9, lower bound: -0.1536798, upper bound: 0.1518385

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -12.1957874, -11.0506153, -12.1886177, -11.0851631, -0.6603127, 0.6899652
1: -10.2288389, -9.2483110, -10.2580509, -9.2516499, -0.5243025, 0.5606484
2: -8.6909332, -7.9624977, -8.7250900, -7.9649119, -0.4835424, 0.5203385
3: -8.3035393, -7.6363668, -8.3733511, -7.6143332, -0.3911419, 0.4420905
4: -3.4876485, -2.9026337, -3.4931655, -2.8693466, -0.3477759, 0.3182430
5: -8.5374537, -7.7256470, -8.5366611, -7.7179594, -0.4161122, 0.4100842
6: -13.7388639, -12.8425817, -13.7433090, -12.8419437, -0.4671693, 0.4678526
7: -3.5742383, -2.9763691, -3.5976233, -2.9718523, -0.4096904, 0.4299583
8: -0.4648695, 0.2371898, -0.4692717, 0.2644713, -0.5230899, 0.5007234
9: 3.5085974, 4.0823011, 3.5047998, 4.1212940, -0.3337114, 0.2937522

Time for backsubstitution: 21.26 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 1942
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 1920
type: A, layer: 3, pos: 3110
type: A, layer: 3, pos: 724
type: A, layer: 3, pos: 704
type: A, layer: 3, pos: 2607
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 2817
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 414

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 1739

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1464543, upper bound: 0.1501159
time: 3.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1431562, upper bound: 0.1518200
time: 3.75 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.1885118, -11.0854540, -12.1960182, -11.0520487, -0.6901693, 0.6855760
1: -10.2579117, -9.2516594, -10.2277527, -9.2338591, -0.5715847, 0.5345345
2: -8.7250214, -7.9651155, -8.6903706, -7.9472485, -0.5360665, 0.4778385
3: -8.3733492, -7.6145096, -8.3035488, -7.6074667, -0.4702382, 0.3828218
4: -3.4926438, -2.8693461, -3.4981649, -2.9025970, -0.3115139, 0.3567519
5: -8.5361900, -7.7180390, -8.5370312, -7.7258935, -0.4194045, 0.4141014
6: -13.7428169, -12.8419466, -13.7359200, -12.8425913, -0.4678931, 0.4764767
7: -3.5976219, -2.9732265, -3.5742569, -2.9753160, -0.4279819, 0.4126654
8: -0.4691675, 0.2644546, -0.4768124, 0.2370601, -0.4974012, 0.5351758
9: 3.5053549, 4.1212950, 3.4952126, 4.0823555, -0.2953033, 0.3408983

Time for backsubstitution: 21.50 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 1942
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 1920
type: A, layer: 3, pos: 3110
type: A, layer: 3, pos: 724
type: A, layer: 3, pos: 704
type: A, layer: 3, pos: 2607
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 2817
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 414

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 1689

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1413231, upper bound: 0.1549750
time: 3.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1501157, upper bound: 0.1549748
time: 5.08 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -12.1957874, -11.0506153, -12.2152405, -11.0737333, -0.6717691, 0.6979508
1: -10.2288389, -9.2483110, -10.2678280, -9.2480555, -0.5279078, 0.5699139
2: -8.6909332, -7.9624977, -8.7317638, -7.9621158, -0.4864187, 0.5263906
3: -8.3035393, -7.6363668, -8.3775864, -7.6040654, -0.3966637, 0.4463630
4: -3.4876485, -2.9026337, -3.4961686, -2.8656979, -0.3501523, 0.3213573
5: -8.5374537, -7.7256470, -8.5379925, -7.7126427, -0.4214976, 0.4114342
6: -13.7388639, -12.8425817, -13.7673702, -12.8313560, -0.4762025, 0.4734583
7: -3.5742383, -2.9763691, -3.6011841, -2.9662700, -0.4140017, 0.4335113
8: -0.4648695, 0.2371898, -0.4931180, 0.2763331, -0.5346494, 0.5058112
9: 3.5085974, 4.0823011, 3.5007811, 4.1295614, -0.3384295, 0.2977984

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 1942
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 1920
type: A, layer: 3, pos: 3110
type: A, layer: 3, pos: 724
type: A, layer: 3, pos: 704
type: A, layer: 3, pos: 2607
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 2817
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 414

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 1739

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1464544, upper bound: 0.1519725
time: 4.07 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1431565, upper bound: 0.1536720
time: 3.93 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -12.1885118, -11.0854540, -12.2227440, -11.0406199, -0.7016268, 0.6914315
1: -10.2579117, -9.2516594, -10.2375317, -9.2302656, -0.5751858, 0.5435846
2: -8.7250214, -7.9651155, -8.6970634, -7.9444532, -0.5389428, 0.4846239
3: -8.3733492, -7.6145096, -8.3077850, -7.5971537, -0.4750650, 0.3870983
4: -3.4926438, -2.8693461, -3.5011473, -2.8989410, -0.3142511, 0.3598580
5: -8.5361900, -7.7180390, -8.5383663, -7.7205343, -0.4248590, 0.4154522
6: -13.7428169, -12.8419466, -13.7601414, -12.8320055, -0.4767809, 0.4821641
7: -3.5976219, -2.9732265, -3.5778098, -2.9697261, -0.4318306, 0.4162092
8: -0.4691675, 0.2644546, -0.5006936, 0.2489219, -0.5093060, 0.5394907
9: 3.5053549, 4.1212950, 3.4911880, 4.0906277, -0.3000646, 0.3449545

Time for backsubstitution: 20.77 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 1739
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 1942
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 1920
type: A, layer: 3, pos: 3110
type: A, layer: 3, pos: 724
type: A, layer: 3, pos: 704
type: A, layer: 3, pos: 2607
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 2817
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 414

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 1689

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1413233, upper bound: 0.1568298
time: 3.85 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1501157, upper bound: 0.1568296
time: 5.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -12.2224712, -11.0391827, -12.1886177, -11.0851631, -0.6690907, 0.7013631
1: -10.2386198, -9.2447147, -10.2580509, -9.2516499, -0.5342679, 0.5642519
2: -8.6976032, -7.9597006, -8.7250900, -7.9649119, -0.4901521, 0.5232124
3: -8.3077774, -7.6260858, -8.3733511, -7.6143332, -0.3954177, 0.4471655
4: -3.4906495, -2.8989804, -3.4931655, -2.8693466, -0.3508830, 0.3210559
5: -8.5387878, -7.7203116, -8.5366611, -7.7179594, -0.4174628, 0.4154956
6: -13.7630463, -12.8319931, -13.7433090, -12.8419437, -0.4728289, 0.4767232
7: -3.5777929, -2.9707870, -3.5976233, -2.9718523, -0.4132371, 0.4339480
8: -0.4887209, 0.2490532, -0.4692717, 0.2644713, -0.5273995, 0.5126290
9: 3.5045724, 4.0905743, 3.5047998, 4.1212940, -0.3377666, 0.2984095

Time for backsubstitution: 20.66 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.11 + 563.23 = 620.34 seconds
