## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.872541919


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8442516, 1.8442519)
1: (-17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7918735, 2.7918735)
2: (-3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3578615, 2.3578615)
3: (-10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.6013856, 2.6013861)
4: (-12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7610822, 2.7610822)
5: (-4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0475702, 2.0475702)
6: (-3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2829180, 2.2829187)
7: (-9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1810565, 3.1810570)
8: (-2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0486369, 2.0486372)
9: (-4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.3078995, 2.3078992)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.90 + 39.02 = 61.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.8742898, upper bound: 0.8742899

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 471

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 933

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8736063, upper bound: 0.8738016
time: 4.98 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8738034, upper bound: 0.8736069
time: 5.20 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.20 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.20
Output dim: 0, lower bound: -0.8736063, upper bound: 0.8738016
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.20
Output dim: 0, lower bound: -0.8738034, upper bound: 0.8736069

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8442469, 1.8450017
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7918649, 2.7933936
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3528061, 2.3542399
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5993819, 2.5985889
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7604313, 2.7606153
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0449162, 2.0438664
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2809801, 2.2802150
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1825533, 3.1810508
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0483165, 2.0481899
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.3052258, 2.3059838

Time for backsubstitution: 21.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 471

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6219

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8736056, upper bound: 0.8738008
time: 4.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8736056, upper bound: 0.8738008
time: 4.90 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8450017, 1.8442471
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7933936, 2.7918649
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3542399, 2.3528063
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5985885, 2.5993824
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7606153, 2.7604311
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0438662, 2.0449159
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2802153, 2.2809808
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1810503, 3.1825528
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0481901, 2.0483165
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.3059840, 2.3052258

Time for backsubstitution: 21.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8693261, upper bound: 0.8691493
time: 8.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8693261, upper bound: 0.8691493
time: 8.92 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 39.36 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 39.36
Output dim: 0, lower bound: -0.8736056, upper bound: 0.8738008
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 39.36
Output dim: 0, lower bound: -0.8736056, upper bound: 0.8738008
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 39.36
Output dim: 0, lower bound: -0.8693261, upper bound: 0.8691493
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 39.36
Output dim: 0, lower bound: -0.8693261, upper bound: 0.8691493

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8439851, 1.8445339
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7871356, 2.7849140
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3525906, 2.3538570
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5949411, 2.5906200
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7574749, 2.7589648
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0433350, 2.0410290
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2803707, 2.2798750
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1747122, 3.1766858
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0482750, 2.0481687
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.3000436, 2.3030915

Time for backsubstitution: 22.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 5773

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4603

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8736005, upper bound: 0.8697073
time: 4.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8695118, upper bound: 0.8737961
time: 5.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8437791, 1.8447402
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7833858, 2.7886648
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3524237, 2.3540239
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5914145, 2.5941463
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7587805, 2.7576590
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0420785, 2.0422854
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2806406, 2.2796049
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1781883, 3.1732101
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0482950, 2.0481486
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.3023334, 2.3008018

Time for backsubstitution: 22.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8691481, upper bound: 0.8693265
time: 5.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8691481, upper bound: 0.8693265
time: 5.72 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 33.52 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.52
Output dim: 0, lower bound: -0.8736005, upper bound: 0.8697073
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.52
Output dim: 0, lower bound: -0.8695118, upper bound: 0.8737961
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 33.52
Output dim: 0, lower bound: -0.8691481, upper bound: 0.8693265
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 33.52
Output dim: 0, lower bound: -0.8691481, upper bound: 0.8693265

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8277440, 1.8259757
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7829037, 2.7822409
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3442297, 2.3441298
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5696397, 2.5697036
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7531743, 2.7537467
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0261087, 2.0266421
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2754693, 2.2755868
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1677790, 3.1687641
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0283508, 2.0307326
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2853427, 2.2860713

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 471

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8735968, upper bound: 0.8652645
time: 6.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8691230, upper bound: 0.8697039
time: 5.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8254275, 1.8282926
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7844629, 2.7806811
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3428636, 2.3454962
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5740247, 2.5653186
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7522569, 2.7546644
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0289478, 2.0238028
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2760825, 2.2749739
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1667910, 3.1697531
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0308390, 2.0282445
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2830234, 2.2883904

Time for backsubstitution: 21.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 5778

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 453

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8694911, upper bound: 0.8737946
time: 4.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8695103, upper bound: 0.8737754
time: 5.06 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.18 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.18
Output dim: 0, lower bound: -0.8735968, upper bound: 0.8652645
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 0, lower bound: -0.8691230, upper bound: 0.8697039
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.18
Output dim: 0, lower bound: -0.8694911, upper bound: 0.8737946
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.18
Output dim: 0, lower bound: -0.8695103, upper bound: 0.8737754

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8247900, 1.8218758
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7810144, 2.7796206
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3381667, 2.3397627
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5689282, 2.5691915
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7391491, 2.7342637
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0222511, 2.0238619
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2589245, 2.2636712
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1613493, 3.1598415
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0281587, 2.0304661
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2694850, 2.2640495

Time for backsubstitution: 21.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 552

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5847

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8735462, upper bound: 0.8652633
time: 5.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8735462, upper bound: 0.8650387
time: 4.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8253622, 1.8281014
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7832026, 2.7769413
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3369451, 2.3434808
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5716205, 2.5582170
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7418466, 2.7511342
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0280790, 2.0212500
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2754827, 2.2732072
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1660404, 3.1675487
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0303435, 2.0267844
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2819209, 2.2851460

Time for backsubstitution: 21.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 552

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8694841, upper bound: 0.8736090
time: 4.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8693057, upper bound: 0.8737874
time: 4.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8252363, 1.8282270
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7807231, 2.7794209
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3408484, 2.3395779
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5669227, 2.5629158
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7487264, 2.7442546
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0263958, 2.0229335
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2743154, 2.2743738
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1645851, 3.1690021
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0293789, 2.0277491
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2797794, 2.2872880

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 859

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 471

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8695065, upper bound: 0.8692680
time: 5.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8650799, upper bound: 0.8737750
time: 9.45 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 36.76 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.76
Output dim: 0, lower bound: -0.8735462, upper bound: 0.8652633
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.76
Output dim: 0, lower bound: -0.8735462, upper bound: 0.8650387
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.76
Output dim: 0, lower bound: -0.8694841, upper bound: 0.8736090
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.76
Output dim: 0, lower bound: -0.8693057, upper bound: 0.8737874
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 36.76
Output dim: 0, lower bound: -0.8695065, upper bound: 0.8692680
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.76
Output dim: 0, lower bound: -0.8650799, upper bound: 0.8737750

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8345363, 1.8334548
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7625275, 2.7584972
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3303075, 2.3328860
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5394940, 2.5355525
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.6917787, 2.6928205
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -1.9971514, 1.9951754
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2459240, 2.2488143
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1537895, 3.1532264
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0238919, 2.0255899
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2277093, 2.2280681

Time for backsubstitution: 21.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 864

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8729948, upper bound: 0.8641078
time: 5.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8724230, upper bound: 0.8647277
time: 5.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8363688, 1.8316226
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7598915, 2.7580833
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3312898, 2.3319037
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5352893, 2.5397573
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.6977067, 2.6868935
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -1.9935646, 1.9987626
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2440672, 2.2506714
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1528482, 3.1522818
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0232830, 2.0261989
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2335038, 2.2222743

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 5773

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8691027, upper bound: 0.8602667
time: 9.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8691027, upper bound: 0.8602667
time: 9.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8260200, 1.8279147
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7827601, 2.7784853
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3368530, 2.3438182
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5716033, 2.5582790
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7434711, 2.7506657
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0278373, 2.0220909
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2749677, 2.2749918
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1699295, 3.1664257
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0302153, 2.0272298
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2834187, 2.2847204

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 471

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8694804, upper bound: 0.8691185
time: 5.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8650365, upper bound: 0.8736062
time: 8.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8251750, 1.8281014
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7832026, 2.7764988
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3369451, 2.3433888
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5716205, 2.5581994
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7413788, 2.7511342
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0280790, 2.0210083
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2754827, 2.2726932
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1649170, 3.1675487
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0303435, 2.0266559
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2814951, 2.2851460

Time for backsubstitution: 22.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5773

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8684491, upper bound: 0.8705894
time: 5.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8661608, upper bound: 0.8728832
time: 15.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8211365, 1.8252730
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7781024, 2.7775326
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3364806, 2.3335142
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5664105, 2.5622048
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7292433, 2.7302289
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0236154, 2.0190759
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2624006, 2.2578282
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1556630, 3.1625724
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0291119, 2.0275569
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2577562, 2.2714293

Time for backsubstitution: 22.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 5778

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8602977, upper bound: 0.8693153
time: 5.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8602977, upper bound: 0.8693153
time: 5.31 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 33.08 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 33.08
Output dim: 0, lower bound: -0.8729948, upper bound: 0.8641078
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 33.08
Output dim: 0, lower bound: -0.8724230, upper bound: 0.8647277
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 33.08
Output dim: 0, lower bound: -0.8691027, upper bound: 0.8602667
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 33.08
Output dim: 0, lower bound: -0.8691027, upper bound: 0.8602667
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 33.08
Output dim: 0, lower bound: -0.8694804, upper bound: 0.8691185
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 33.08
Output dim: 0, lower bound: -0.8650365, upper bound: 0.8736062
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 33.08
Output dim: 0, lower bound: -0.8684491, upper bound: 0.8705894
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 33.08
Output dim: 0, lower bound: -0.8661608, upper bound: 0.8728832
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 33.08
Output dim: 0, lower bound: -0.8602977, upper bound: 0.8693153
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 33.08
Output dim: 0, lower bound: -0.8602977, upper bound: 0.8693153

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8345263, 1.8334522
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7622204, 2.7585602
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3304858, 2.3320384
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5389428, 2.5356693
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.6917996, 2.6927252
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -1.9965701, 1.9952960
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2455792, 2.2488852
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1537552, 3.1532326
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0239115, 2.0254717
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2277789, 2.2277391

Time for backsubstitution: 22.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 453

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8729758, upper bound: 0.8641065
time: 5.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8729935, upper bound: 0.8640714
time: 7.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8219206, 1.8249600
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7801404, 2.7765970
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3324847, 2.3377540
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5710912, 2.5575690
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7239885, 2.7366407
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0250568, 2.0182331
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2630520, 2.2584453
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1610069, 3.1599960
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0299482, 2.0270376
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2613964, 2.2688622

Time for backsubstitution: 22.54 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 61.92 + 551.87 = 613.79 seconds
