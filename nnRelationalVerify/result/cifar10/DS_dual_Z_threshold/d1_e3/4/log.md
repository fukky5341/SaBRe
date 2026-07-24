## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0215536248


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027902, 0.1027902)
1: (-1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0935273, 0.0935273)
2: (-1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2906588, 0.2906588)
3: (-3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2194172, 0.2194172)
4: (-3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3774712, 0.3774713)
5: (-4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2932299, 0.2932299)
6: (-5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4721535, 0.4721537)
7: (-5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1573423, 0.1573423)
8: (0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117389, 0.4117389)
9: (-1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557623, 0.2557624)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.11 + 18.83 = 25.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0215737, upper bound: 0.0215750

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3306

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215653, upper bound: 0.0215750
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215740, upper bound: 0.0215662
time: 2.80 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.82 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.82
Output dim: 0, lower bound: -0.0215653, upper bound: 0.0215750
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.82
Output dim: 0, lower bound: -0.0215740, upper bound: 0.0215662

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027900, 0.1027901
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0935278, 0.0935267
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2906597, 0.2906581
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2194115, 0.2193986
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3774861, 0.3774579
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2932299, 0.2932049
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4721589, 0.4721473
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1573419, 0.1573260
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117368, 0.4117379
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557620, 0.2557616

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2497

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215623, upper bound: 0.0215738
time: 30.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215643, upper bound: 0.0215715
time: 23.72 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027902, 0.1027900
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0935267, 0.0935273
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2906581, 0.2906588
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2193986, 0.2194172
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3774580, 0.3774713
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2932049, 0.2932299
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4721472, 0.4721537
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1573260, 0.1573423
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117389, 0.4117368
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557623, 0.2557621

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2497

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215710, upper bound: 0.0215651
time: 8.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215730, upper bound: 0.0215632
time: 3.63 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 17.38 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 17.38
Output dim: 0, lower bound: -0.0215623, upper bound: 0.0215738
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 17.38
Output dim: 0, lower bound: -0.0215643, upper bound: 0.0215715
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 17.38
Output dim: 0, lower bound: -0.0215710, upper bound: 0.0215651
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 17.38
Output dim: 0, lower bound: -0.0215730, upper bound: 0.0215632

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027882, 0.1027886
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0935175, 0.0935135
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2906541, 0.2906515
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2194099, 0.2193968
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3774807, 0.3774511
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2932288, 0.2932031
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4721535, 0.4721408
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1573383, 0.1573210
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117353, 0.4117364
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557619, 0.2557614

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3227

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215513, upper bound: 0.0215734
time: 14.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215620, upper bound: 0.0215625
time: 29.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027885, 0.1027883
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0935145, 0.0935165
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2906530, 0.2906525
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2194095, 0.2193971
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3774793, 0.3774524
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2932281, 0.2932038
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4721524, 0.4721419
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1573369, 0.1573224
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117353, 0.4117364
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557619, 0.2557614

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3227

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215533, upper bound: 0.0215711
time: 85.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215640, upper bound: 0.0215609
time: 65.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027885, 0.1027885
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0935165, 0.0935140
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2906525, 0.2906522
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2193971, 0.2194153
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3774524, 0.3774645
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2932038, 0.2932283
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4721419, 0.4721471
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1573224, 0.1573373
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117374, 0.4117354
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557623, 0.2557619

Time for backsubstitution: 5.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3227

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215601, upper bound: 0.0215649
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215707, upper bound: 0.0215538
time: 162.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027887, 0.1027882
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0935135, 0.0935171
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2906514, 0.2906532
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2193968, 0.2194157
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3774512, 0.3774657
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2932031, 0.2932290
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4721409, 0.4721482
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1573210, 0.1573387
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117373, 0.4117354
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557623, 0.2557619

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3227

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215621, upper bound: 0.0215630
time: 9.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215727, upper bound: 0.0215519
time: 53.96 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 69.81 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 69.81
Output dim: 0, lower bound: -0.0215513, upper bound: 0.0215734
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 69.81
Output dim: 0, lower bound: -0.0215620, upper bound: 0.0215625
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 69.81
Output dim: 0, lower bound: -0.0215533, upper bound: 0.0215711
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 69.81
Output dim: 0, lower bound: -0.0215640, upper bound: 0.0215609
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 69.81
Output dim: 0, lower bound: -0.0215601, upper bound: 0.0215649
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 69.81
Output dim: 0, lower bound: -0.0215707, upper bound: 0.0215538
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 69.81
Output dim: 0, lower bound: -0.0215621, upper bound: 0.0215630
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 69.81
Output dim: 0, lower bound: -0.0215727, upper bound: 0.0215519

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027807, 0.1027812
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934927, 0.0934877
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905933, 0.2905901
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2190934, 0.2190739
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3775444, 0.3775165
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2929944, 0.2929640
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722081, 0.4721940
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1571567, 0.1571350
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117821, 0.4117814
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557345, 0.2557335

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3228

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215471, upper bound: 0.0215738
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215514, upper bound: 0.0215685
time: 25.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027808, 0.1027810
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934918, 0.0934886
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905928, 0.2905907
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2190870, 0.2190802
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3775460, 0.3775148
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2929897, 0.2929688
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722067, 0.4721954
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1571522, 0.1571394
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117802, 0.4117833
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557339, 0.2557340

Time for backsubstitution: 6.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3228

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215576, upper bound: 0.0215630
time: 15.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215621, upper bound: 0.0215589
time: 22.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027809, 0.1027810
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934896, 0.0934907
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905923, 0.2905911
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2190931, 0.2190742
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3775431, 0.3775177
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2929937, 0.2929646
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722070, 0.4721950
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1571553, 0.1571363
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117821, 0.4117814
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557344, 0.2557335

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3228

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215492, upper bound: 0.0215712
time: 32.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215534, upper bound: 0.0215672
time: 3.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027811, 0.1027808
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934887, 0.0934916
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905917, 0.2905917
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2190866, 0.2190806
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3775446, 0.3775162
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2929890, 0.2929694
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722058, 0.4721965
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1571509, 0.1571408
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117802, 0.4117833
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557339, 0.2557340

Time for backsubstitution: 6.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3228

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215596, upper bound: 0.0215609
time: 20.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215641, upper bound: 0.0215569
time: 15.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027809, 0.1027811
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934916, 0.0934883
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905917, 0.2905908
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2190806, 0.2190924
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3775162, 0.3775297
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2929694, 0.2929892
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4721965, 0.4722004
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1571408, 0.1571513
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117843, 0.4117803
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557348, 0.2557339

Time for backsubstitution: 6.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3228

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215559, upper bound: 0.0215649
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215602, upper bound: 0.0215605
time: 21.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027811, 0.1027809
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934907, 0.0934892
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905912, 0.2905914
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2190742, 0.2190988
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3775177, 0.3775282
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2929646, 0.2929939
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4721950, 0.4722018
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1571363, 0.1571558
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117824, 0.4117822
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557342, 0.2557345

Time for backsubstitution: 6.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3228

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215663, upper bound: 0.0215534
time: 24.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215708, upper bound: 0.0215495
time: 24.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027811, 0.1027808
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934886, 0.0934913
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905907, 0.2905918
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2190803, 0.2190927
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3775148, 0.3775311
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2929688, 0.2929898
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4721953, 0.4722015
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1571394, 0.1571527
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117843, 0.4117803
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557347, 0.2557339

Time for backsubstitution: 6.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3228

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215579, upper bound: 0.0215624
time: 54.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215622, upper bound: 0.0215583
time: 16.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027813, 0.1027807
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934877, 0.0934922
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905901, 0.2905924
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2190738, 0.2190991
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3775165, 0.3775295
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2929640, 0.2929946
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4721941, 0.4722029
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1571350, 0.1571571
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117824, 0.4117822
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557342, 0.2557345

Time for backsubstitution: 6.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3228

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215683, upper bound: 0.0215514
time: 100.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215728, upper bound: 0.0215481
time: 15.95 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 123.46 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215471, upper bound: 0.0215738
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215514, upper bound: 0.0215685
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215576, upper bound: 0.0215630
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215621, upper bound: 0.0215589
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215492, upper bound: 0.0215712
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215534, upper bound: 0.0215672
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215596, upper bound: 0.0215609
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215641, upper bound: 0.0215569
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215559, upper bound: 0.0215649
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215602, upper bound: 0.0215605
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215663, upper bound: 0.0215534
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215708, upper bound: 0.0215495
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215579, upper bound: 0.0215624
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215622, upper bound: 0.0215583
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215683, upper bound: 0.0215514
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 123.46
Output dim: 0, lower bound: -0.0215728, upper bound: 0.0215481

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027745, 0.1027752
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934774, 0.0934719
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905384, 0.2905338
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2188797, 0.2188559
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3776081, 0.3775799
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2928118, 0.2927778
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722185, 0.4722019
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1569839, 0.1569578
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4118440, 0.4118415
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557146, 0.2557131

Time for backsubstitution: 6.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215468, upper bound: 0.0215689
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215414, upper bound: 0.0215579
time: 33.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027747, 0.1027750
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934768, 0.0934724
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905369, 0.2905353
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2188755, 0.2188602
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3776078, 0.3775802
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2928082, 0.2927814
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722161, 0.4722044
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1569795, 0.1569622
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4118423, 0.4118430
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557141, 0.2557135

Time for backsubstitution: 6.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215511, upper bound: 0.0215644
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215456, upper bound: 0.0215674
time: 21.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027747, 0.1027751
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934765, 0.0934728
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905378, 0.2905343
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2188733, 0.2188624
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3776096, 0.3775783
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2928070, 0.2927825
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722171, 0.4722034
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1569794, 0.1569622
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4118421, 0.4118434
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557140, 0.2557136

Time for backsubstitution: 6.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215572, upper bound: 0.0215581
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215518, upper bound: 0.0215576
time: 19.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027749, 0.1027749
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934760, 0.0934733
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905364, 0.2905358
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2188690, 0.2188666
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3776094, 0.3775785
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2928034, 0.2927861
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722147, 0.4722058
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1569750, 0.1569666
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4118404, 0.4118449
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557136, 0.2557141

Time for backsubstitution: 6.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215617, upper bound: 0.0215533
time: 27.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215563, upper bound: 0.0215575
time: 56.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027748, 0.1027750
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934743, 0.0934749
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905374, 0.2905348
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2188794, 0.2188563
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3776067, 0.3775811
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2928111, 0.2927784
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722176, 0.4722030
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1569825, 0.1569591
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4118438, 0.4118416
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557145, 0.2557131

Time for backsubstitution: 6.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215489, upper bound: 0.0215657
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215443, upper bound: 0.0215712
time: 3.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027749, 0.1027748
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934738, 0.0934755
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905360, 0.2905362
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2188751, 0.2188605
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3776065, 0.3775814
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2928075, 0.2927820
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722149, 0.4722055
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1569781, 0.1569635
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4118423, 0.4118431
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557141, 0.2557136

Time for backsubstitution: 6.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215531, upper bound: 0.0215612
time: 69.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215486, upper bound: 0.0215572
time: 55.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027749, 0.1027748
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934734, 0.0934758
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905369, 0.2905353
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2188730, 0.2188627
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3776083, 0.3775796
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2928064, 0.2927832
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722161, 0.4722044
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1569781, 0.1569636
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4118421, 0.4118434
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557140, 0.2557136

Time for backsubstitution: 6.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215593, upper bound: 0.0215549
time: 72.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215547, upper bound: 0.0215605
time: 49.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027751, 0.1027746
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934729, 0.0934763
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905354, 0.2905368
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2188687, 0.2188669
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3776081, 0.3775798
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2928028, 0.2927868
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722137, 0.4722068
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1569737, 0.1569680
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4118404, 0.4118449
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557135, 0.2557141

Time for backsubstitution: 6.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215638, upper bound: 0.0215497
time: 30.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215592, upper bound: 0.0215560
time: 34.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027747, 0.1027751
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934763, 0.0934724
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905368, 0.2905344
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2188669, 0.2188745
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3775798, 0.3775933
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2927867, 0.2928030
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722068, 0.4722085
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1569679, 0.1569741
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4118458, 0.4118404
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557149, 0.2557136

Time for backsubstitution: 6.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215555, upper bound: 0.0215600
time: 27.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215500, upper bound: 0.0215646
time: 4.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027749, 0.1027749
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934758, 0.0934730
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905353, 0.2905359
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2188627, 0.2188788
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3775796, 0.3775935
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2927831, 0.2928066
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722044, 0.4722110
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1569636, 0.1569785
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4118443, 0.4118420
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557145, 0.2557140

Time for backsubstitution: 6.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215597, upper bound: 0.0215547
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215543, upper bound: 0.0215490
time: 25.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027749, 0.1027749
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934754, 0.0934733
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905362, 0.2905350
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2188605, 0.2188810
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3775814, 0.3775917
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2927820, 0.2928077
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722054, 0.4722099
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1569635, 0.1569786
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4118440, 0.4118423
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557144, 0.2557141

Time for backsubstitution: 6.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215659, upper bound: 0.0215492
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215604, upper bound: 0.0215529
time: 17.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027751, 0.1027747
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934749, 0.0934738
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905348, 0.2905364
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2188563, 0.2188852
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3775811, 0.3775918
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2927784, 0.2928113
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722030, 0.4722124
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1569591, 0.1569829
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4118425, 0.4118438
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557139, 0.2557146

Time for backsubstitution: 6.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215704, upper bound: 0.0215450
time: 40.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215649, upper bound: 0.0215489
time: 30.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027750, 0.1027749
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934733, 0.0934754
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905359, 0.2905354
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2188666, 0.2188749
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3775785, 0.3775946
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2927861, 0.2928036
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722059, 0.4722096
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1569666, 0.1569755
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4118458, 0.4118404
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557149, 0.2557136

Time for backsubstitution: 6.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215575, upper bound: 0.0215572
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215530, upper bound: 0.0215621
time: 23.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027752, 0.1027747
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934728, 0.0934760
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905344, 0.2905369
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2188624, 0.2188791
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3775783, 0.3775948
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2927825, 0.2928072
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722035, 0.4722120
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1569622, 0.1569798
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4118443, 0.4118420
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557144, 0.2557140

Time for backsubstitution: 6.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3537

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215617, upper bound: 0.0215523
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215572, upper bound: 0.0215466
time: 21.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027751, 0.1027747
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934724, 0.0934763
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2905353, 0.2905359
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2188601, 0.2188812
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3775802, 0.3775930
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2927814, 0.2928084
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4722044, 0.4722110
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1569622, 0.1569799
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4118439, 0.4118423
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557143, 0.2557141

Time for backsubstitution: 6.89 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 25.94 + 1779.29 = 1805.23 seconds
