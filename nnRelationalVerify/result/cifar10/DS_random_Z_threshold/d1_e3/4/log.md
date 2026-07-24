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
execution time: IAR + RelationalAnalysis = 8.33 + 19.72 = 28.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0215737, upper bound: 0.0215750

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2372

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2565

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215731, upper bound: 0.0215746
time: 3.88 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215732, upper bound: 0.0215743
time: 3.34 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.23 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.23
Output dim: 0, lower bound: -0.0215731, upper bound: 0.0215746
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.23
Output dim: 0, lower bound: -0.0215732, upper bound: 0.0215743

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027894, 0.1027894
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0935078, 0.0935074
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2906554, 0.2906543
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2194148, 0.2194143
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3774672, 0.3774673
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2932266, 0.2932259
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4721384, 0.4721354
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1573420, 0.1573420
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117386, 0.4117385
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557493, 0.2557488

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3340

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2320

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215703, upper bound: 0.0215708
time: 63.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215702, upper bound: 0.0215715
time: 24.07 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027902, 0.1027894
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0935074, 0.0935273
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2906543, 0.2906588
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2194143, 0.2194172
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3774712, 0.3774672
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2932259, 0.2932299
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4721355, 0.4721537
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1573423, 0.1573420
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117385, 0.4117389
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2557488, 0.2557624

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3011

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3523

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0214878, upper bound: 0.0215736
time: 44.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215732, upper bound: 0.0214888
time: 3.56 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 54.35 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 54.35
Output dim: 0, lower bound: -0.0215703, upper bound: 0.0215708
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 54.35
Output dim: 0, lower bound: -0.0215702, upper bound: 0.0215715
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 54.35
Output dim: 0, lower bound: -0.0214878, upper bound: 0.0215736
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 54.35
Output dim: 0, lower bound: -0.0215732, upper bound: 0.0214888

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027759, 0.1027777
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934670, 0.0934593
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2906317, 0.2906313
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2193235, 0.2193204
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3774530, 0.3774527
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2931396, 0.2931364
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4719615, 0.4719977
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1572768, 0.1572736
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117210, 0.4117212
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2556925, 0.2556784

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3265

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2573

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215597, upper bound: 0.0215639
time: 4.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215630, upper bound: 0.0215605
time: 32.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027777, 0.1027760
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934598, 0.0934666
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2906325, 0.2906306
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2193209, 0.2193229
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3774525, 0.3774533
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2931370, 0.2931390
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4720006, 0.4719585
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1572736, 0.1572768
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117213, 0.4117209
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2556790, 0.2556920

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3228

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 714

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215690, upper bound: 0.0215703
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215690, upper bound: 0.0215700
time: 105.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1005771, 0.1006519
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0897678, 0.0896528
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2866647, 0.2868024
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2179054, 0.2178580
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3765100, 0.3765362
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2923974, 0.2923765
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4710696, 0.4710517
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1571261, 0.1571197
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4104848, 0.4104348
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2497597, 0.2495581

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 713

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3047

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0214866, upper bound: 0.0215687
time: 146.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0214830, upper bound: 0.0215729
time: 13.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1006528, 0.1005763
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0896329, 0.0897877
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2867980, 0.2866691
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2178551, 0.2179084
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3765403, 0.3765058
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2923724, 0.2924014
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4710338, 0.4710875
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1571200, 0.1571257
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4104345, 0.4104851
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2495445, 0.2497733

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3063

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215650, upper bound: 0.0214783
time: 5.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215628, upper bound: 0.0214805
time: 18.06 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.69 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -0.0215597, upper bound: 0.0215639
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -0.0215630, upper bound: 0.0215605
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -0.0215690, upper bound: 0.0215703
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -0.0215690, upper bound: 0.0215700
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -0.0214866, upper bound: 0.0215687
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -0.0214830, upper bound: 0.0215729
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -0.0215650, upper bound: 0.0214783
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -0.0215628, upper bound: 0.0214805

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027687, 0.1027719
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0933115, 0.0933102
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2904713, 0.2904609
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2184501, 0.2184224
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3768710, 0.3768479
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2920892, 0.2920607
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4701774, 0.4701421
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1564829, 0.1565448
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4116983, 0.4116987
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2556136, 0.2555991

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3291

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 713

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215591, upper bound: 0.0215625
time: 123.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215583, upper bound: 0.0215633
time: 16.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027701, 0.1027706
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0933179, 0.0933038
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2904614, 0.2904709
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2184255, 0.2184470
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3768483, 0.3768706
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2920640, 0.2920859
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4701059, 0.4702137
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1565480, 0.1564797
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4116985, 0.4116985
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2556132, 0.2555996

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2304

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 782

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215616, upper bound: 0.0215588
time: 16.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215617, upper bound: 0.0215596
time: 3.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027738, 0.1027711
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934340, 0.0934384
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2906026, 0.2906012
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2193237, 0.2193258
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3774567, 0.3774574
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2931395, 0.2931414
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4721833, 0.4721403
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1572407, 0.1572441
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117162, 0.4117156
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2556798, 0.2556927

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3046

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2066

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215687, upper bound: 0.0215547
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215530, upper bound: 0.0215543
time: 90.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027728, 0.1027720
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0934317, 0.0934407
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2906031, 0.2906007
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2193237, 0.2193258
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3774568, 0.3774574
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2931394, 0.2931415
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4721824, 0.4721413
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1572409, 0.1572440
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117161, 0.4117157
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2556797, 0.2556928

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3357

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3047

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215678, upper bound: 0.0215656
time: 127.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215641, upper bound: 0.0215693
time: 45.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1005566, 0.1006295
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0894683, 0.0893654
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2868451, 0.2869919
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2163126, 0.2163762
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3761411, 0.3761833
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2906233, 0.2907145
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4695967, 0.4696367
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1555188, 0.1556307
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4104927, 0.4104427
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2497323, 0.2495354

Time for backsubstitution: 6.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0214862, upper bound: 0.0215636
time: 35.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0214816, upper bound: 0.0215689
time: 16.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1005547, 0.1006314
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0894805, 0.0893532
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2868541, 0.2869829
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2164237, 0.2162652
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3761570, 0.3761674
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2907354, 0.2906024
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4696542, 0.4695792
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1556370, 0.1555124
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4104927, 0.4104428
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2497369, 0.2495308

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3324

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3265

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0214751, upper bound: 0.0215729
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0214829, upper bound: 0.0215650
time: 3.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1005573, 0.1004794
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0885603, 0.0886953
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2858131, 0.2856670
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2094758, 0.2096346
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3742303, 0.3741559
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2844042, 0.2845453
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4583484, 0.4586707
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1522755, 0.1521479
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4101936, 0.4102452
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2492146, 0.2494307

Time for backsubstitution: 6.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2573

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3373

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215649, upper bound: 0.0214782
time: 61.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215649, upper bound: 0.0214784
time: 75.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1005559, 0.1004808
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0885404, 0.0887151
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2857960, 0.2856841
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2095812, 0.2095291
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3741904, 0.3741958
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2845162, 0.2844333
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4586170, 0.4584023
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1521423, 0.1522812
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4101944, 0.4102443
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2492018, 0.2494434

Time for backsubstitution: 6.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3304

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215618, upper bound: 0.0214703
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0215527, upper bound: 0.0214795
time: 3.65 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 13.62 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0215591, upper bound: 0.0215625
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0215583, upper bound: 0.0215633
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0215616, upper bound: 0.0215588
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0215617, upper bound: 0.0215596
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0215687, upper bound: 0.0215547
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0215530, upper bound: 0.0215543
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0215678, upper bound: 0.0215656
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0215641, upper bound: 0.0215693
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0214862, upper bound: 0.0215636
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0214816, upper bound: 0.0215689
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0214751, upper bound: 0.0215729
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0214829, upper bound: 0.0215650
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0215649, upper bound: 0.0214782
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0215649, upper bound: 0.0214784
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0215618, upper bound: 0.0214703
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 13.62
Output dim: 0, lower bound: -0.0215527, upper bound: 0.0214795

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027582, 0.1027600
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0932858, 0.0932843
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2904286, 0.2904188
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2184264, 0.2183998
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3768440, 0.3768199
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2920641, 0.2920355
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4701140, 0.4700775
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1564818, 0.1565437
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4116892, 0.4116886
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2556128, 0.2555982

Time for backsubstitution: 6.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2364

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3291

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215306, upper bound: 0.0215623
time: 136.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215591, upper bound: 0.0215341
time: 3.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027568, 0.1027614
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0932857, 0.0932844
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2904292, 0.2904183
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2184275, 0.2183987
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3768431, 0.3768211
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2920640, 0.2920356
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4701128, 0.4700786
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1564818, 0.1565437
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4116882, 0.4116895
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2556128, 0.2555982

Time for backsubstitution: 6.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3291

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3046

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215580, upper bound: 0.0215619
time: 6.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215568, upper bound: 0.0215626
time: 112.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027670, 0.1027674
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0932875, 0.0932765
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2904374, 0.2904480
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2181260, 0.2182017
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3767926, 0.3768192
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2917574, 0.2918389
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4699311, 0.4700735
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1561581, 0.1561368
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4116948, 0.4116950
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2555892, 0.2555787

Time for backsubstitution: 6.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2061

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2628

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215416, upper bound: 0.0215565
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215592, upper bound: 0.0215394
time: 7.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027669, 0.1027674
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0932905, 0.0932735
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2904385, 0.2904469
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2181802, 0.2181475
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3767968, 0.3768148
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2918171, 0.2917792
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4699657, 0.4700389
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1562051, 0.1560898
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4116949, 0.4116949
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2555923, 0.2555755

Time for backsubstitution: 6.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3011

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215578, upper bound: 0.0215550
time: 27.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215580, upper bound: 0.0215555
time: 98.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1025652, 0.1025542
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0906258, 0.0907087
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2901977, 0.2901963
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2190380, 0.2190481
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3767101, 0.3766836
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2929707, 0.2929772
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4714038, 0.4713914
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1564106, 0.1564157
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117428, 0.4117421
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2545681, 0.2546107

Time for backsubstitution: 6.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3226

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215675, upper bound: 0.0215492
time: 22.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215644, upper bound: 0.0215532
time: 43.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1025568, 0.1025626
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0907043, 0.0906303
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2901978, 0.2901962
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2190461, 0.2190400
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3766829, 0.3767108
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2929753, 0.2929725
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4714344, 0.4713608
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1564124, 0.1564139
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117427, 0.4117422
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2545978, 0.2545809

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3288

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3373

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215529, upper bound: 0.0215702
time: 5.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215529, upper bound: 0.0215701
time: 26.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.1733330, 1.4548734, 1.1733330, 1.4548734, -0.1027523, 0.1027496
1: -1.2592157, -0.5768337, -1.2592157, -0.5768337, -0.0931321, 0.0931534
2: -1.7643511, -0.9834541, -1.7643511, -0.9834541, -0.2907836, 0.2907903
3: -3.4581785, -2.2832046, -3.4581785, -2.2832046, -0.2177309, 0.2178440
4: -3.9971070, -2.9876928, -3.9971070, -2.9876928, -0.3770882, 0.3771047
5: -4.3770971, -3.1457317, -4.3770971, -3.1457317, -0.2913653, 0.2914796
6: -5.3936253, -3.6086500, -5.3936253, -3.6086500, -0.4707093, 0.4707258
7: -5.3017774, -4.0830317, -5.3017774, -4.0830317, -0.1556336, 0.1557549
8: 0.0474678, 0.6737417, 0.0474678, 0.6737417, -0.4117241, 0.4117235
9: -1.5319901, -0.7777864, -1.5319901, -0.7777864, -0.2556524, 0.2556701

Time for backsubstitution: 6.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3463
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 329
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 595
type: DSZ, layer: 1, pos: 3025

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215662, upper bound: 0.0215606
time: 30.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215637, upper bound: 0.0215641
time: 50.58 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 87.61 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 87.61
Output dim: 0, lower bound: -0.0215306, upper bound: 0.0215623
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 87.61
Output dim: 0, lower bound: -0.0215591, upper bound: 0.0215341
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 87.61
Output dim: 0, lower bound: -0.0215580, upper bound: 0.0215619
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 87.61
Output dim: 0, lower bound: -0.0215568, upper bound: 0.0215626
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 87.61
Output dim: 0, lower bound: -0.0215416, upper bound: 0.0215565
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 87.61
Output dim: 0, lower bound: -0.0215592, upper bound: 0.0215394
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 87.61
Output dim: 0, lower bound: -0.0215578, upper bound: 0.0215550
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 87.61
Output dim: 0, lower bound: -0.0215580, upper bound: 0.0215555
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 87.61
Output dim: 0, lower bound: -0.0215675, upper bound: 0.0215492
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 87.61
Output dim: 0, lower bound: -0.0215644, upper bound: 0.0215532
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 87.61
Output dim: 0, lower bound: -0.0215529, upper bound: 0.0215702
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 87.61
Output dim: 0, lower bound: -0.0215529, upper bound: 0.0215701
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 87.61
Output dim: 0, lower bound: -0.0215662, upper bound: 0.0215606
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 87.61
Output dim: 0, lower bound: -0.0215637, upper bound: 0.0215641
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 87.61
Output dim: 0, lower bound: -0.0215641, upper bound: 0.0215693
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 87.61
Output dim: 0, lower bound: -0.0214862, upper bound: 0.0215636
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 87.61
Output dim: 0, lower bound: -0.0214816, upper bound: 0.0215689
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 87.61
Output dim: 0, lower bound: -0.0214751, upper bound: 0.0215729
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 87.61
Output dim: 0, lower bound: -0.0214829, upper bound: 0.0215650
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 87.61
Output dim: 0, lower bound: -0.0215649, upper bound: 0.0214782
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 87.61
Output dim: 0, lower bound: -0.0215649, upper bound: 0.0214784
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 87.61
Output dim: 0, lower bound: -0.0215618, upper bound: 0.0214703

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 28.05 + 1813.19 = 1841.24 seconds
