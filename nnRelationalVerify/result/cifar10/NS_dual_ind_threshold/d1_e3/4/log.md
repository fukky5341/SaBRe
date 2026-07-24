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
execution time: IAR + RelationalAnalysis = 7.78 + 19.40 = 27.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0215737, upper bound: 0.0215750

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3537

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213639, upper bound: 0.0215745
time: 70.39 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215739, upper bound: 0.0215750
time: 4.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 75.03 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 75.03
Output dim: 0, lower bound: -0.0213639, upper bound: 0.0215745
NS_A2, status: Status.UNKNOWN, split count: 1, time: 75.03
Output dim: 0, lower bound: -0.0215739, upper bound: 0.0215750

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 1.1749811, 1.4548731, 1.1747231, 1.4548731, -0.1011659, 0.1014004
1: -1.2591699, -0.5790222, -1.2591771, -0.5786794, -0.0915690, 0.0912546
2: -1.7619327, -0.9835079, -1.7623116, -0.9834991, -0.2882416, 0.2886160
3: -3.4581628, -2.2840471, -3.4581654, -2.2839384, -0.2187018, 0.2185721
4: -3.9962525, -2.9876928, -3.9963846, -2.9876928, -0.3765233, 0.3766259
5: -4.3770556, -3.1461363, -4.3770618, -3.1460774, -0.2926421, 0.2925388
6: -5.3936234, -3.6092007, -5.3936238, -3.6091321, -0.4713435, 0.4712927
7: -5.3017697, -4.0831599, -5.3017707, -4.0831404, -0.1571557, 0.1571297
8: 0.0474708, 0.6728528, 0.0474705, 0.6729919, -0.4107434, 0.4106822
9: -1.5319812, -0.7812617, -1.5319828, -0.7807174, -0.2530263, 0.2525517

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3523

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0213636, upper bound: 0.0214896
time: 3.98 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213638, upper bound: 0.0215747
time: 14.98 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 1.1733145, 1.4587438, 1.1733358, 1.4548733, -0.1014867, 0.1064895
1: -1.2647135, -0.5768335, -1.2592152, -0.5768363, -0.0987967, 0.0916366
2: -1.7643512, -0.9774867, -1.7643483, -0.9834546, -0.2888699, 0.2966524
3: -3.4602427, -2.2832146, -3.4581778, -2.2832432, -0.2208461, 0.2187814
4: -3.9971945, -2.9858513, -3.9970984, -2.9876928, -0.3770384, 0.3786172
5: -4.3782539, -3.1460438, -4.3770967, -3.1460495, -0.2941186, 0.2927666
6: -5.3946905, -3.6086259, -5.3936253, -3.6088336, -0.4724332, 0.4724100
7: -5.3020239, -4.0832720, -5.3017774, -4.0832438, -0.1576095, 0.1571759
8: 0.0455594, 0.6737947, 0.0474684, 0.6737397, -0.4128959, 0.4115604
9: -1.5407782, -0.7777889, -1.5319898, -0.7777897, -0.2638552, 0.2529721

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3523

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215737, upper bound: 0.0214900
time: 3.54 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215737, upper bound: 0.0215748
time: 27.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 36.91 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 36.91
Output dim: 0, lower bound: -0.0213636, upper bound: 0.0214896
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 36.91
Output dim: 0, lower bound: -0.0213638, upper bound: 0.0215747
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 36.91
Output dim: 0, lower bound: -0.0215737, upper bound: 0.0214900
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 36.91
Output dim: 0, lower bound: -0.0215737, upper bound: 0.0215748

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 1.1749820, 1.4548730, 1.1747243, 1.4548731, -0.1011597, 0.0992598
1: -1.2591699, -0.5790240, -1.2591769, -0.5786818, -0.0878296, 0.0912313
2: -1.7619308, -0.9835081, -1.7623093, -0.9835001, -0.2882332, 0.2847567
3: -3.4581628, -2.2840536, -3.4581654, -2.2839456, -0.2171757, 0.2185425
4: -3.9962356, -2.9876928, -3.9963641, -2.9876928, -0.3765154, 0.3756916
5: -4.3770542, -3.1461539, -4.3770609, -3.1460958, -0.2917871, 0.2925140
6: -5.3936243, -3.6092093, -5.3936243, -3.6091413, -0.4702669, 0.4712772
7: -5.3017697, -4.0831842, -5.3017707, -4.0831680, -0.1569380, 0.1571148
8: 0.0474715, 0.6728505, 0.0474710, 0.6729894, -0.4094833, 0.4106181
9: -1.5319812, -0.7812644, -1.5319825, -0.7807212, -0.2470356, 0.2525209

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2389

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213579, upper bound: 0.0215655
time: 3.12 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213580, upper bound: 0.0215693
time: 3.78 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 1.1746659, 1.4587433, 1.1748960, 1.4540322, -0.0993176, 0.1049434
1: -1.2646712, -0.5790495, -1.2576743, -0.5793866, -0.0962337, 0.0879209
2: -1.7615485, -0.9775724, -1.7611226, -0.9853862, -0.2841894, 0.2934222
3: -3.4602377, -2.2841461, -3.4576216, -2.2842937, -0.2197685, 0.2171800
4: -3.9966309, -2.9858513, -3.9964414, -2.9880333, -0.3761034, 0.3779576
5: -4.3782425, -3.1466119, -4.3768120, -3.1466932, -0.2934323, 0.2917924
6: -5.3946867, -3.6093113, -5.3931894, -3.6096027, -0.4716725, 0.4712658
7: -5.3020172, -4.0834055, -5.3017216, -4.0833893, -0.1574808, 0.1569384
8: 0.0455613, 0.6729747, 0.0479791, 0.6727929, -0.4119964, 0.4102135
9: -1.5407724, -0.7813348, -1.5295885, -0.7818688, -0.2599435, 0.2472064

Time for backsubstitution: 6.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2389

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215682, upper bound: 0.0214801
time: 64.69 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0214838
time: 17.95 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 1.1733154, 1.4587438, 1.1733367, 1.4548733, -0.1014804, 0.1043452
1: -1.2647138, -0.5768355, -1.2592152, -0.5768387, -0.0950446, 0.0916116
2: -1.7643492, -0.9774867, -1.7643461, -0.9834547, -0.2888613, 0.2927929
3: -3.4602427, -2.2832203, -3.4581780, -2.2832508, -0.2193156, 0.2187519
4: -3.9971771, -2.9858513, -3.9970779, -2.9876928, -0.3770304, 0.3776828
5: -4.3782530, -3.1460605, -4.3770957, -3.1460676, -0.2932624, 0.2927418
6: -5.3946905, -3.6086345, -5.3936253, -3.6088419, -0.4713535, 0.4723938
7: -5.3020239, -4.0832953, -5.3017774, -4.0832701, -0.1573917, 0.1571610
8: 0.0455596, 0.6737928, 0.0474689, 0.6737369, -0.4116306, 0.4114965
9: -1.5407782, -0.7777920, -1.5319896, -0.7777938, -0.2578621, 0.2529411

Time for backsubstitution: 6.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2389

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0215655
time: 29.45 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0215693
time: 3.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 38.94 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 38.94
Output dim: 0, lower bound: -0.0213579, upper bound: 0.0215655
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 38.94
Output dim: 0, lower bound: -0.0213580, upper bound: 0.0215693
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 38.94
Output dim: 0, lower bound: -0.0215682, upper bound: 0.0214801
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 38.94
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0214838
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 38.94
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0215655
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 38.94
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0215693

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 1.1749945, 1.4548132, 1.1747351, 1.4548210, -0.1010993, 0.0991943
1: -1.2591580, -0.5790396, -1.2591664, -0.5786952, -0.0878161, 0.0912165
2: -1.7617050, -0.9835839, -1.7621124, -0.9835663, -0.2876631, 0.2841914
3: -3.4538360, -2.2840753, -3.4543839, -2.2839651, -0.2133780, 0.2150721
4: -3.9949961, -2.9876928, -3.9952779, -2.9876928, -0.3751303, 0.3744602
5: -4.3723512, -3.1461728, -4.3729506, -3.1461120, -0.2875090, 0.2887561
6: -5.3912468, -3.6092176, -5.3915405, -3.6091490, -0.4681796, 0.4694334
7: -5.2983131, -4.0831838, -5.2987452, -4.0831690, -0.1532221, 0.1538169
8: 0.0474904, 0.6728408, 0.0474881, 0.6729805, -0.4092324, 0.4103661
9: -1.5318716, -0.7812877, -1.5318874, -0.7807410, -0.2469268, 0.2524251

Time for backsubstitution: 6.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0213578, upper bound: 0.0213763
time: 17.93 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213578, upper bound: 0.0215650
time: 21.73 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 1.1749012, 1.4548086, 1.1747350, 1.4548155, -0.1012156, 0.0991920
1: -1.2591858, -0.5790470, -1.2591672, -0.5787028, -0.0878927, 0.0912133
2: -1.7616732, -0.9833526, -1.7620813, -0.9835179, -0.2882172, 0.2845690
3: -3.4573145, -2.2789226, -3.4573984, -2.2839651, -0.2137128, 0.2226692
4: -3.9958897, -2.9863844, -3.9960318, -2.9876928, -0.3753227, 0.3767409
5: -4.3761435, -3.1405332, -4.3762312, -3.1461086, -0.2879114, 0.2971085
6: -5.3929663, -3.6064231, -5.3930397, -3.6091504, -0.4683765, 0.4730564
7: -5.3005719, -4.0794787, -5.3007050, -4.0831680, -0.1534974, 0.1612026
8: 0.0474620, 0.6728797, 0.0474846, 0.6729831, -0.4094787, 0.4104579
9: -1.5318599, -0.7812407, -1.5318730, -0.7807431, -0.2469209, 0.2524690

Time for backsubstitution: 6.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0213580, upper bound: 0.0213799
time: 101.09 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213580, upper bound: 0.0215684
time: 199.74 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 1.1746784, 1.4586835, 1.1749071, 1.4539800, -0.0992571, 0.1048776
1: -1.2646599, -0.5790646, -1.2576637, -0.5793998, -0.0962199, 0.0879062
2: -1.7613224, -0.9776478, -1.7609257, -0.9854524, -0.2836191, 0.2928563
3: -3.4559107, -2.2841687, -3.4538410, -2.2843137, -0.2159709, 0.2137097
4: -3.9953918, -2.9858513, -3.9953556, -2.9880333, -0.3747178, 0.3767264
5: -4.3735394, -3.1466300, -4.3727021, -3.1467087, -0.2891541, 0.2880346
6: -5.3923111, -3.6093199, -5.3911052, -3.6096117, -0.4695849, 0.4694221
7: -5.2985601, -4.0834060, -5.2986960, -4.0833898, -0.1537649, 0.1536405
8: 0.0455800, 0.6729649, 0.0479965, 0.6727846, -0.4117453, 0.4099613
9: -1.5406630, -0.7813567, -1.5294935, -0.7818890, -0.2598348, 0.2471108

Time for backsubstitution: 6.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215680, upper bound: 0.0212701
time: 56.19 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215682, upper bound: 0.0212702
time: 46.53 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 1.1745851, 1.4586790, 1.1749070, 1.4539747, -0.0993734, 0.1048755
1: -1.2646875, -0.5790724, -1.2576643, -0.5794075, -0.0962961, 0.0879030
2: -1.7612909, -0.9774168, -1.7608947, -0.9854044, -0.2841734, 0.2932295
3: -3.4593894, -2.2790158, -3.4568551, -2.2843132, -0.2163054, 0.2213068
4: -3.9962859, -2.9845428, -3.9961097, -2.9880333, -0.3749101, 0.3790069
5: -4.3773317, -3.1409900, -4.3759818, -3.1467052, -0.2895565, 0.2963867
6: -5.3940296, -3.6065259, -5.3926044, -3.6096127, -0.4697821, 0.4730448
7: -5.3008199, -4.0796995, -5.3006554, -4.0833893, -0.1540401, 0.1610263
8: 0.0455517, 0.6730032, 0.0479932, 0.6727870, -0.4119916, 0.4100531
9: -1.5406507, -0.7813101, -1.5294785, -0.7818911, -0.2598288, 0.2471547

Time for backsubstitution: 6.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215680, upper bound: 0.0212735
time: 6.07 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0212740
time: 3.18 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 1.1733278, 1.4586838, 1.1733476, 1.4548211, -0.1014200, 0.1042796
1: -1.2647017, -0.5768502, -1.2592049, -0.5768516, -0.0950309, 0.0915969
2: -1.7641232, -0.9775622, -1.7641499, -0.9835211, -0.2882912, 0.2922277
3: -3.4559159, -2.2832432, -3.4543967, -2.2832704, -0.2155181, 0.2152815
4: -3.9959383, -2.9858513, -3.9959927, -2.9876928, -0.3756449, 0.3764517
5: -4.3735495, -3.1460786, -4.3729854, -3.1460834, -0.2889841, 0.2889840
6: -5.3923120, -3.6086440, -5.3915415, -3.6088512, -0.4692664, 0.4705497
7: -5.2985673, -4.0832958, -5.2987509, -4.0832710, -0.1536760, 0.1538632
8: 0.0455788, 0.6737831, 0.0474861, 0.6737285, -0.4113791, 0.4112446
9: -1.5406690, -0.7778149, -1.5318944, -0.7778139, -0.2577534, 0.2528453

Time for backsubstitution: 6.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0213546
time: 19.81 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215680, upper bound: 0.0213556
time: 3.36 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 1.1732346, 1.4586792, 1.1733477, 1.4548157, -0.1015364, 0.1042773
1: -1.2647294, -0.5768580, -1.2592055, -0.5768600, -0.0951071, 0.0915937
2: -1.7640921, -0.9773312, -1.7641182, -0.9834734, -0.2888449, 0.2926030
3: -3.4593945, -2.2780900, -3.4574106, -2.2832692, -0.2158526, 0.2228784
4: -3.9968319, -2.9845428, -3.9967465, -2.9876928, -0.3758373, 0.3787323
5: -4.3773413, -3.1404390, -4.3762655, -3.1460795, -0.2893866, 0.2973362
6: -5.3940320, -3.6058478, -5.3930407, -3.6088517, -0.4694633, 0.4741726
7: -5.3008265, -4.0795898, -5.3007107, -4.0832710, -0.1539512, 0.1612488
8: 0.0455503, 0.6738218, 0.0474827, 0.6737311, -0.4116256, 0.4113365
9: -1.5406556, -0.7777674, -1.5318798, -0.7778161, -0.2577474, 0.2528895

Time for backsubstitution: 6.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0213595
time: 3.22 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0213587
time: 3.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 12.90 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 12.90
Output dim: 0, lower bound: -0.0213578, upper bound: 0.0213763
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.90
Output dim: 0, lower bound: -0.0213578, upper bound: 0.0215650
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 12.90
Output dim: 0, lower bound: -0.0213580, upper bound: 0.0213799
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.90
Output dim: 0, lower bound: -0.0213580, upper bound: 0.0215684
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.90
Output dim: 0, lower bound: -0.0215680, upper bound: 0.0212701
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.90
Output dim: 0, lower bound: -0.0215682, upper bound: 0.0212702
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.90
Output dim: 0, lower bound: -0.0215680, upper bound: 0.0212735
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.90
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0212740
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.90
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0213546
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.90
Output dim: 0, lower bound: -0.0215680, upper bound: 0.0213556
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.90
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0213595
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.90
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0213587

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 1.1749945, 1.4548132, 1.1733264, 1.4586915, -0.1048127, 0.1005730
1: -1.2591580, -0.5790396, -1.2647032, -0.5768487, -0.0896184, 0.0966142
2: -1.7617050, -0.9835839, -1.7641521, -0.9775532, -0.2937008, 0.2861849
3: -3.4538360, -2.2840753, -3.4564614, -2.2832413, -0.2136343, 0.2169363
4: -3.9949961, -2.9876928, -3.9960890, -2.9858513, -0.3766208, 0.3749796
5: -4.3723512, -3.1461728, -4.3741426, -3.1460781, -0.2877184, 0.2899855
6: -5.3912468, -3.6092176, -5.3926048, -3.6086440, -0.4684442, 0.4703710
7: -5.2983131, -4.0831838, -5.2989979, -4.0832996, -0.1532480, 0.1542225
8: 0.0474904, 0.6728408, 0.0455768, 0.6737840, -0.4097431, 0.4118689
9: -1.5318716, -0.7812877, -1.5406823, -0.7778130, -0.2494769, 0.2606820

Time for backsubstitution: 6.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2623

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213515, upper bound: 0.0215611
time: 6.01 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213535, upper bound: 0.0215605
time: 54.32 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 1.1749012, 1.4548086, 1.1733263, 1.4586864, -0.1049291, 0.1005708
1: -1.2591858, -0.5790470, -1.2647039, -0.5768566, -0.0896951, 0.0966110
2: -1.7616732, -0.9833526, -1.7641212, -0.9775051, -0.2942555, 0.2865623
3: -3.4573145, -2.2789226, -3.4594755, -2.2832413, -0.2139689, 0.2245334
4: -3.9958897, -2.9863844, -3.9968426, -2.9858513, -0.3768134, 0.3772600
5: -4.3761435, -3.1405332, -4.3774223, -3.1460748, -0.2881210, 0.2983378
6: -5.3929663, -3.6064231, -5.3941050, -3.6086457, -0.4686412, 0.4739940
7: -5.3005719, -4.0794787, -5.3009572, -4.0832992, -0.1535232, 0.1616082
8: 0.0474620, 0.6728797, 0.0455736, 0.6737864, -0.4099894, 0.4119610
9: -1.5318599, -0.7812407, -1.5406680, -0.7778151, -0.2494708, 0.2607261

Time for backsubstitution: 5.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2623

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213516, upper bound: 0.0215646
time: 35.35 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213533, upper bound: 0.0215650
time: 3.70 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 1.1746784, 1.4586835, 1.1765518, 1.4539797, -0.1005466, 0.1032686
1: -1.2646599, -0.5790646, -1.2576180, -0.5815856, -0.0940772, 0.0896372
2: -1.7613224, -0.9776478, -1.7585098, -0.9855062, -0.2853589, 0.2904835
3: -3.4559107, -2.2841687, -3.4538255, -2.2851160, -0.2155628, 0.2138859
4: -3.9953918, -2.9858513, -3.9945102, -2.9880333, -0.3748224, 0.3761232
5: -4.3735394, -3.1466300, -4.3726616, -3.1467953, -0.2888039, 0.2881199
6: -5.3923111, -3.6093199, -5.3911028, -3.6099787, -0.4693837, 0.4686177
7: -5.2985601, -4.0834060, -5.2986875, -4.0833063, -0.1536908, 0.1536455
8: 0.0455800, 0.6729649, 0.0479989, 0.6718982, -0.4110367, 0.4096538
9: -1.5406630, -0.7813567, -1.5294850, -0.7853601, -0.2567887, 0.2497149

Time for backsubstitution: 5.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2623

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215614, upper bound: 0.0212672
time: 52.31 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215636, upper bound: 0.0212669
time: 20.44 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 1.1746784, 1.4586835, 1.1748854, 1.4578505, -0.0992583, 0.0998771
1: -1.2646599, -0.5790646, -1.2631621, -0.5793968, -0.0890676, 0.0879096
2: -1.7613224, -0.9776478, -1.7609284, -0.9794847, -0.2845318, 0.2859865
3: -3.4559107, -2.2841687, -3.4559042, -2.2842855, -0.2139707, 0.2137726
4: -3.9953918, -2.9858513, -3.9954519, -2.9861913, -0.3747029, 0.3751337
5: -4.3735394, -3.1466300, -4.3738599, -3.1467037, -0.2879978, 0.2882301
6: -5.3923111, -3.6093199, -5.3921685, -3.6094031, -0.4695673, 0.4694250
7: -5.2985601, -4.0834060, -5.2989430, -4.0834184, -0.1533355, 0.1536444
8: 0.0455800, 0.6729649, 0.0460877, 0.6728396, -0.4104221, 0.4099711
9: -1.5406630, -0.7813567, -1.5382807, -0.7818878, -0.2489661, 0.2471237

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2623

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215615, upper bound: 0.0212684
time: 4.53 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215638, upper bound: 0.0212682
time: 3.90 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 1.1745851, 1.4586790, 1.1765521, 1.4539742, -0.1006631, 0.1032663
1: -1.2646875, -0.5790724, -1.2576189, -0.5815935, -0.0941533, 0.0896341
2: -1.7612909, -0.9774168, -1.7584786, -0.9854585, -0.2859132, 0.2908564
3: -3.4593894, -2.2790158, -3.4568398, -2.2851160, -0.2158973, 0.2214831
4: -3.9962859, -2.9845428, -3.9952641, -2.9880333, -0.3750148, 0.3784037
5: -4.3773317, -3.1409900, -4.3759413, -3.1467924, -0.2892064, 0.2964717
6: -5.3940296, -3.6065259, -5.3926010, -3.6099801, -0.4695810, 0.4722406
7: -5.3008199, -4.0796995, -5.3006482, -4.0833063, -0.1539660, 0.1610314
8: 0.0455517, 0.6730032, 0.0479957, 0.6719000, -0.4112831, 0.4097458
9: -1.5406507, -0.7813101, -1.5294700, -0.7853622, -0.2567827, 0.2497589

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2623

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215614, upper bound: 0.0212721
time: 12.36 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215635, upper bound: 0.0212712
time: 11.32 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 1.1745851, 1.4586790, 1.1748855, 1.4578450, -0.0993746, 0.0998749
1: -1.2646875, -0.5790724, -1.2631631, -0.5794045, -0.0891441, 0.0879064
2: -1.7612909, -0.9774168, -1.7608970, -0.9794371, -0.2850869, 0.2863594
3: -3.4593894, -2.2790158, -3.4589186, -2.2842839, -0.2143053, 0.2213697
4: -3.9962859, -2.9845428, -3.9962058, -2.9861913, -0.3748951, 0.3774139
5: -4.3773317, -3.1409900, -4.3771381, -3.1467009, -0.2884003, 0.2965822
6: -5.3940296, -3.6065259, -5.3936672, -3.6094046, -0.4697645, 0.4730479
7: -5.3008199, -4.0796995, -5.3009019, -4.0834179, -0.1536107, 0.1610303
8: 0.0455517, 0.6730032, 0.0460839, 0.6728421, -0.4106684, 0.4100628
9: -1.5406507, -0.7813101, -1.5382662, -0.7818902, -0.2489600, 0.2471676

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2623

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215613, upper bound: 0.0212720
time: 3.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215638, upper bound: 0.0212717
time: 58.46 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 1.1733278, 1.4586838, 1.1749928, 1.4548208, -0.1027132, 0.1026701
1: -1.2647017, -0.5768502, -1.2591592, -0.5790379, -0.0928881, 0.0933345
2: -1.7641232, -0.9775622, -1.7617338, -0.9835749, -0.2900307, 0.2898547
3: -3.4559159, -2.2832432, -3.4543815, -2.2840734, -0.2151097, 0.2154601
4: -3.9959383, -2.9858513, -3.9951468, -2.9876928, -0.3757517, 0.3758485
5: -4.3735495, -3.1460786, -4.3729453, -3.1461716, -0.2886340, 0.2890693
6: -5.3923120, -3.6086440, -5.3915396, -3.6092184, -0.4690652, 0.4697500
7: -5.2985673, -4.0832958, -5.2987432, -4.0831881, -0.1536018, 0.1538687
8: 0.0455788, 0.6737831, 0.0474885, 0.6728419, -0.4106710, 0.4109409
9: -1.5406690, -0.7778149, -1.5318861, -0.7812853, -0.2547072, 0.2554497

Time for backsubstitution: 5.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2623

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215615, upper bound: 0.0213506
time: 125.02 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215638, upper bound: 0.0213510
time: 3.37 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 1.1733278, 1.4586838, 1.1733264, 1.4586915, -0.1014212, 0.0992845
1: -1.2647017, -0.5768502, -1.2647032, -0.5768487, -0.0878917, 0.0916004
2: -1.7641232, -0.9775622, -1.7641521, -0.9775532, -0.2892038, 0.2853579
3: -3.4559159, -2.2832432, -3.4564614, -2.2832413, -0.2135209, 0.2153445
4: -3.9959383, -2.9858513, -3.9960890, -2.9858513, -0.3756298, 0.3748600
5: -4.3735495, -3.1460786, -4.3741426, -3.1460781, -0.2878288, 0.2891795
6: -5.3923120, -3.6086440, -5.3926048, -3.6086440, -0.4692510, 0.4705530
7: -5.2985673, -4.0832958, -5.2989979, -4.0832996, -0.1532470, 0.1538671
8: 0.0455788, 0.6737831, 0.0455768, 0.6737840, -0.4100603, 0.4112545
9: -1.5406690, -0.7778149, -1.5406823, -0.7778130, -0.2468855, 0.2528594

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2623

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215615, upper bound: 0.0213507
time: 25.08 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215638, upper bound: 0.0213511
time: 3.29 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 1.1732346, 1.4586792, 1.1749928, 1.4548155, -0.1028297, 0.1026680
1: -1.2647294, -0.5768580, -1.2591598, -0.5790455, -0.0929643, 0.0933313
2: -1.7640921, -0.9773312, -1.7617024, -0.9835263, -0.2905848, 0.2902301
3: -3.4593945, -2.2780900, -3.4573956, -2.2840731, -0.2154443, 0.2230572
4: -3.9968319, -2.9845428, -3.9959016, -2.9876928, -0.3759440, 0.3781290
5: -4.3773413, -3.1404390, -4.3762245, -3.1461687, -0.2890365, 0.2974215
6: -5.3940320, -3.6058478, -5.3930392, -3.6092203, -0.4692624, 0.4733729
7: -5.3008265, -4.0795898, -5.3007030, -4.0831876, -0.1538771, 0.1612545
8: 0.0455503, 0.6738218, 0.0474854, 0.6728443, -0.4109175, 0.4110336
9: -1.5406556, -0.7777674, -1.5318720, -0.7812873, -0.2547011, 0.2554938

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2623

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215615, upper bound: 0.0213546
time: 4.28 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215636, upper bound: 0.0213542
time: 34.26 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 1.1732346, 1.4586792, 1.1733263, 1.4586864, -0.1015375, 0.0992823
1: -1.2647294, -0.5768580, -1.2647039, -0.5768566, -0.0879683, 0.0915972
2: -1.7640921, -0.9773312, -1.7641212, -0.9775051, -0.2897585, 0.2857332
3: -3.4593945, -2.2780900, -3.4594755, -2.2832413, -0.2138555, 0.2229414
4: -3.9968319, -2.9845428, -3.9968426, -2.9858513, -0.3758223, 0.3771403
5: -4.3773413, -3.1404390, -4.3774223, -3.1460748, -0.2882314, 0.2975318
6: -5.3940320, -3.6058478, -5.3941050, -3.6086457, -0.4694488, 0.4741758
7: -5.3008265, -4.0795898, -5.3009572, -4.0832992, -0.1535223, 0.1612528
8: 0.0455503, 0.6738218, 0.0455736, 0.6737864, -0.4103068, 0.4113464
9: -1.5406556, -0.7777674, -1.5406680, -0.7778151, -0.2468795, 0.2529035

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2623

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215615, upper bound: 0.0213549
time: 3.16 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215638, upper bound: 0.0213543
time: 14.40 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 23.63 seconds
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0213515, upper bound: 0.0215611
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0213535, upper bound: 0.0215605
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0213516, upper bound: 0.0215646
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0213533, upper bound: 0.0215650
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215614, upper bound: 0.0212672
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215636, upper bound: 0.0212669
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215615, upper bound: 0.0212684
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215638, upper bound: 0.0212682
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215614, upper bound: 0.0212721
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215635, upper bound: 0.0212712
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215613, upper bound: 0.0212720
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215638, upper bound: 0.0212717
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215615, upper bound: 0.0213506
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215638, upper bound: 0.0213510
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215615, upper bound: 0.0213507
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215638, upper bound: 0.0213511
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215615, upper bound: 0.0213546
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215636, upper bound: 0.0213542
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215615, upper bound: 0.0213549
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 0, lower bound: -0.0215638, upper bound: 0.0213543

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 1.1750729, 1.4548132, 1.1733949, 1.4586916, -0.1047302, 0.1004601
1: -1.2591554, -0.5791982, -1.2647009, -0.5769875, -0.0893633, 0.0964169
2: -1.7615292, -0.9835981, -1.7639995, -0.9775653, -0.2935103, 0.2860160
3: -3.4538364, -2.2845244, -3.4564614, -2.2836792, -0.2128779, 0.2162036
4: -3.9949939, -2.9877219, -3.9960866, -2.9858761, -0.3765545, 0.3749149
5: -4.3723516, -3.1466007, -4.3741426, -3.1464787, -0.2869811, 0.2892255
6: -5.3912463, -3.6094642, -5.3926058, -3.6088555, -0.4681363, 0.4700799
7: -5.2983122, -4.0837493, -5.2989969, -4.0837889, -0.1524045, 0.1533597
8: 0.0474914, 0.6727760, 0.0455780, 0.6737275, -0.4096527, 0.4118011
9: -1.5318720, -0.7815597, -1.5406822, -0.7780521, -0.2490926, 0.2604327

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2573

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0213444, upper bound: 0.0215497
time: 6.57 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213443, upper bound: 0.0215539
time: 4.05 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 1.1750764, 1.4548800, 1.1733997, 1.4586916, -0.1047322, 0.1007034
1: -1.2593327, -0.5791726, -1.2647012, -0.5769663, -0.0899583, 0.0964262
2: -1.7615231, -0.9834527, -1.7639902, -0.9775634, -0.2935193, 0.2863284
3: -3.4540925, -2.2847679, -3.4564614, -2.2838366, -0.2148007, 0.2162124
4: -3.9949784, -2.9877877, -3.9960864, -2.9859333, -0.3766911, 0.3749141
5: -4.3724232, -3.1470947, -4.3741431, -3.1468492, -0.2887688, 0.2892270
6: -5.3913622, -3.6096201, -5.3926048, -3.6089938, -0.4686472, 0.4700768
7: -5.2986970, -4.0839276, -5.2989974, -4.0839477, -0.1545069, 0.1533756
8: 0.0474281, 0.6727765, 0.0455779, 0.6737258, -0.4098438, 0.4118004
9: -1.5321665, -0.7815212, -1.5406821, -0.7780199, -0.2499602, 0.2604465

Time for backsubstitution: 5.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2573

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0213463, upper bound: 0.0215508
time: 4.03 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213461, upper bound: 0.0215539
time: 2.91 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 1.1749797, 1.4548085, 1.1733950, 1.4586864, -0.1048465, 0.1004579
1: -1.2591834, -0.5792054, -1.2647017, -0.5769957, -0.0894400, 0.0964137
2: -1.7614975, -0.9833665, -1.7639686, -0.9775174, -0.2940649, 0.2863931
3: -3.4573150, -2.2793708, -3.4594755, -2.2836788, -0.2132124, 0.2238007
4: -3.9958882, -2.9864135, -3.9968410, -2.9858761, -0.3767469, 0.3771952
5: -4.3761435, -3.1409605, -4.3774223, -3.1464753, -0.2873836, 0.2975778
6: -5.3929653, -3.6066701, -5.3941040, -3.6088583, -0.4683336, 0.4737029
7: -5.3005719, -4.0800438, -5.3009572, -4.0837889, -0.1526798, 0.1607454
8: 0.0474628, 0.6728142, 0.0455744, 0.6737298, -0.4098991, 0.4118931
9: -1.5318596, -0.7815132, -1.5406675, -0.7780540, -0.2490867, 0.2604767

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2573

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213445, upper bound: 0.0215539
time: 12.39 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213444, upper bound: 0.0215571
time: 162.15 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 1.1749835, 1.4548755, 1.1733994, 1.4586864, -0.1048484, 0.1007011
1: -1.2593607, -0.5791800, -1.2647018, -0.5769744, -0.0900347, 0.0964230
2: -1.7614920, -0.9832217, -1.7639589, -0.9775153, -0.2940738, 0.2867053
3: -3.4575710, -2.2796154, -3.4594755, -2.2838364, -0.2151353, 0.2238096
4: -3.9958720, -2.9864793, -3.9968405, -2.9859333, -0.3768834, 0.3771947
5: -4.3762140, -3.1414547, -4.3774223, -3.1468449, -0.2891715, 0.2975793
6: -5.3930812, -3.6068242, -5.3941050, -3.6089938, -0.4688443, 0.4736999
7: -5.3009567, -4.0802217, -5.3009572, -4.0839477, -0.1547821, 0.1607613
8: 0.0473996, 0.6728147, 0.0455745, 0.6737286, -0.4100900, 0.4118921
9: -1.5321536, -0.7814742, -1.5406675, -0.7780224, -0.2499539, 0.2604907

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2573

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213462, upper bound: 0.0215547
time: 3.37 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213463, upper bound: 0.0215573
time: 97.86 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 1.1747570, 1.4586835, 1.1766205, 1.4539796, -0.1004281, 0.1031624
1: -1.2646573, -0.5792295, -1.2576159, -0.5817322, -0.0938239, 0.0893891
2: -1.7611438, -0.9776618, -1.7583544, -0.9855191, -0.2851649, 0.2903101
3: -3.4559102, -2.2847066, -3.4538250, -2.2855458, -0.2147795, 0.2130532
4: -3.9953895, -2.9858804, -3.9945078, -2.9880574, -0.3747672, 0.3760472
5: -4.3735390, -3.1471288, -4.3726616, -3.1472139, -0.2880157, 0.2872804
6: -5.3923101, -3.6095870, -5.3911023, -3.6102171, -0.4690592, 0.4682710
7: -5.2985597, -4.0839791, -5.2986884, -4.0838094, -0.1528470, 0.1526981
8: 0.0455807, 0.6728981, 0.0480000, 0.6718395, -0.4109625, 0.4095618
9: -1.5406625, -0.7816361, -1.5294844, -0.7856085, -0.2564532, 0.2493386

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2573

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215541, upper bound: 0.0212574
time: 19.02 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215542, upper bound: 0.0212607
time: 11.27 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 1.1747544, 1.4587505, 1.1766155, 1.4539796, -0.1004318, 0.1033581
1: -1.2648348, -0.5791922, -1.2576165, -0.5816871, -0.0942795, 0.0893984
2: -1.7611556, -0.9775164, -1.7583787, -0.9855154, -0.2851797, 0.2906335
3: -3.4562550, -2.2847667, -3.4538245, -2.2856445, -0.2165067, 0.2130636
4: -3.9953647, -2.9859462, -3.9945078, -2.9881244, -0.3748596, 0.3760450
5: -4.3736897, -3.1474166, -4.3726616, -3.1475272, -0.2896842, 0.2872822
6: -5.3924265, -3.6096933, -5.3911028, -3.6102903, -0.4695349, 0.4682717
7: -5.2989440, -4.0841084, -5.2986879, -4.0839195, -0.1547759, 0.1527144
8: 0.0455183, 0.6729021, 0.0479994, 0.6718419, -0.4110883, 0.4095641
9: -1.5409573, -0.7815737, -1.5294845, -0.7855343, -0.2570852, 0.2493530

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2573

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215564, upper bound: 0.0212575
time: 12.74 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215563, upper bound: 0.0212609
time: 3.57 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 1.1747570, 1.4586835, 1.1749543, 1.4578502, -0.0991559, 0.0997708
1: -1.2646573, -0.5792295, -1.2631598, -0.5795432, -0.0888204, 0.0876754
2: -1.7611438, -0.9776618, -1.7607734, -0.9794970, -0.2843383, 0.2858133
3: -3.4559102, -2.2847066, -3.4559040, -2.2847619, -0.2131875, 0.2129458
4: -3.9953895, -2.9858804, -3.9954495, -2.9862151, -0.3746465, 0.3750560
5: -4.3735390, -3.1471288, -4.3738594, -3.1471515, -0.2872092, 0.2873909
6: -5.3923101, -3.6095870, -5.3921680, -3.6096416, -0.4692508, 0.4691039
7: -5.2985597, -4.0839791, -5.2989421, -4.0839210, -0.1524934, 0.1527048
8: 0.0455807, 0.6728981, 0.0460883, 0.6727815, -0.4103475, 0.4098914
9: -1.5406625, -0.7816361, -1.5382806, -0.7821355, -0.2486302, 0.2468032

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2573

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215541, upper bound: 0.0212571
time: 28.89 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215542, upper bound: 0.0212603
time: 41.98 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 27.18 + 1789.99 = 1817.17 seconds
