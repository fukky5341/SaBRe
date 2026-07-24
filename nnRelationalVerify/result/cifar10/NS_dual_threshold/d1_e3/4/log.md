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
execution time: IAR + RelationalAnalysis = 7.77 + 18.85 = 26.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0215737, upper bound: 0.0215750

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3537
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 3523
type: A, layer: 1, pos: 2389
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 3316
type: B, layer: 1, pos: 3316
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 3463
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 3265
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2329
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3537

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213639, upper bound: 0.0215745
time: 71.71 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215739, upper bound: 0.0215750
time: 4.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 76.40 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 76.40
Output dim: 0, lower bound: -0.0213639, upper bound: 0.0215745
NS_A2, status: Status.UNKNOWN, split count: 1, time: 76.40
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
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 2389
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 3316
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: A, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3523

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0213636, upper bound: 0.0214896
time: 4.07 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213638, upper bound: 0.0215747
time: 15.28 seconds

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

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3523
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 2389
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 3316
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 3463
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 2329

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3523

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215737, upper bound: 0.0214900
time: 3.51 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215737, upper bound: 0.0215748
time: 27.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 37.38 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 37.38
Output dim: 0, lower bound: -0.0213636, upper bound: 0.0214896
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 37.38
Output dim: 0, lower bound: -0.0213638, upper bound: 0.0215747
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 37.38
Output dim: 0, lower bound: -0.0215737, upper bound: 0.0214900
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 37.38
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

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 2389
type: B, layer: 1, pos: 3316
type: A, layer: 1, pos: 3316
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3463
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 3047
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3265
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 2329
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: A, layer: 1, pos: 2329

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213542, upper bound: 0.0215685
time: 144.62 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213580, upper bound: 0.0215686
time: 24.20 seconds

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

Time for backsubstitution: 5.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2389
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 3316
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2329
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 2329

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2389

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215682, upper bound: 0.0214801
time: 64.58 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0214838
time: 18.04 seconds

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

Time for backsubstitution: 5.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2389
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 3316
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3463
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 2329
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 2329

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2389

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0215655
time: 29.62 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0215693
time: 3.10 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 38.36 seconds
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 38.36
Output dim: 0, lower bound: -0.0213542, upper bound: 0.0215685
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 38.36
Output dim: 0, lower bound: -0.0213580, upper bound: 0.0215686
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 38.36
Output dim: 0, lower bound: -0.0215682, upper bound: 0.0214801
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 38.36
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0214838
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 38.36
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0215655
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 38.36
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0215693

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: 1.1749928, 1.4548208, 1.1747366, 1.4548132, -0.1010942, 0.0991994
1: -1.2591591, -0.5790377, -1.2591649, -0.5786971, -0.0878149, 0.0912176
2: -1.7617339, -0.9835748, -1.7620834, -0.9835753, -0.2876679, 0.2841864
3: -3.4543815, -2.2840724, -3.4538383, -2.2839684, -0.2137053, 0.2147449
4: -3.9951501, -2.9876928, -3.9951248, -2.9876928, -0.3752847, 0.3743063
5: -4.3729448, -3.1461706, -4.3723574, -3.1461143, -0.2880294, 0.2882358
6: -5.3915396, -3.6092172, -5.3912473, -3.6091497, -0.4684235, 0.4691898
7: -5.2987447, -4.0831838, -5.2983141, -4.0831690, -0.1536400, 0.1533990
8: 0.0474885, 0.6728421, 0.0474899, 0.6729794, -0.4092317, 0.4103668
9: -1.5318861, -0.7812847, -1.5318737, -0.7807438, -0.2469397, 0.2524123

Time for backsubstitution: 5.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3316
type: A, layer: 1, pos: 3316
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3463
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 3047
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3265
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: A, layer: 1, pos: 2329

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3316

## Relational analysis of NS_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213316, upper bound: 0.0215686
time: 24.78 seconds

## Relational analysis of NS_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213540, upper bound: 0.0215690
time: 3.35 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: 1.1749927, 1.4548155, 1.1746434, 1.4548087, -0.1010920, 0.0993158
1: -1.2591598, -0.5790452, -1.2591931, -0.5787044, -0.0878116, 0.0912941
2: -1.7617028, -0.9835263, -1.7620521, -0.9833444, -0.2880458, 0.2847404
3: -3.4573958, -2.2840722, -3.4573174, -2.2788153, -0.2213022, 0.2150795
4: -3.9959044, -2.9876928, -3.9960182, -2.9863844, -0.3775651, 0.3744984
5: -4.3762245, -3.1461675, -4.3761497, -3.1404743, -0.2963816, 0.2886382
6: -5.3930392, -3.6092191, -5.3929663, -3.6063547, -0.4720465, 0.4693871
7: -5.3007030, -4.0831842, -5.3005738, -4.0794616, -0.1610258, 0.1536742
8: 0.0474855, 0.6728443, 0.0474615, 0.6730175, -0.4093232, 0.4106134
9: -1.5318718, -0.7812870, -1.5318608, -0.7806971, -0.2469839, 0.2524062

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3316
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 2389
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2329
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 2329

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3316

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0213578, upper bound: 0.0215461
time: 110.41 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213579, upper bound: 0.0215696
time: 5.41 seconds

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

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3316
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2329
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2329

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3316

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0214573
time: 68.11 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215679, upper bound: 0.0214791
time: 23.60 seconds

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

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3316
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3463
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3316

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0214609
time: 32.04 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215680, upper bound: 0.0214839
time: 3.24 seconds

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

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3316
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3463
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 2329
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 2329

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3316

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0215430
time: 24.31 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0215651
time: 34.79 seconds

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

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3316
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3047
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: A, layer: 1, pos: 2329

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3316

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215455, upper bound: 0.0215689
time: 24.00 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0215686
time: 26.48 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 56.55 seconds
NS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 56.55
Output dim: 0, lower bound: -0.0213316, upper bound: 0.0215686
NS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 56.55
Output dim: 0, lower bound: -0.0213540, upper bound: 0.0215690
NS_A1_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 56.55
Output dim: 0, lower bound: -0.0213578, upper bound: 0.0215461
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 56.55
Output dim: 0, lower bound: -0.0213579, upper bound: 0.0215696
NS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 56.55
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0214573
NS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 56.55
Output dim: 0, lower bound: -0.0215679, upper bound: 0.0214791
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 56.55
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0214609
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 56.55
Output dim: 0, lower bound: -0.0215680, upper bound: 0.0214839
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 56.55
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0215430
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 56.55
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0215651
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 56.55
Output dim: 0, lower bound: -0.0215455, upper bound: 0.0215689
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 56.55
Output dim: 0, lower bound: -0.0215681, upper bound: 0.0215686

## BFS NS instance: NS_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: 1.1750031, 1.4547747, 1.1747484, 1.4547616, -0.1010446, 0.0991554
1: -1.2591470, -0.5791206, -1.2591513, -0.5787925, -0.0877075, 0.0911236
2: -1.7617319, -0.9836237, -1.7620811, -0.9836314, -0.2876561, 0.2841754
3: -3.4543817, -2.2841136, -3.4538376, -2.2840140, -0.2136899, 0.2147297
4: -3.9951448, -2.9876971, -3.9951200, -2.9876983, -0.3752713, 0.3742909
5: -4.3729448, -3.1462171, -4.3723569, -3.1461663, -0.2879951, 0.2882013
6: -5.3915396, -3.6093376, -5.3912468, -3.6092892, -0.4682847, 0.4690689
7: -5.2987413, -4.0831995, -5.2983108, -4.0831852, -0.1536169, 0.1533786
8: 0.0475425, 0.6728359, 0.0475519, 0.6729726, -0.4091446, 0.4102768
9: -1.5318849, -0.7813256, -1.5318713, -0.7807910, -0.2469118, 0.2523859

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3463
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 3047
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3265
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2329
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0213316, upper bound: 0.0212948
time: 53.56 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2

### Relational analysis result of NS_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213316, upper bound: 0.0215695
time: 3.87 seconds

## BFS NS instance: NS_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: 1.1749930, 1.4548206, 1.1744385, 1.4548129, -0.1010506, 0.0995370
1: -1.2591593, -0.5790493, -1.2593579, -0.5787075, -0.0877195, 0.0915111
2: -1.7617338, -0.9835867, -1.7620649, -0.9835865, -0.2876696, 0.2840824
3: -3.4543819, -2.2840726, -3.4539137, -2.2839060, -0.2138571, 0.2147745
4: -3.9951491, -2.9877369, -3.9951029, -2.9877434, -0.3753005, 0.3742134
5: -4.3729448, -3.1461754, -4.3724556, -3.1460004, -0.2882524, 0.2882544
6: -5.3915396, -3.6092181, -5.3914223, -3.6091325, -0.4683323, 0.4693617
7: -5.2987437, -4.0831895, -5.2983432, -4.0831733, -0.1536208, 0.1534522
8: 0.0474895, 0.6728417, 0.0474460, 0.6731470, -0.4093094, 0.4103937
9: -1.5318861, -0.7812916, -1.5319352, -0.7807498, -0.2469194, 0.2524658

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3463
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 2389
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 3316
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2329
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2329
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A1_B2_B1_B2_B1

### Relational analysis result of NS_A1_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0213541, upper bound: 0.0213798
time: 17.25 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2

### Relational analysis result of NS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213541, upper bound: 0.0215684
time: 52.22 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: 1.1746948, 1.4548154, 1.1746435, 1.4548084, -0.1014295, 0.0992723
1: -1.2593521, -0.5790560, -1.2591932, -0.5787163, -0.0881055, 0.0911988
2: -1.7616844, -0.9835377, -1.7620518, -0.9833565, -0.2879416, 0.2847423
3: -3.4574709, -2.2840095, -3.4573176, -2.2788157, -0.2213317, 0.2152315
4: -3.9958808, -2.9877434, -3.9960182, -2.9864285, -0.3774720, 0.3745144
5: -4.3763218, -3.1460536, -4.3761492, -3.1404793, -0.2964001, 0.2888613
6: -5.3932133, -3.6091993, -5.3929663, -3.6063545, -0.4722180, 0.4692960
7: -5.3007321, -4.0831895, -5.3005733, -4.0794668, -0.1610788, 0.1536550
8: 0.0474415, 0.6730114, 0.0474627, 0.6730171, -0.4093499, 0.4106915
9: -1.5319331, -0.7812930, -1.5318606, -0.7807038, -0.2470372, 0.2523859

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3463
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 3047
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3265
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2389
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3316
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0213579, upper bound: 0.0213802
time: 22.47 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213579, upper bound: 0.0215687
time: 11.93 seconds

## BFS NS instance: NS_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: 1.1746902, 1.4586320, 1.1749172, 1.4539341, -0.0992130, 0.1048281
1: -1.2646456, -0.5791600, -1.2576517, -0.5794829, -0.0961258, 0.0877988
2: -1.7613205, -0.9777039, -1.7609239, -0.9855015, -0.2836081, 0.2928447
3: -3.4559102, -2.2842145, -3.4538403, -2.2843537, -0.2159557, 0.2136942
4: -3.9953876, -2.9858570, -3.9953513, -2.9880371, -0.3747027, 0.3767129
5: -4.3735385, -3.1466823, -4.3727016, -3.1467552, -0.2891195, 0.2880004
6: -5.3923101, -3.6094577, -5.3911047, -3.6097319, -0.4694644, 0.4692838
7: -5.2985568, -4.0834227, -5.2986927, -4.0834045, -0.1537446, 0.1536175
8: 0.0456420, 0.6729577, 0.0480508, 0.6727784, -0.4116549, 0.4098744
9: -1.5406615, -0.7814038, -1.5294913, -0.7819302, -0.2598084, 0.2470829

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2329
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2329

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 2573

## Relational analysis of NS_A2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0214468
time: 23.10 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215609, upper bound: 0.0214507
time: 3.54 seconds

## BFS NS instance: NS_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: 1.1743802, 1.4586833, 1.1749070, 1.4539795, -0.0995947, 0.1048341
1: -1.2648523, -0.5790752, -1.2576637, -0.5794116, -0.0965127, 0.0878109
2: -1.7613041, -0.9776593, -1.7609258, -0.9854639, -0.2835153, 0.2928580
3: -3.4559863, -2.2841063, -3.4538407, -2.2843142, -0.2160004, 0.2138616
4: -3.9953680, -2.9859018, -3.9953556, -2.9880769, -0.3746250, 0.3767426
5: -4.3736377, -3.1465166, -4.3727021, -3.1467140, -0.2891726, 0.2882575
6: -5.3924856, -3.6093011, -5.3911047, -3.6096127, -0.4697567, 0.4693309
7: -5.2985902, -4.0834112, -5.2986960, -4.0833941, -0.1538181, 0.1536214
8: 0.0455362, 0.6731322, 0.0479978, 0.6727840, -0.4117719, 0.4100394
9: -1.5407245, -0.7813625, -1.5294930, -0.7818955, -0.2598882, 0.2470903

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 2329

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 2573

## Relational analysis of NS_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0214692
time: 19.20 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0214727
time: 39.37 seconds

## BFS NS instance: NS_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: 1.1745970, 1.4586273, 1.1749170, 1.4539287, -0.0993293, 0.1048261
1: -1.2646739, -0.5791676, -1.2576523, -0.5794909, -0.0962019, 0.0877956
2: -1.7612886, -0.9774725, -1.7608924, -0.9854538, -0.2841621, 0.2932176
3: -3.4593892, -2.2790625, -3.4568543, -2.2843540, -0.2162904, 0.2212913
4: -3.9962811, -2.9845486, -3.9961059, -2.9880371, -0.3748950, 0.3789933
5: -4.3773293, -3.1410427, -4.3759809, -3.1467514, -0.2895221, 0.2963524
6: -5.3940296, -3.6066632, -5.3926039, -3.6097333, -0.4696614, 0.4729066
7: -5.3008175, -4.0797172, -5.3006520, -4.0834045, -0.1540197, 0.1610032
8: 0.0456140, 0.6729960, 0.0480469, 0.6727805, -0.4119012, 0.4099665
9: -1.5406485, -0.7813570, -1.5294766, -0.7819321, -0.2598023, 0.2471269

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3463
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2329
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 2329
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 2573

## Relational analysis of NS_A2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0214510
time: 14.40 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215609, upper bound: 0.0214536
time: 28.34 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: 1.1742871, 1.4586785, 1.1749068, 1.4539745, -0.0997109, 0.1048319
1: -1.2648801, -0.5790826, -1.2576646, -0.5794196, -0.0965888, 0.0878077
2: -1.7612727, -0.9774281, -1.7608945, -0.9854161, -0.2840692, 0.2932311
3: -3.4594650, -2.2789536, -3.4568548, -2.2843134, -0.2163349, 0.2214583
4: -3.9962626, -2.9845939, -3.9961097, -2.9880769, -0.3748173, 0.3790231
5: -4.3774290, -3.1408761, -4.3759818, -3.1467106, -0.2895752, 0.2966101
6: -5.3942051, -3.6065063, -5.3926039, -3.6096115, -0.4699538, 0.4729539
7: -5.3008490, -4.0797052, -5.3006554, -4.0833945, -0.1540932, 0.1610071
8: 0.0455078, 0.6731706, 0.0479946, 0.6727866, -0.4120183, 0.4101312
9: -1.5407120, -0.7813159, -1.5294785, -0.7818978, -0.2598820, 0.2471344

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3463
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 3047
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: A, layer: 1, pos: 2329

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2573

## Relational analysis of NS_A2_B1_A2_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215576, upper bound: 0.0214760
time: 26.09 seconds

## Relational analysis of NS_A2_B1_A2_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215609, upper bound: 0.0214768
time: 3.70 seconds

## BFS NS instance: NS_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: 1.1733396, 1.4586322, 1.1733581, 1.4547751, -0.1013760, 0.1042301
1: -1.2646874, -0.5769460, -1.2591931, -0.5769352, -0.0949368, 0.0914894
2: -1.7641215, -0.9776185, -1.7641478, -0.9835704, -0.2882802, 0.2922159
3: -3.4559155, -2.2832885, -3.4543958, -2.2833109, -0.2155031, 0.2152662
4: -3.9959338, -2.9858570, -3.9959888, -2.9876966, -0.3756298, 0.3764384
5: -4.3735485, -3.1461320, -4.3729844, -3.1461287, -0.2889498, 0.2889498
6: -5.3923116, -3.6087818, -5.3915415, -3.6089709, -0.4691454, 0.4704115
7: -5.2985640, -4.0833130, -5.2987485, -4.0832863, -0.1536556, 0.1538399
8: 0.0456409, 0.6737764, 0.0475404, 0.6737226, -0.4112890, 0.4111580
9: -1.5406666, -0.7778618, -1.5318928, -0.7778552, -0.2577269, 0.2528174

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3463
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 2329
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 2329

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2573

## Relational analysis of NS_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215609, upper bound: 0.0215323
time: 2.98 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0215360
time: 7.28 seconds

## BFS NS instance: NS_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: 1.1730297, 1.4586837, 1.1733477, 1.4548210, -0.1017576, 0.1042361
1: -1.2648938, -0.5768609, -1.2592050, -0.5768638, -0.0953238, 0.0915016
2: -1.7641051, -0.9775737, -1.7641493, -0.9835333, -0.2881871, 0.2922294
3: -3.4559913, -2.2831807, -3.4543967, -2.2832711, -0.2155475, 0.2154336
4: -3.9959157, -2.9859018, -3.9959927, -2.9877369, -0.3755519, 0.3764678
5: -4.3736472, -3.1459646, -4.3729858, -3.1460876, -0.2890027, 0.2892071
6: -5.3924866, -3.6086235, -5.3915415, -3.6088521, -0.4694381, 0.4704587
7: -5.2985969, -4.0833015, -5.2987509, -4.0832763, -0.1537292, 0.1538439
8: 0.0455346, 0.6739506, 0.0474874, 0.6737282, -0.4114059, 0.4113228
9: -1.5407300, -0.7778209, -1.5318946, -0.7778203, -0.2578068, 0.2528250

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3463
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 3047
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3265
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 3316
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 2329
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2573

## Relational analysis of NS_A2_B2_A1_A2_A1

### Relational analysis result of NS_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215576, upper bound: 0.0215583
time: 4.39 seconds

## Relational analysis of NS_A2_B2_A1_A2_A2

### Relational analysis result of NS_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0215574
time: 25.27 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 1.1732447, 1.4586333, 1.1733596, 1.4547645, -0.1014869, 0.1042334
1: -1.2647172, -0.5769411, -1.2591915, -0.5769550, -0.0949996, 0.0914996
2: -1.7640898, -0.9773803, -1.7641162, -0.9835296, -0.2888330, 0.2925919
3: -3.4593945, -2.2781301, -3.4574103, -2.2833157, -0.2158372, 0.2228633
4: -3.9968276, -2.9845476, -3.9967418, -2.9876986, -0.3758240, 0.3787169
5: -4.3773389, -3.1404848, -4.3762641, -3.1461329, -0.2893525, 0.2973021
6: -5.3940315, -3.6059692, -5.3930407, -3.6089907, -0.4693254, 0.4740521
7: -5.3008242, -4.0796051, -5.3007078, -4.0832882, -0.1539281, 0.1612284
8: 0.0456043, 0.6738156, 0.0475446, 0.6737236, -0.4115385, 0.4112464
9: -1.5406542, -0.7778090, -1.5318778, -0.7778631, -0.2577193, 0.2528629

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3047
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 3316
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: A, layer: 1, pos: 2329

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2573

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215352, upper bound: 0.0215617
time: 3.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215385, upper bound: 0.0215622
time: 8.40 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 1.1732347, 1.4586790, 1.1730493, 1.4548157, -0.1014929, 0.1046153
1: -1.2647295, -0.5768695, -1.2593980, -0.5768703, -0.0950118, 0.0918871
2: -1.7640924, -0.9773433, -1.7640996, -0.9834847, -0.2888467, 0.2924989
3: -3.4593949, -2.2780902, -3.4574862, -2.2832079, -0.2160047, 0.2229078
4: -3.9968319, -2.9845867, -3.9967241, -2.9877434, -0.3758534, 0.3786393
5: -4.3773408, -3.1404443, -4.3763633, -3.1459665, -0.2896098, 0.2973548
6: -5.3940320, -3.6058488, -5.3932161, -3.6088331, -0.4693727, 0.4743445
7: -5.3008261, -4.0795951, -5.3007398, -4.0832763, -0.1539319, 0.1613018
8: 0.0455517, 0.6738214, 0.0474390, 0.6738983, -0.4117039, 0.4113628
9: -1.5406559, -0.7777745, -1.5319414, -0.7778219, -0.2577270, 0.2529429

Time for backsubstitution: 5.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3463
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 3316
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 2329
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2329
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2573

## Relational analysis of NS_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0215588
time: 3.09 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0215619
time: 3.08 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 11.94 seconds
NS_A1_B2_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0213316, upper bound: 0.0212948
NS_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0213316, upper bound: 0.0215695
NS_A1_B2_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0213541, upper bound: 0.0213798
NS_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0213541, upper bound: 0.0215684
NS_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0213579, upper bound: 0.0213802
NS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0213579, upper bound: 0.0215687
NS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0214468
NS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215609, upper bound: 0.0214507
NS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0214692
NS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0214727
NS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0214510
NS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215609, upper bound: 0.0214536
NS_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215576, upper bound: 0.0214760
NS_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215609, upper bound: 0.0214768
NS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215609, upper bound: 0.0215323
NS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0215360
NS_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215576, upper bound: 0.0215583
NS_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0215574
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215352, upper bound: 0.0215617
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215385, upper bound: 0.0215622
NS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0215588
NS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0215619

## BFS NS instance: NS_A1_B2_B1_B1_B2

### Backsubstitution after applying NS history:
0: 1.1750031, 1.4547747, 1.1733398, 1.4586322, -0.1047580, 0.1005341
1: -1.2591470, -0.5791206, -1.2646874, -0.5769460, -0.0895098, 0.0965211
2: -1.7617319, -0.9836237, -1.7641215, -0.9776185, -0.2936939, 0.2861688
3: -3.4543817, -2.2841136, -3.4559150, -2.2832899, -0.2139462, 0.2165940
4: -3.9951448, -2.9876971, -3.9959307, -2.9858570, -0.3767618, 0.3748101
5: -4.3729448, -3.1462171, -4.3735485, -3.1461327, -0.2882047, 0.2894306
6: -5.3915396, -3.6093376, -5.3923116, -3.6087847, -0.4685494, 0.4700066
7: -5.2987413, -4.0831995, -5.2985640, -4.0833168, -0.1536428, 0.1537843
8: 0.0475425, 0.6728359, 0.0456412, 0.6737760, -0.4096558, 0.4117796
9: -1.5318849, -0.7813256, -1.5406671, -0.7778623, -0.2494616, 0.2606429

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3463
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 3047
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3265
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: A, layer: 1, pos: 2329

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2573

## Relational analysis of NS_A1_B2_B1_B1_B2_A1

### Relational analysis result of NS_A1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213211, upper bound: 0.0215618
time: 64.64 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2_A2

### Relational analysis result of NS_A1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213246, upper bound: 0.0215621
time: 9.03 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: 1.1749930, 1.4548206, 1.1730297, 1.4586837, -0.1047640, 0.1009160
1: -1.2591593, -0.5790493, -1.2648939, -0.5768611, -0.0895218, 0.0969078
2: -1.7617338, -0.9835867, -1.7641053, -0.9775736, -0.2937073, 0.2860758
3: -3.4543819, -2.2840726, -3.4559910, -2.2831817, -0.2141136, 0.2166387
4: -3.9951491, -2.9877369, -3.9959133, -2.9859018, -0.3767910, 0.3747325
5: -4.3729448, -3.1461754, -4.3736467, -3.1459656, -0.2884620, 0.2894837
6: -5.3915396, -3.6092181, -5.3924875, -3.6086245, -0.4685969, 0.4702992
7: -5.2987437, -4.0831895, -5.2985969, -4.0833049, -0.1536467, 0.1538578
8: 0.0474895, 0.6728417, 0.0455350, 0.6739501, -0.4098208, 0.4118966
9: -1.5318861, -0.7812916, -1.5407298, -0.7778218, -0.2494693, 0.2607227

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3047
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3265
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 3316
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2573

## Relational analysis of NS_A1_B2_B1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213435, upper bound: 0.0215619
time: 36.80 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213469, upper bound: 0.0215623
time: 3.04 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 1.1746948, 1.4548154, 1.1732348, 1.4586790, -0.1051429, 0.1006512
1: -1.2593521, -0.5790560, -1.2647294, -0.5768700, -0.0899079, 0.0965961
2: -1.7616844, -0.9835377, -1.7640921, -0.9773436, -0.2939769, 0.2867358
3: -3.4574709, -2.2840095, -3.4593945, -2.2780907, -0.2215880, 0.2170957
4: -3.9958808, -2.9877434, -3.9968295, -2.9845867, -0.3789625, 0.3750336
5: -4.3763218, -3.1460536, -4.3773403, -3.1404448, -0.2966097, 0.2900906
6: -5.3932133, -3.6091993, -5.3940320, -3.6058512, -0.4724827, 0.4702336
7: -5.3007321, -4.0831895, -5.3008261, -4.0795984, -0.1611047, 0.1540607
8: 0.0474415, 0.6730114, 0.0455518, 0.6738213, -0.4098611, 0.4121945
9: -1.5319331, -0.7812930, -1.5406560, -0.7777753, -0.2495871, 0.2606429

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3463
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 3047
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 2389
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3265
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 3316
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2329
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: A, layer: 1, pos: 2329

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2573

## Relational analysis of NS_A1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213474, upper bound: 0.0215621
time: 3.86 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0213507, upper bound: 0.0215616
time: 82.65 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: 1.1747600, 1.4586320, 1.1750029, 1.4539340, -0.0991602, 0.1047633
1: -1.2645044, -0.5791606, -1.2574778, -0.5794839, -0.0959612, 0.0876004
2: -1.7613187, -0.9778274, -1.7609220, -0.9856534, -0.2834168, 0.2926882
3: -3.4559083, -2.2849574, -3.4538374, -2.2852669, -0.2150119, 0.2128329
4: -3.9953830, -2.9862111, -3.9953456, -2.9884710, -0.3740396, 0.3761395
5: -4.3735323, -3.1474671, -4.3726950, -3.1477194, -0.2879906, 0.2870044
6: -5.3923101, -3.6107802, -5.3911037, -3.6113546, -0.4675698, 0.4676853
7: -5.2985554, -4.0837045, -5.2986908, -4.0837479, -0.1530077, 0.1530051
8: 0.0456576, 0.6729397, 0.0480692, 0.6727568, -0.4115868, 0.4098064
9: -1.5406590, -0.7815545, -1.5294893, -0.7821152, -0.2596054, 0.2469141

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2329

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A2_B1_A1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0212370
time: 3.15 seconds

## Relational analysis of NS_A2_B1_A1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215607, upper bound: 0.0212374
time: 3.39 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: 1.1749196, 1.4586318, 1.1751698, 1.4540975, -0.0989417, 0.1044813
1: -1.2641702, -0.5791616, -1.2571027, -0.5790509, -0.0963168, 0.0874725
2: -1.7613108, -0.9780754, -1.7614193, -0.9859320, -0.2834259, 0.2929843
3: -3.4559059, -2.2870770, -3.4564199, -2.2877169, -0.2147210, 0.2175358
4: -3.9953668, -2.9869752, -3.9966815, -2.9893582, -0.3741968, 0.3774194
5: -4.3735261, -3.1497490, -4.3753271, -3.1503634, -0.2877603, 0.2920063
6: -5.3923101, -3.6133835, -5.3957143, -3.6144412, -0.4675624, 0.4728559
7: -5.2985487, -4.0857906, -5.2991648, -4.0861988, -0.1529311, 0.1562942
8: 0.0457052, 0.6729010, 0.0481229, 0.6727381, -0.4115641, 0.4096748
9: -1.5406542, -0.7819970, -1.5300658, -0.7825608, -0.2592693, 0.2471425

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 2329
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: A, layer: 1, pos: 2329

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A2_B1_A1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215609, upper bound: 0.0212404
time: 3.49 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0212401
time: 107.52 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: 1.1744502, 1.4586830, 1.1749926, 1.4539796, -0.0995418, 0.1047693
1: -1.2647108, -0.5790758, -1.2574896, -0.5794123, -0.0963481, 0.0876124
2: -1.7613025, -0.9777825, -1.7609234, -0.9856163, -0.2833237, 0.2927018
3: -3.4559836, -2.2848489, -3.4538379, -2.2852268, -0.2150565, 0.2130004
4: -3.9953642, -2.9862556, -3.9953499, -2.9885106, -0.3739617, 0.3761688
5: -4.3736315, -3.1473017, -4.3726950, -3.1476786, -0.2880439, 0.2872618
6: -5.3924851, -3.6106234, -5.3911037, -3.6112339, -0.4678622, 0.4677327
7: -5.2985878, -4.0836930, -5.2986927, -4.0837379, -0.1530813, 0.1530090
8: 0.0455513, 0.6731147, 0.0480168, 0.6727624, -0.4117035, 0.4099715
9: -1.5407230, -0.7815131, -1.5294909, -0.7820807, -0.2596850, 0.2469215

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 3316
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 2272
type: B, layer: 1, pos: 2329
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A2_B1_A1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215607, upper bound: 0.0212596
time: 3.36 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0212594
time: 15.05 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: 1.1746099, 1.4586830, 1.1751596, 1.4541433, -0.0993232, 0.1044872
1: -1.2643758, -0.5790769, -1.2571145, -0.5789793, -0.0967036, 0.0874845
2: -1.7612946, -0.9780309, -1.7614204, -0.9858952, -0.2833332, 0.2929979
3: -3.4559808, -2.2869687, -3.4564209, -2.2876763, -0.2147659, 0.2177031
4: -3.9953482, -2.9870200, -3.9966855, -2.9893970, -0.3741189, 0.3774489
5: -4.3736253, -3.1495824, -4.3753276, -3.1503220, -0.2878141, 0.2922639
6: -5.3924837, -3.6132262, -5.3957143, -3.6143215, -0.4678552, 0.4729031
7: -5.2985806, -4.0857782, -5.2991672, -4.0861888, -0.1530046, 0.1562982
8: 0.0455992, 0.6730751, 0.0480698, 0.6727437, -0.4116809, 0.4098400
9: -1.5407178, -0.7819558, -1.5300673, -0.7825265, -0.2593489, 0.2471502

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3047
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 2329
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: A, layer: 1, pos: 2329

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215607, upper bound: 0.0212627
time: 31.61 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0212625
time: 7.23 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: 1.1746669, 1.4586275, 1.1750028, 1.4539286, -0.0992766, 0.1047611
1: -1.2645324, -0.5791678, -1.2574785, -0.5794919, -0.0960373, 0.0875972
2: -1.7612870, -0.9775959, -1.7608907, -0.9856051, -0.2839708, 0.2930613
3: -3.4593871, -2.2798042, -3.4568512, -2.2852654, -0.2153463, 0.2204303
4: -3.9962759, -2.9849026, -3.9960992, -2.9884710, -0.3742319, 0.3784198
5: -4.3773241, -3.1418276, -4.3759737, -3.1477151, -0.2883930, 0.2953567
6: -5.3940287, -3.6079850, -5.3926029, -3.6113560, -0.4677672, 0.4713082
7: -5.3008151, -4.0799985, -5.3006501, -4.0837479, -0.1532829, 0.1603909
8: 0.0456291, 0.6729788, 0.0480661, 0.6727589, -0.4118328, 0.4098985
9: -1.5406471, -0.7815068, -1.5294746, -0.7821172, -0.2595990, 0.2469580

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: A, layer: 1, pos: 3302
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2306
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3463
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 3047
type: A, layer: 1, pos: 3047
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 3226
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2329
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A2_B1_A2_A1_B1_B1

### Relational analysis result of NS_A2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215607, upper bound: 0.0212411
time: 5.23 seconds

## Relational analysis of NS_A2_B1_A2_A1_B1_B2

### Relational analysis result of NS_A2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215607, upper bound: 0.0212414
time: 3.35 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: 1.1748264, 1.4586272, 1.1751697, 1.4540921, -0.0990582, 0.1044790
1: -1.2641978, -0.5791690, -1.2571032, -0.5790586, -0.0963929, 0.0874693
2: -1.7612796, -0.9778445, -1.7613878, -0.9858840, -0.2839799, 0.2933576
3: -3.4593844, -2.2819238, -3.4594350, -2.2877159, -0.2150555, 0.2251334
4: -3.9962597, -2.9856668, -3.9974365, -2.9893582, -0.3743889, 0.3796995
5: -4.3773184, -3.1441097, -4.3786058, -3.1503601, -0.2881628, 0.3003582
6: -5.3940287, -3.6105895, -5.3972139, -3.6144421, -0.4677597, 0.4764786
7: -5.3008075, -4.0820847, -5.3011236, -4.0861988, -0.1532062, 0.1636801
8: 0.0456768, 0.6729395, 0.0481195, 0.6727402, -0.4118108, 0.4097669
9: -1.5406423, -0.7819493, -1.5300517, -0.7825632, -0.2592632, 0.2471865

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 3463
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 3047
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 2323
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2329
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 3357
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 3285
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 3228
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: A, layer: 1, pos: 2329

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A2_B1_A2_A1_B2_B1

### Relational analysis result of NS_A2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215607, upper bound: 0.0212444
time: 3.30 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2_B2

### Relational analysis result of NS_A2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215608, upper bound: 0.0212440
time: 21.62 seconds

## BFS NS instance: NS_A2_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: 1.1743728, 1.4586787, 1.1749766, 1.4539745, -0.0996460, 0.1047792
1: -1.2647058, -0.5790836, -1.2575232, -0.5794203, -0.0963906, 0.0876429
2: -1.7612709, -0.9775800, -1.7608930, -0.9855399, -0.2839125, 0.2930399
3: -3.4594622, -2.2798657, -3.4568524, -2.2850556, -0.2154739, 0.2205146
4: -3.9962573, -2.9850273, -3.9961047, -2.9884312, -0.3742440, 0.3783593
5: -4.3774214, -3.1418409, -4.3759756, -3.1474950, -0.2885795, 0.2954812
6: -5.3942046, -3.6081290, -5.3926034, -3.6109331, -0.4683554, 0.4710600
7: -5.3008471, -4.0800486, -5.3006535, -4.0836763, -0.1534808, 0.1602703
8: 0.0455266, 0.6731493, 0.0480099, 0.6727688, -0.4119501, 0.4100630
9: -1.5407097, -0.7815002, -1.5294764, -0.7820482, -0.2597133, 0.2469313

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3302
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 2623
type: B, layer: 1, pos: 2623
type: A, layer: 1, pos: 2306
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3463
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3463
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 3047
type: B, layer: 1, pos: 3047
type: B, layer: 1, pos: 3071
type: A, layer: 1, pos: 3071
type: B, layer: 1, pos: 2066
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2628
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2320
type: B, layer: 1, pos: 2320
type: A, layer: 1, pos: 3340
type: B, layer: 1, pos: 3340
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 3226
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 2981
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 2323
type: B, layer: 1, pos: 2323
type: A, layer: 1, pos: 2981
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 3265
type: A, layer: 1, pos: 3265
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3291
type: A, layer: 1, pos: 3291
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 2389
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 3324
type: A, layer: 1, pos: 3324
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 3357
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 3285
type: A, layer: 1, pos: 3306
type: B, layer: 1, pos: 2581
type: A, layer: 1, pos: 2581
type: B, layer: 1, pos: 3306
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3228
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 329
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 3373
type: A, layer: 1, pos: 3494
type: B, layer: 1, pos: 329
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 3373
type: B, layer: 1, pos: 3494
type: A, layer: 1, pos: 2329

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3537

## Relational analysis of NS_A2_B1_A2_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215575, upper bound: 0.0212666
time: 3.04 seconds

## Relational analysis of NS_A2_B1_A2_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0215575, upper bound: 0.0212668
time: 3.84 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 26.61 + 1777.68 = 1804.29 seconds
