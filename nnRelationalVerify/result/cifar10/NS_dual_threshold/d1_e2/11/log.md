## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 11)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0128948922


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1869113, 0.1869113)
1: (-6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2422895, 0.2422895)
2: (-0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513033, 0.0513033)
3: (-1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0587083, 0.0587083)
4: (-0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0381045, 0.0381045)
5: (-0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0624915, 0.0624915)
6: (-4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1184303, 0.1184303)
7: (1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257999, 0.0257999)
8: (-6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1088091, 0.1088091)
9: (-5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1513630, 0.1513630)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.15 + 18.56 = 25.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0129078, upper bound: 0.0129076

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 278
type: B, layer: 1, pos: 278
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 3577
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2476
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2130
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: B, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 2271
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 278

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129078, upper bound: 0.0126452
time: 2.55 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129078, upper bound: 0.0129081
time: 7.05 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.69 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.69
Output dim: 7, lower bound: -0.0129078, upper bound: 0.0126452
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.69
Output dim: 7, lower bound: -0.0129078, upper bound: 0.0129081

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.6830840, -3.2785509, -3.6831346, -3.2785413, -0.1865904, 0.1866787
1: -6.4960670, -5.8339081, -6.4960690, -5.8339081, -0.2419111, 0.2419571
2: -0.4305590, -0.2741565, -0.4305590, -0.2741480, -0.0512264, 0.0512400
3: -1.0987203, -0.8155191, -1.0987202, -0.8155131, -0.0586752, 0.0586625
4: -0.6262993, -0.4625589, -0.6263303, -0.4625583, -0.0378781, 0.0379412
5: -0.0520846, 0.2255305, -0.0520847, 0.2256966, -0.0620153, 0.0618509
6: -4.1222105, -3.6151609, -4.1224060, -3.6151581, -0.1177303, 0.1179260
7: 1.2784514, 1.5137215, 1.2784510, 1.5139666, -0.0251786, 0.0249469
8: -6.2269735, -5.8316193, -6.2269869, -5.8316193, -0.1086072, 0.1086231
9: -5.3906507, -4.9294538, -5.3906536, -4.9293590, -0.1511141, 0.1510192

Time for backsubstitution: 5.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 3577
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 278
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: B, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3577

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128786, upper bound: 0.0126433
time: 3.13 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129056, upper bound: 0.0126430
time: 2.64 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.6824484, -3.2779112, -3.6826496, -3.2785201, -0.1865240, 0.1877635
1: -6.4944334, -5.8339214, -6.4947591, -5.8339081, -0.2419240, 0.2430190
2: -0.4305622, -0.2741420, -0.4305590, -0.2741938, -0.0511109, 0.0515218
3: -1.0987765, -0.8156479, -1.0987206, -0.8156145, -0.0587882, 0.0586576
4: -0.6261969, -0.4620836, -0.6262191, -0.4625567, -0.0379135, 0.0389182
5: -0.0544764, 0.2255154, -0.0520844, 0.2256005, -0.0653156, 0.0619835
6: -4.1228738, -3.6126392, -4.1228824, -3.6151512, -0.1179279, 0.1209570
7: 1.2752513, 1.5145922, 1.2784497, 1.5145928, -0.0284859, 0.0251534
8: -6.2262459, -5.8314676, -6.2263761, -5.8316193, -0.1086485, 0.1093761
9: -5.3918648, -4.9291301, -5.3906598, -4.9291263, -0.1525834, 0.1511785

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3577
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 278
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3577

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129056, upper bound: 0.0128789
time: 31.39 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129056, upper bound: 0.0129063
time: 2.27 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 39.22 seconds
NS_A1_A1, status: Status.VERIFIED, split count: 2, time: 39.22
Output dim: 7, lower bound: -0.0128786, upper bound: 0.0126433
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 39.22
Output dim: 7, lower bound: -0.0129056, upper bound: 0.0126430
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 39.22
Output dim: 7, lower bound: -0.0129056, upper bound: 0.0128789
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 39.22
Output dim: 7, lower bound: -0.0129056, upper bound: 0.0129063

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -3.6831388, -3.2785590, -3.6831346, -3.2785485, -0.1866596, 0.1866280
1: -6.4962287, -5.8339348, -6.4960690, -5.8339300, -0.2421269, 0.2418016
2: -0.4305587, -0.2741537, -0.4305584, -0.2741479, -0.0512237, 0.0512440
3: -1.0987746, -0.8155296, -1.0987202, -0.8155223, -0.0587593, 0.0586017
4: -0.6263080, -0.4625589, -0.6263303, -0.4625582, -0.0378858, 0.0379356
5: -0.0521281, 0.2255260, -0.0520847, 0.2256924, -0.0620736, 0.0618091
6: -4.1223850, -3.6151743, -4.1224060, -3.6151702, -0.1179349, 0.1177801
7: 1.2784764, 1.5140288, 1.2784715, 1.5139666, -0.0249259, 0.0253020
8: -6.2269740, -5.8315878, -6.2269869, -5.8316193, -0.1085854, 0.1086540
9: -5.3908386, -4.9294839, -5.3906527, -4.9293833, -0.1513590, 0.1508448

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 278
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: B, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A1_A2_A1

### Relational analysis result of NS_A1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0127767, upper bound: 0.0126406
time: 12.27 seconds

## Relational analysis of NS_A1_A2_A2

### Relational analysis result of NS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0126407
time: 4.02 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.6824484, -3.2779515, -3.6826496, -3.2785678, -0.1864687, 0.1877177
1: -6.4944329, -5.8340421, -6.4947586, -5.8340492, -0.2417548, 0.2428808
2: -0.4305601, -0.2741420, -0.4305566, -0.2741938, -0.0511078, 0.0515181
3: -1.0987765, -0.8156891, -1.0987206, -0.8156613, -0.0587220, 0.0586035
4: -0.6261969, -0.4620891, -0.6262191, -0.4625634, -0.0379070, 0.0389128
5: -0.0544764, 0.2254845, -0.0520844, 0.2255642, -0.0652697, 0.0619461
6: -4.1228738, -3.6127577, -4.1228824, -3.6152971, -0.1177660, 0.1208248
7: 1.2754618, 1.5145922, 1.2787050, 1.5145928, -0.0282371, 0.0248738
8: -6.2262259, -5.8314676, -6.2263508, -5.8316193, -0.1086278, 0.1093509
9: -5.3918643, -4.9292693, -5.3906593, -4.9292917, -0.1523927, 0.1510209

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 278
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0127766, upper bound: 0.0128759
time: 20.29 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129030, upper bound: 0.0128760
time: 2.66 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.6824484, -3.2779181, -3.6827047, -3.2785287, -0.1864734, 0.1878327
1: -6.4944329, -5.8339448, -6.4949203, -5.8339348, -0.2417684, 0.2432348
2: -0.4305616, -0.2741420, -0.4305587, -0.2741909, -0.0511149, 0.0515191
3: -1.0987765, -0.8156569, -1.0987747, -0.8156249, -0.0587275, 0.0587416
4: -0.6261969, -0.4620836, -0.6262276, -0.4625567, -0.0379078, 0.0389259
5: -0.0544764, 0.2255114, -0.0521282, 0.2255958, -0.0652738, 0.0620418
6: -4.1228738, -3.6126504, -4.1230574, -3.6151657, -0.1177820, 0.1211615
7: 1.2752719, 1.5145922, 1.2784746, 1.5149000, -0.0284859, 0.0249007
8: -6.2262459, -5.8314676, -6.2263756, -5.8315878, -0.1086793, 0.1093542
9: -5.3918643, -4.9291549, -5.3908486, -4.9291563, -0.1524090, 0.1514235

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 278
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0127768, upper bound: 0.0129031
time: 2.47 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0129031
time: 2.53 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 10.55 seconds
NS_A1_A2_A1, status: Status.VERIFIED, split count: 3, time: 10.55
Output dim: 7, lower bound: -0.0127767, upper bound: 0.0126406
NS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 10.55
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0126407
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 10.55
Output dim: 7, lower bound: -0.0127766, upper bound: 0.0128759
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 10.55
Output dim: 7, lower bound: -0.0129030, upper bound: 0.0128760
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 10.55
Output dim: 7, lower bound: -0.0127768, upper bound: 0.0129031
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 10.55
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0129031

## BFS NS instance: NS_A1_A2_A2

### Backsubstitution after applying NS history:
0: -3.6830924, -3.2765906, -3.6831079, -3.2785494, -0.1865232, 0.1897938
1: -6.4964404, -5.8323088, -6.4960647, -5.8339319, -0.2427684, 0.2427328
2: -0.4305341, -0.2730346, -0.4305443, -0.2741480, -0.0510461, 0.0533574
3: -1.1017313, -0.8155636, -1.0987202, -0.8155440, -0.0617566, 0.0585209
4: -0.6262815, -0.4604609, -0.6263151, -0.4625584, -0.0378559, 0.0401057
5: -0.0547003, 0.2255260, -0.0520846, 0.2256924, -0.0646468, 0.0617419
6: -4.1223841, -3.6147835, -4.1224055, -3.6151810, -0.1176436, 0.1192813
7: 1.2784153, 1.5163507, 1.2785009, 1.5139666, -0.0252998, 0.0272091
8: -6.2284932, -5.8317080, -6.2269869, -5.8317323, -0.1102728, 0.1085948
9: -5.3920269, -4.9295945, -5.3906536, -4.9294748, -0.1546652, 0.1503654

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 278
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 3349
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 278

## Relational analysis of NS_A1_A2_A2_B1

### Relational analysis result of NS_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126994, upper bound: 0.0126410
time: 2.52 seconds

## Relational analysis of NS_A1_A2_A2_B2

### Relational analysis result of NS_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126994, upper bound: 0.0126398
time: 239.20 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.6824017, -3.2759752, -3.6826234, -3.2785697, -0.1863325, 0.1908857
1: -6.4946442, -5.8324170, -6.4947538, -5.8340511, -0.2423971, 0.2438117
2: -0.4305355, -0.2730252, -0.4305423, -0.2741938, -0.0509302, 0.0536318
3: -1.1017330, -0.8157228, -1.0987206, -0.8156832, -0.0617192, 0.0585226
4: -0.6261706, -0.4599889, -0.6262040, -0.4625634, -0.0378772, 0.0410833
5: -0.0570486, 0.2254846, -0.0520847, 0.2255642, -0.0678430, 0.0618789
6: -4.1228733, -3.6123657, -4.1228814, -3.6153066, -0.1174747, 0.1223261
7: 1.2754004, 1.5169141, 1.2787342, 1.5145928, -0.0283257, 0.0267811
8: -6.2277441, -5.8315878, -6.2263508, -5.8317318, -0.1103155, 0.1092917
9: -5.3930511, -4.9293795, -5.3906593, -4.9293838, -0.1557001, 0.1505415

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 278
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2403

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0128209
time: 6.27 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0128749
time: 2.53 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.6822345, -3.2780082, -3.6826379, -3.2785568, -0.1862058, 0.1877372
1: -6.4942780, -5.8339443, -6.4948702, -5.8339353, -0.2416310, 0.2431409
2: -0.4304521, -0.2741790, -0.4305239, -0.2742024, -0.0509380, 0.0514501
3: -1.0987756, -0.8158957, -1.0987743, -0.8156987, -0.0586523, 0.0584997
4: -0.6260233, -0.4620897, -0.6261739, -0.4625586, -0.0377316, 0.0388703
5: -0.0544760, 0.2253040, -0.0521280, 0.2255315, -0.0652095, 0.0618343
6: -4.1228385, -3.6126914, -4.1230454, -3.6151786, -0.1176416, 0.1210718
7: 1.2754409, 1.5145922, 1.2785374, 1.5149000, -0.0282511, 0.0248237
8: -6.2262411, -5.8315582, -6.2263746, -5.8316202, -0.1086351, 0.1092225
9: -5.3918343, -4.9292421, -5.3908396, -4.9291854, -0.1522412, 0.1511217

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 278
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2403

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0127213, upper bound: 0.0129017
time: 2.53 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0127756, upper bound: 0.0129019
time: 2.59 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.6824017, -3.2759418, -3.6826782, -3.2785304, -0.1863372, 0.1910005
1: -6.4946442, -5.8323183, -6.4949160, -5.8339376, -0.2424110, 0.2441657
2: -0.4305370, -0.2730252, -0.4305446, -0.2741909, -0.0509373, 0.0536328
3: -1.1017330, -0.8156906, -1.0987747, -0.8156466, -0.0617247, 0.0586607
4: -0.6261706, -0.4599835, -0.6262127, -0.4625567, -0.0378780, 0.0410963
5: -0.0570486, 0.2255114, -0.0521284, 0.2255958, -0.0678471, 0.0619746
6: -4.1228733, -3.6122594, -4.1230564, -3.6151762, -0.1174908, 0.1226629
7: 1.2752106, 1.5169141, 1.2785038, 1.5149000, -0.0285744, 0.0268080
8: -6.2277641, -5.8315878, -6.2263756, -5.8317013, -0.1103671, 0.1092949
9: -5.3930511, -4.9292650, -5.3908486, -4.9292479, -0.1557164, 0.1509442

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 278
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2403

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129018, upper bound: 0.0128474
time: 2.50 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0129018
time: 2.65 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 10.70 seconds
NS_A1_A2_A2_B1, status: Status.VERIFIED, split count: 4, time: 10.70
Output dim: 7, lower bound: -0.0126994, upper bound: 0.0126410
NS_A1_A2_A2_B2, status: Status.VERIFIED, split count: 4, time: 10.70
Output dim: 7, lower bound: -0.0126994, upper bound: 0.0126398
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.70
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0128209
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.70
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0128749
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 10.70
Output dim: 7, lower bound: -0.0127213, upper bound: 0.0129017
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 10.70
Output dim: 7, lower bound: -0.0127756, upper bound: 0.0129019
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.70
Output dim: 7, lower bound: -0.0129018, upper bound: 0.0128474
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.70
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0129018

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.6824017, -3.2761164, -3.6826234, -3.2787442, -0.1860873, 0.1906803
1: -6.4946432, -5.8326759, -6.4947543, -5.8343701, -0.2419956, 0.2434821
2: -0.4304925, -0.2730254, -0.4304897, -0.2741940, -0.0508797, 0.0535702
3: -1.1017331, -0.8157636, -1.0987204, -0.8157334, -0.0616612, 0.0584756
4: -0.6261470, -0.4599890, -0.6261752, -0.4625636, -0.0378511, 0.0410512
5: -0.0570444, 0.2254677, -0.0520792, 0.2255437, -0.0678112, 0.0618490
6: -4.1228380, -3.6123824, -4.1228380, -3.6153262, -0.1174075, 0.1222463
7: 1.2755952, 1.5169141, 1.2789731, 1.5145928, -0.0280747, 0.0264971
8: -6.2277436, -5.8321977, -6.2263513, -5.8324862, -0.1092515, 0.1084228
9: -5.3930516, -4.9298573, -5.3906598, -4.9299641, -0.1551560, 0.1500965

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 278
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 278

## Relational analysis of NS_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126392, upper bound: 0.0128211
time: 26.34 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126392, upper bound: 0.0128211
time: 2.57 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.6824017, -3.2760851, -3.6826198, -3.2786522, -0.1863042, 0.1908588
1: -6.4946446, -5.8324690, -6.4948153, -5.8340569, -0.2423620, 0.2439208
2: -0.4305153, -0.2730257, -0.4305257, -0.2741390, -0.0509884, 0.0535899
3: -1.1017331, -0.8157508, -1.0987761, -0.8157116, -0.0616745, 0.0585732
4: -0.6261622, -0.4599892, -0.6261961, -0.4625283, -0.0379153, 0.0410588
5: -0.0570487, 0.2254757, -0.0521102, 0.2255535, -0.0678241, 0.0618913
6: -4.1228604, -3.6123776, -4.1228986, -3.6153202, -0.1174669, 0.1223120
7: 1.2754114, 1.5169141, 1.2787118, 1.5146861, -0.0283246, 0.0267262
8: -6.2277436, -5.8317480, -6.2263985, -5.8318386, -0.1101396, 0.1096057
9: -5.3930507, -4.9294100, -5.3907518, -4.9293485, -0.1557315, 0.1507024

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 278
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 278

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126390, upper bound: 0.0128749
time: 2.92 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126390, upper bound: 0.0128751
time: 6.19 seconds

## BFS NS instance: NS_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -3.6822345, -3.2781825, -3.6826379, -3.2786973, -0.1859992, 0.1874911
1: -6.4942775, -5.8342619, -6.4948688, -5.8341966, -0.2413037, 0.2427399
2: -0.4303994, -0.2741792, -0.4304810, -0.2742024, -0.0508763, 0.0513996
3: -1.0987757, -0.8159461, -1.0987743, -0.8157395, -0.0586052, 0.0584418
4: -0.6259944, -0.4620900, -0.6261503, -0.4625589, -0.0376995, 0.0388441
5: -0.0544709, 0.2252831, -0.0521239, 0.2255152, -0.0651796, 0.0618024
6: -4.1227942, -3.6127110, -4.1230097, -3.6151950, -0.1175614, 0.1210046
7: 1.2756809, 1.5145922, 1.2787322, 1.5149000, -0.0279530, 0.0245868
8: -6.2262402, -5.8323150, -6.2263746, -5.8322330, -0.1077658, 0.1081528
9: -5.3918343, -4.9298258, -5.3908391, -4.9296656, -0.1517964, 0.1505778

Time for backsubstitution: 5.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 278
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 278

## Relational analysis of NS_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124587, upper bound: 0.0129017
time: 17.67 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124587, upper bound: 0.0129014
time: 389.38 seconds

## BFS NS instance: NS_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -3.6822309, -3.2780912, -3.6826379, -3.2786663, -0.1861789, 0.1877080
1: -6.4943380, -5.8339496, -6.4948697, -5.8339882, -0.2417400, 0.2431066
2: -0.4304355, -0.2741244, -0.4305039, -0.2742026, -0.0508961, 0.0515083
3: -1.0988312, -0.8159241, -1.0987744, -0.8157270, -0.0587030, 0.0584551
4: -0.6260155, -0.4620548, -0.6261656, -0.4625590, -0.0377071, 0.0389084
5: -0.0545017, 0.2252930, -0.0521282, 0.2255229, -0.0652219, 0.0618152
6: -4.1228552, -3.6127052, -4.1230345, -3.6151900, -0.1176277, 0.1210642
7: 1.2754192, 1.5146855, 1.2785482, 1.5149000, -0.0281942, 0.0249043
8: -6.2262878, -5.8316679, -6.2263737, -5.8317814, -0.1089492, 0.1090422
9: -5.3919263, -4.9292097, -5.3908396, -4.9292173, -0.1524023, 0.1511536

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 278
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 278

## Relational analysis of NS_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125129, upper bound: 0.0129013
time: 15.21 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125129, upper bound: 0.0129016
time: 2.39 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.6824017, -3.2760828, -3.6826782, -3.2787049, -0.1860921, 0.1907952
1: -6.4946432, -5.8325777, -6.4949155, -5.8342552, -0.2420094, 0.2438360
2: -0.4304941, -0.2730254, -0.4304920, -0.2741911, -0.0508869, 0.0535712
3: -1.1017331, -0.8157315, -1.0987746, -0.8156972, -0.0616666, 0.0586136
4: -0.6261470, -0.4599833, -0.6261837, -0.4625569, -0.0378519, 0.0410643
5: -0.0570444, 0.2254946, -0.0521230, 0.2255752, -0.0678153, 0.0619447
6: -4.1228380, -3.6122761, -4.1230125, -3.6151962, -0.1174236, 0.1225830
7: 1.2754053, 1.5169141, 1.2787426, 1.5149000, -0.0283235, 0.0265240
8: -6.2277632, -5.8321977, -6.2263761, -5.8324537, -0.1093031, 0.1084261
9: -5.3930516, -4.9297423, -5.3908482, -4.9298277, -0.1551724, 0.1504993

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 278
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 278

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126391, upper bound: 0.0128478
time: 2.70 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126391, upper bound: 0.0128476
time: 20.51 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.6824017, -3.2760518, -3.6826749, -3.2786131, -0.1863090, 0.1909737
1: -6.4946446, -5.8323708, -6.4949765, -5.8339424, -0.2423757, 0.2442749
2: -0.4305168, -0.2730257, -0.4305279, -0.2741362, -0.0509955, 0.0535910
3: -1.1017331, -0.8157189, -1.0988302, -0.8156750, -0.0616799, 0.0587114
4: -0.6261622, -0.4599838, -0.6262047, -0.4625217, -0.0379161, 0.0410718
5: -0.0570487, 0.2255025, -0.0521537, 0.2255851, -0.0678281, 0.0619871
6: -4.1228604, -3.6122713, -4.1230731, -3.6151888, -0.1174830, 0.1226487
7: 1.2752215, 1.5169141, 1.2784812, 1.5149933, -0.0285734, 0.0267532
8: -6.2277641, -5.8317480, -6.2264228, -5.8318071, -0.1101912, 0.1096088
9: -5.3930507, -4.9292955, -5.3909407, -4.9292126, -0.1557479, 0.1511053

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 278
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 278

## Relational analysis of NS_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126391, upper bound: 0.0129018
time: 2.60 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126391, upper bound: 0.0129017
time: 2.39 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 10.53 seconds
NS_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 10.53
Output dim: 7, lower bound: -0.0126392, upper bound: 0.0128211
NS_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 10.53
Output dim: 7, lower bound: -0.0126392, upper bound: 0.0128211
NS_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 10.53
Output dim: 7, lower bound: -0.0126390, upper bound: 0.0128749
NS_A2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 10.53
Output dim: 7, lower bound: -0.0126390, upper bound: 0.0128751
NS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 10.53
Output dim: 7, lower bound: -0.0124587, upper bound: 0.0129017
NS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 10.53
Output dim: 7, lower bound: -0.0124587, upper bound: 0.0129014
NS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 10.53
Output dim: 7, lower bound: -0.0125129, upper bound: 0.0129013
NS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 10.53
Output dim: 7, lower bound: -0.0125129, upper bound: 0.0129016
NS_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 10.53
Output dim: 7, lower bound: -0.0126391, upper bound: 0.0128478
NS_A2_B2_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 10.53
Output dim: 7, lower bound: -0.0126391, upper bound: 0.0128476
NS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 10.53
Output dim: 7, lower bound: -0.0126391, upper bound: 0.0129018
NS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 10.53
Output dim: 7, lower bound: -0.0126391, upper bound: 0.0129017

## BFS NS instance: NS_A2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -3.6822345, -3.2781825, -3.6830721, -3.2787275, -0.1859307, 0.1875603
1: -6.4942775, -5.8342619, -6.4961777, -5.8341966, -0.2409208, 0.2429756
2: -0.4303994, -0.2741792, -0.4304810, -0.2741651, -0.0509860, 0.0511010
3: -1.0987757, -0.8159461, -1.0987740, -0.8156440, -0.0586245, 0.0584067
4: -0.6259944, -0.4620900, -0.6262306, -0.4625611, -0.0378793, 0.0386263
5: -0.0544709, 0.2252831, -0.0521237, 0.2254453, -0.0646747, 0.0621554
6: -4.1227942, -3.6127110, -4.1223378, -3.6152043, -0.1180333, 0.1203279
7: 1.2756809, 1.5145922, 1.2787341, 1.5140288, -0.0270753, 0.0251916
8: -6.2262402, -5.8323150, -6.2269721, -5.8322330, -0.1076238, 0.1082236
9: -5.3918343, -4.9298258, -5.3908296, -4.9299932, -0.1514658, 0.1507446

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2531

## Relational analysis of NS_A2_B2_A1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124543, upper bound: 0.0128857
time: 2.55 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124542, upper bound: 0.0128973
time: 2.48 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -3.6822345, -3.2781825, -3.6824365, -3.2780879, -0.1862772, 0.1865294
1: -6.4942775, -5.8342619, -6.4945450, -5.8342104, -0.2413033, 0.2416451
2: -0.4303994, -0.2741792, -0.4304843, -0.2741505, -0.0512855, 0.0513977
3: -1.0987757, -0.8159461, -1.0988305, -0.8157727, -0.0584800, 0.0584471
4: -0.6259944, -0.4620900, -0.6261281, -0.4620855, -0.0378272, 0.0379670
5: -0.0544709, 0.2252831, -0.0545158, 0.2254301, -0.0618489, 0.0618039
6: -4.1227942, -3.6127110, -4.1230021, -3.6126819, -0.1177354, 0.1181496
7: 1.2756809, 1.5145922, 1.2755334, 1.5148994, -0.0251112, 0.0246250
8: -6.2262402, -5.8323150, -6.2262440, -5.8320804, -0.1077657, 0.1074251
9: -5.3918343, -4.9298258, -5.3920431, -4.9296694, -0.1507742, 0.1509603

Time for backsubstitution: 5.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124587, upper bound: 0.0127760
time: 3.45 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124587, upper bound: 0.0129019
time: 2.59 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -3.6822309, -3.2780912, -3.6830721, -3.2786970, -0.1861103, 0.1877772
1: -6.4943380, -5.8339496, -6.4961767, -5.8339882, -0.2413573, 0.2433422
2: -0.4304355, -0.2741244, -0.4305039, -0.2741654, -0.0510058, 0.0512097
3: -1.0988312, -0.8159241, -1.0987741, -0.8156316, -0.0587223, 0.0584201
4: -0.6260155, -0.4620548, -0.6262457, -0.4625612, -0.0378869, 0.0386905
5: -0.0545017, 0.2252930, -0.0521280, 0.2254530, -0.0647170, 0.0621682
6: -4.1228552, -3.6127052, -4.1223621, -3.6151984, -0.1180996, 0.1203874
7: 1.2754192, 1.5146855, 1.2785500, 1.5140288, -0.0273164, 0.0255091
8: -6.2262878, -5.8316679, -6.2269726, -5.8317814, -0.1088071, 0.1091131
9: -5.3919263, -4.9292097, -5.3908286, -4.9295454, -0.1520718, 0.1513203

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2531

## Relational analysis of NS_A2_B2_A1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0125084, upper bound: 0.0128856
time: 2.49 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125084, upper bound: 0.0128971
time: 2.42 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3.6822309, -3.2780912, -3.6824365, -3.2780571, -0.1864567, 0.1867463
1: -6.4943380, -5.8339496, -6.4945450, -5.8340015, -0.2417400, 0.2420118
2: -0.4304355, -0.2741244, -0.4305070, -0.2741509, -0.0513053, 0.0515064
3: -1.0988312, -0.8159241, -1.0988307, -0.8157604, -0.0585777, 0.0584605
4: -0.6260155, -0.4620548, -0.6261435, -0.4620859, -0.0378348, 0.0380312
5: -0.0545017, 0.2252930, -0.0545202, 0.2254379, -0.0618913, 0.0618167
6: -4.1228552, -3.6127052, -4.1230254, -3.6126773, -0.1178016, 0.1182091
7: 1.2754192, 1.5146855, 1.2753494, 1.5148994, -0.0253410, 0.0249425
8: -6.2262878, -5.8316679, -6.2262440, -5.8316293, -0.1089490, 0.1083146
9: -5.3919263, -4.9292097, -5.3920436, -4.9292216, -0.1513801, 0.1515360

Time for backsubstitution: 5.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0125129, upper bound: 0.0127755
time: 2.50 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125129, upper bound: 0.0129017
time: 2.43 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -3.6824017, -3.2760518, -3.6831090, -3.2786438, -0.1862403, 0.1910430
1: -6.4946446, -5.8323708, -6.4962835, -5.8339424, -0.2419885, 0.2445099
2: -0.4305168, -0.2730257, -0.4305279, -0.2740990, -0.0511051, 0.0532889
3: -1.1017331, -0.8157189, -1.0988300, -0.8155799, -0.0616992, 0.0586763
4: -0.6261622, -0.4599838, -0.6262850, -0.4625238, -0.0380958, 0.0408540
5: -0.0570487, 0.2255025, -0.0521537, 0.2255151, -0.0673232, 0.0623400
6: -4.1228604, -3.6122713, -4.1224008, -3.6151974, -0.1179549, 0.1219720
7: 1.2752215, 1.5169141, 1.2784830, 1.5141222, -0.0276956, 0.0273577
8: -6.2277641, -5.8317480, -6.2270217, -5.8318071, -0.1100457, 0.1096797
9: -5.3930507, -4.9292955, -5.3909311, -4.9295406, -0.1554173, 0.1512721

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2531

## Relational analysis of NS_A2_B2_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0128853
time: 2.53 seconds

## Relational analysis of NS_A2_B2_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0128972
time: 2.63 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3.6824017, -3.2760518, -3.6824737, -3.2780042, -0.1865867, 0.1900084
1: -6.4946446, -5.8323708, -6.4946508, -5.8339562, -0.2423757, 0.2431792
2: -0.4305168, -0.2730257, -0.4305312, -0.2740843, -0.0514044, 0.0535891
3: -1.1017331, -0.8157189, -1.0988860, -0.8157087, -0.0615547, 0.0587168
4: -0.6261622, -0.4599838, -0.6261827, -0.4620485, -0.0380437, 0.0401947
5: -0.0570487, 0.2255025, -0.0545458, 0.2255000, -0.0644975, 0.0619885
6: -4.1228604, -3.6122713, -4.1230640, -3.6126766, -0.1176571, 0.1197937
7: 1.2752215, 1.5169141, 1.2752829, 1.5149928, -0.0260011, 0.0267905
8: -6.2277641, -5.8317480, -6.2262936, -5.8316555, -0.1101912, 0.1088812
9: -5.3930507, -4.9292955, -5.3921447, -4.9292164, -0.1547257, 0.1514876

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2531

## Relational analysis of NS_A2_B2_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0128857
time: 2.47 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0128978
time: 2.75 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 10.76 seconds
NS_A2_B2_A1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 10.76
Output dim: 7, lower bound: -0.0124543, upper bound: 0.0128857
NS_A2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 10.76
Output dim: 7, lower bound: -0.0124542, upper bound: 0.0128973
NS_A2_B2_A1_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 10.76
Output dim: 7, lower bound: -0.0124587, upper bound: 0.0127760
NS_A2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 10.76
Output dim: 7, lower bound: -0.0124587, upper bound: 0.0129019
NS_A2_B2_A1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 10.76
Output dim: 7, lower bound: -0.0125084, upper bound: 0.0128856
NS_A2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 10.76
Output dim: 7, lower bound: -0.0125084, upper bound: 0.0128971
NS_A2_B2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 10.76
Output dim: 7, lower bound: -0.0125129, upper bound: 0.0127755
NS_A2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 10.76
Output dim: 7, lower bound: -0.0125129, upper bound: 0.0129017
NS_A2_B2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 10.76
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0128853
NS_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 10.76
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0128972
NS_A2_B2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 10.76
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0128857
NS_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 10.76
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0128978

## BFS NS instance: NS_A2_B2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.6818957, -3.2777133, -3.6828003, -3.2787294, -0.1856439, 0.1881453
1: -6.4939446, -5.8337746, -6.4959097, -5.8341966, -0.2406025, 0.2434717
2: -0.4304278, -0.2742249, -0.4304810, -0.2742034, -0.0509901, 0.0510738
3: -1.0989115, -0.8160663, -1.0987741, -0.8157406, -0.0588387, 0.0582980
4: -0.6259948, -0.4620907, -0.6262306, -0.4625624, -0.0378760, 0.0386212
5: -0.0545999, 0.2251598, -0.0521238, 0.2253461, -0.0648869, 0.0620467
6: -4.1228919, -3.6128302, -4.1223378, -3.6153007, -0.1181168, 0.1202367
7: 1.2755713, 1.5145335, 1.2787343, 1.5139815, -0.0270737, 0.0251096
8: -6.2260675, -5.8319678, -6.2268333, -5.8322330, -0.1074333, 0.1085955
9: -5.3917198, -4.9297161, -5.3907380, -4.9299932, -0.1513868, 0.1507663

Time for backsubstitution: 5.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124541, upper bound: 0.0127710
time: 2.61 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124543, upper bound: 0.0128970
time: 2.51 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -3.6822345, -3.2781825, -3.6824567, -3.2760837, -0.1890585, 0.1865790
1: -6.4942775, -5.8342619, -6.4948063, -5.8325825, -0.2424709, 0.2416435
2: -0.4303994, -0.2741792, -0.4304945, -0.2730225, -0.0527393, 0.0513995
3: -1.0987757, -0.8159461, -1.1017872, -0.8157330, -0.0585546, 0.0614446
4: -0.6259944, -0.4620900, -0.6261554, -0.4599834, -0.0399981, 0.0380213
5: -0.0544709, 0.2252831, -0.0570879, 0.2254941, -0.0619130, 0.0643773
6: -4.1227942, -3.6127110, -4.1230121, -3.6122770, -0.1182585, 0.1181006
7: 1.2756809, 1.5145922, 1.2754095, 1.5172213, -0.0271489, 0.0246713
8: -6.2262402, -5.8323150, -6.2277637, -5.8321662, -0.1077918, 0.1091254
9: -5.3918343, -4.9298258, -5.3932400, -4.9297476, -0.1507134, 0.1524526

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2476
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 3150
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2130
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 3448
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 3001
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 3006
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2038
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2531

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124426, upper bound: 0.0128977
time: 2.66 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124542, upper bound: 0.0128974
time: 11.32 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.6818922, -3.2776220, -3.6828003, -3.2786982, -0.1858234, 0.1883621
1: -6.4940052, -5.8334627, -6.4959097, -5.8339882, -0.2410390, 0.2438385
2: -0.4304638, -0.2741700, -0.4305039, -0.2742036, -0.0510099, 0.0511825
3: -1.0989671, -0.8160443, -1.0987742, -0.8157281, -0.0589364, 0.0583113
4: -0.6260160, -0.4620556, -0.6262457, -0.4625624, -0.0378836, 0.0386854
5: -0.0546305, 0.2251695, -0.0521277, 0.2253540, -0.0649292, 0.0620595
6: -4.1229525, -3.6128242, -4.1223621, -3.6152959, -0.1181831, 0.1202962
7: 1.2753099, 1.5146269, 1.2785503, 1.5139815, -0.0273149, 0.0254271
8: -6.2261152, -5.8313203, -6.2268333, -5.8317814, -0.1086167, 0.1094848
9: -5.3918133, -4.9291000, -5.3907375, -4.9295454, -0.1519927, 0.1513419

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0125084, upper bound: 0.0127708
time: 2.71 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125085, upper bound: 0.0128971
time: 2.59 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3.6822309, -3.2780912, -3.6824567, -3.2760532, -0.1892371, 0.1867959
1: -6.4943380, -5.8339491, -6.4948053, -5.8323755, -0.2429096, 0.2420100
2: -0.4304355, -0.2741244, -0.4305174, -0.2730229, -0.0527590, 0.0515081
3: -1.0988312, -0.8159239, -1.1017871, -0.8157204, -0.0586523, 0.0614579
4: -0.6260155, -0.4620548, -0.6261708, -0.4599838, -0.0400056, 0.0380855
5: -0.0545018, 0.2252930, -0.0570922, 0.2255020, -0.0619554, 0.0643902
6: -4.1228552, -3.6127052, -4.1230354, -3.6122723, -0.1183243, 0.1181601
7: 1.2754194, 1.5146855, 1.2752256, 1.5172213, -0.0273788, 0.0249877
8: -6.2262878, -5.8316679, -6.2277637, -5.8317165, -0.1089730, 0.1100149
9: -5.3919263, -4.9292097, -5.3932400, -4.9293017, -0.1513194, 0.1530287

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2476
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 3150
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2130
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 3448
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 3001
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 3006
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2038
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2531

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124968, upper bound: 0.0128979
time: 3.05 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125084, upper bound: 0.0128973
time: 2.66 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -3.6820629, -3.2755837, -3.6828368, -3.2786455, -0.1859536, 0.1916263
1: -6.4943123, -5.8318830, -6.4960165, -5.8339424, -0.2416711, 0.2450066
2: -0.4305452, -0.2730713, -0.4305279, -0.2741373, -0.0511092, 0.0532617
3: -1.1018691, -0.8158392, -1.0988299, -0.8156763, -0.0619134, 0.0585676
4: -0.6261625, -0.4599842, -0.6262850, -0.4625254, -0.0380926, 0.0408489
5: -0.0571774, 0.2253789, -0.0521537, 0.2254161, -0.0675354, 0.0622314
6: -4.1229587, -3.6123900, -4.1224008, -3.6152954, -0.1180384, 0.1218817
7: 1.2751122, 1.5168555, 1.2784833, 1.5140749, -0.0276941, 0.0272758
8: -6.2275915, -5.8314009, -6.2268825, -5.8318071, -0.1098551, 0.1100517
9: -5.3929377, -4.9291863, -5.3908391, -4.9295406, -0.1553383, 0.1512936

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0127707
time: 2.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0127709
time: 2.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -3.6820629, -3.2755837, -3.6822009, -3.2780058, -0.1863004, 0.1905919
1: -6.4943123, -5.8318830, -6.4943848, -5.8339562, -0.2420580, 0.2436755
2: -0.4305452, -0.2730713, -0.4305312, -0.2741227, -0.0514085, 0.0535619
3: -1.1018691, -0.8158392, -1.0988861, -0.8158052, -0.0617689, 0.0586080
4: -0.6261625, -0.4599842, -0.6261827, -0.4620501, -0.0380405, 0.0401896
5: -0.0571774, 0.2253789, -0.0545458, 0.2254008, -0.0647097, 0.0618798
6: -4.1229587, -3.6123900, -4.1230640, -3.6127739, -0.1177406, 0.1197034
7: 1.2751122, 1.5168555, 1.2752831, 1.5149454, -0.0261712, 0.0267085
8: -6.2275915, -5.8314009, -6.2261543, -5.8316555, -0.1100006, 0.1092530
9: -5.3929377, -4.9291863, -5.3920527, -4.9292164, -0.1546467, 0.1515092

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2271

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0127712
time: 8.32 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126347, upper bound: 0.0127711
time: 2.41 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 16.26 seconds
NS_A2_B2_A1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 16.26
Output dim: 7, lower bound: -0.0124541, upper bound: 0.0127710
NS_A2_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 16.26
Output dim: 7, lower bound: -0.0124543, upper bound: 0.0128970
NS_A2_B2_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 16.26
Output dim: 7, lower bound: -0.0124426, upper bound: 0.0128977
NS_A2_B2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 16.26
Output dim: 7, lower bound: -0.0124542, upper bound: 0.0128974
NS_A2_B2_A1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 16.26
Output dim: 7, lower bound: -0.0125084, upper bound: 0.0127708
NS_A2_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 16.26
Output dim: 7, lower bound: -0.0125085, upper bound: 0.0128971
NS_A2_B2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 16.26
Output dim: 7, lower bound: -0.0124968, upper bound: 0.0128979
NS_A2_B2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 16.26
Output dim: 7, lower bound: -0.0125084, upper bound: 0.0128973
NS_A2_B2_A2_B2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 16.26
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0127707
NS_A2_B2_A2_B2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 16.26
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0127709
NS_A2_B2_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 16.26
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0127712
NS_A2_B2_A2_B2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 16.26
Output dim: 7, lower bound: -0.0126347, upper bound: 0.0127711

## BFS NS instance: NS_A2_B2_A1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.6818957, -3.2777133, -3.6828203, -3.2767322, -0.1884273, 0.1881949
1: -6.4939446, -5.8337746, -6.4961734, -5.8325682, -0.2417700, 0.2434715
2: -0.4304278, -0.2742249, -0.4304912, -0.2730730, -0.0524437, 0.0510756
3: -1.0989115, -0.8160663, -1.1017314, -0.8157007, -0.0589133, 0.0612953
4: -0.6259948, -0.4620907, -0.6262580, -0.4604623, -0.0400466, 0.0386755
5: -0.0545999, 0.2251598, -0.0546957, 0.2254102, -0.0649510, 0.0646201
6: -4.1228919, -3.6128302, -4.1223478, -3.6148973, -0.1186415, 0.1201877
7: 1.2755713, 1.5145335, 1.2786102, 1.5163034, -0.0270531, 0.0251571
8: -6.2260675, -5.8319678, -6.2283554, -5.8323183, -0.1074595, 0.1102949
9: -5.3917198, -4.9297161, -5.3919353, -4.9300718, -0.1513261, 0.1522568

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 2271
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2164

## Relational analysis of NS_A2_B2_A1_A1_B1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124506, upper bound: 0.0128876
time: 2.62 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124506, upper bound: 0.0128963
time: 2.59 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -3.6820087, -3.2781832, -3.6821725, -3.2760856, -0.1888122, 0.1862793
1: -6.4940376, -5.8342619, -6.4945011, -5.8325825, -0.2421970, 0.2412997
2: -0.4303994, -0.2741953, -0.4304945, -0.2730429, -0.0527156, 0.0513807
3: -1.0987756, -0.8160210, -1.1017874, -0.8158278, -0.0584403, 0.0613539
4: -0.6259944, -0.4620907, -0.6261554, -0.4599842, -0.0399946, 0.0380179
5: -0.0544708, 0.2252110, -0.0570880, 0.2254038, -0.0617979, 0.0642862
6: -4.1227942, -3.6127768, -4.1230121, -3.6123586, -0.1181634, 0.1180244
7: 1.2756809, 1.5145390, 1.2754097, 1.5171540, -0.0270638, 0.0246035
8: -6.2260742, -5.8323150, -6.2275534, -5.8321662, -0.1076235, 0.1089122
9: -5.3917704, -4.9298258, -5.3931580, -4.9297476, -0.1506508, 0.1523735

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2476
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 3150
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2130
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 3448
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 3001
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 3006
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2038
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2164

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124368, upper bound: 0.0128967
time: 2.49 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124391, upper bound: 0.0128964
time: 8.71 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -3.6819625, -3.2781844, -3.6821179, -3.2756162, -0.1896420, 0.1862928
1: -6.4940100, -5.8342619, -6.4944730, -5.8320947, -0.2429670, 0.2413257
2: -0.4303994, -0.2742175, -0.4305228, -0.2730680, -0.0527121, 0.0514035
3: -1.0987756, -0.8160427, -1.1019235, -0.8158531, -0.0584458, 0.0616587
4: -0.6259944, -0.4620916, -0.6261559, -0.4599841, -0.0399930, 0.0380180
5: -0.0544709, 0.2251841, -0.0572170, 0.2253706, -0.0618042, 0.0645896
6: -4.1227942, -3.6128085, -4.1231089, -3.6123977, -0.1181685, 0.1181841
7: 1.2756810, 1.5145448, 1.2753000, 1.5171628, -0.0270669, 0.0248415
8: -6.2261014, -5.8323150, -6.2275915, -5.8318191, -0.1081636, 0.1089348
9: -5.3917427, -4.9298258, -5.3931270, -4.9296384, -0.1507348, 0.1523737

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2476
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 3150
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2130
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 3448
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 3001
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3039
type: A, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 3006
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2038
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2164

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124485, upper bound: 0.0128968
time: 2.35 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124505, upper bound: 0.0128964
time: 2.53 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.6818922, -3.2776220, -3.6828203, -3.2767019, -0.1886058, 0.1884118
1: -6.4940052, -5.8334618, -6.4961734, -5.8323617, -0.2422086, 0.2438385
2: -0.4304638, -0.2741700, -0.4305141, -0.2730732, -0.0524634, 0.0511842
3: -1.0989671, -0.8160442, -1.1017313, -0.8156884, -0.0590110, 0.0613087
4: -0.6260161, -0.4620556, -0.6262732, -0.4604626, -0.0400541, 0.0387397
5: -0.0546306, 0.2251695, -0.0547000, 0.2254180, -0.0649934, 0.0646329
6: -4.1229520, -3.6128242, -4.1223717, -3.6148920, -0.1187073, 0.1202472
7: 1.2753099, 1.5146269, 1.2784262, 1.5163034, -0.0272943, 0.0254735
8: -6.2261152, -5.8313203, -6.2283554, -5.8318691, -0.1086407, 0.1111845
9: -5.3918133, -4.9291000, -5.3919344, -4.9296255, -0.1519320, 0.1528330

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 2271
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2164

## Relational analysis of NS_A2_B2_A1_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0125075, upper bound: 0.0128883
time: 2.52 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125075, upper bound: 0.0128963
time: 2.66 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -3.6820047, -3.2780921, -3.6821725, -3.2760546, -0.1889906, 0.1864961
1: -6.4940982, -5.8339491, -6.4945011, -5.8323755, -0.2426356, 0.2416666
2: -0.4304355, -0.2741406, -0.4305174, -0.2730431, -0.0527354, 0.0514893
3: -1.0988314, -0.8159990, -1.1017872, -0.8158157, -0.0585380, 0.0613672
4: -0.6260155, -0.4620556, -0.6261708, -0.4599846, -0.0400022, 0.0380821
5: -0.0545017, 0.2252208, -0.0570922, 0.2254114, -0.0618403, 0.0642990
6: -4.1228552, -3.6127701, -4.1230354, -3.6123538, -0.1182292, 0.1180839
7: 1.2754195, 1.5146325, 1.2752258, 1.5171540, -0.0272936, 0.0249199
8: -6.2261214, -5.8316679, -6.2275529, -5.8317165, -0.1088048, 0.1098017
9: -5.3918629, -4.9292097, -5.3931580, -4.9293017, -0.1512568, 0.1529495

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2476
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 3150
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2130
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 3448
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 3001
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 3006
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2038
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2164

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124874, upper bound: 0.0127699
time: 12.36 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124959, upper bound: 0.0128959
time: 4.20 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -3.6819587, -3.2780929, -3.6821179, -3.2755852, -0.1898204, 0.1865095
1: -6.4940705, -5.8339491, -6.4944725, -5.8318882, -0.2434056, 0.2416924
2: -0.4304355, -0.2741627, -0.4305457, -0.2730684, -0.0527318, 0.0515122
3: -1.0988312, -0.8160205, -1.1019233, -0.8158408, -0.0585435, 0.0616721
4: -0.6260155, -0.4620564, -0.6261712, -0.4599844, -0.0400005, 0.0380822
5: -0.0545018, 0.2251937, -0.0572211, 0.2253784, -0.0618466, 0.0646024
6: -4.1228552, -3.6128025, -4.1231322, -3.6123927, -0.1182342, 0.1182436
7: 1.2754195, 1.5146383, 1.2751162, 1.5171628, -0.0272967, 0.0251579
8: -6.2261486, -5.8316679, -6.2275920, -5.8313694, -0.1093448, 0.1098243
9: -5.3918352, -4.9292097, -5.3931255, -4.9291925, -0.1513408, 0.1529498

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2476
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 3150
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2130
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 3448
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 3001
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3039
type: A, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 3006
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2038
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2164

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B2_A1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124988, upper bound: 0.0128962
time: 2.59 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B2_A2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125075, upper bound: 0.0128961
time: 2.60 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 10.74 seconds
NS_A2_B2_A1_A1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 10.74
Output dim: 7, lower bound: -0.0124506, upper bound: 0.0128876
NS_A2_B2_A1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 10.74
Output dim: 7, lower bound: -0.0124506, upper bound: 0.0128963
NS_A2_B2_A1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 10.74
Output dim: 7, lower bound: -0.0124368, upper bound: 0.0128967
NS_A2_B2_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 10.74
Output dim: 7, lower bound: -0.0124391, upper bound: 0.0128964
NS_A2_B2_A1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 10.74
Output dim: 7, lower bound: -0.0124485, upper bound: 0.0128968
NS_A2_B2_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 10.74
Output dim: 7, lower bound: -0.0124505, upper bound: 0.0128964
NS_A2_B2_A1_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 10.74
Output dim: 7, lower bound: -0.0125075, upper bound: 0.0128883
NS_A2_B2_A1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 10.74
Output dim: 7, lower bound: -0.0125075, upper bound: 0.0128963
NS_A2_B2_A1_A2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 10.74
Output dim: 7, lower bound: -0.0124874, upper bound: 0.0127699
NS_A2_B2_A1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 10.74
Output dim: 7, lower bound: -0.0124959, upper bound: 0.0128959
NS_A2_B2_A1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 10.74
Output dim: 7, lower bound: -0.0124988, upper bound: 0.0128962
NS_A2_B2_A1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 10.74
Output dim: 7, lower bound: -0.0125075, upper bound: 0.0128961

## BFS NS instance: NS_A2_B2_A1_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3.6818957, -3.2777762, -3.6828203, -3.2768130, -0.1840927, 0.1881908
1: -6.4939446, -5.8338389, -6.4961734, -5.8326511, -0.2342672, 0.2434651
2: -0.4304278, -0.2742289, -0.4304912, -0.2730781, -0.0522202, 0.0510751
3: -1.0988595, -0.8160663, -1.1016649, -0.8157007, -0.0589118, 0.0595211
4: -0.6259948, -0.4620926, -0.6262580, -0.4604650, -0.0400256, 0.0386752
5: -0.0545659, 0.2251598, -0.0546584, 0.2254102, -0.0649492, 0.0624767
6: -4.1228857, -3.6128302, -4.1223407, -3.6148977, -0.1186396, 0.1199740
7: 1.2755754, 1.5145335, 1.2786156, 1.5163034, -0.0269868, 0.0249586
8: -6.2260675, -5.8320942, -6.2283554, -5.8324800, -0.1031204, 0.1102912
9: -5.3917208, -4.9297590, -5.3919344, -4.9301257, -0.1467713, 0.1522501

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2516

## Relational analysis of NS_A2_B2_A1_A1_B1_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124369, upper bound: 0.0128939
time: 2.53 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124484, upper bound: 0.0128936
time: 2.47 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -3.6810539, -3.2809801, -3.6821725, -3.2783341, -0.1849062, 0.1831543
1: -6.4922428, -5.8391733, -6.4945016, -5.8366909, -0.2354239, 0.2358958
2: -0.4303470, -0.2743478, -0.4304945, -0.2731692, -0.0525103, 0.0512143
3: -1.0976396, -0.8163749, -1.1008772, -0.8158278, -0.0571669, 0.0597596
4: -0.6259956, -0.4621025, -0.6261554, -0.4599938, -0.0399745, 0.0380007
5: -0.0530661, 0.2247053, -0.0559613, 0.2254038, -0.0602556, 0.0623531
6: -4.1226516, -3.6128221, -4.1228933, -3.6123590, -0.1180056, 0.1178252
7: 1.2757854, 1.5145239, 1.2754967, 1.5171540, -0.0269216, 0.0244266
8: -6.2251978, -5.8351097, -6.2275534, -5.8344059, -0.1037202, 0.1057921
9: -5.3908587, -4.9326596, -5.3931575, -4.9321117, -0.1465374, 0.1490889

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2476
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 3150
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2130
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 3448
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 3001
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 3006
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2038
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2516

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124329, upper bound: 0.0128833
time: 2.80 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124330, upper bound: 0.0128942
time: 6.44 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -3.6820087, -3.2782643, -3.6821725, -3.2761481, -0.1888082, 0.1819448
1: -6.4940376, -5.8343458, -6.4945002, -5.8326473, -0.2421906, 0.2337984
2: -0.4303994, -0.2742005, -0.4304945, -0.2730470, -0.0527151, 0.0511583
3: -1.0987093, -0.8160210, -1.1017357, -0.8158278, -0.0566688, 0.0613524
4: -0.6259944, -0.4620931, -0.6261554, -0.4599862, -0.0399943, 0.0379971
5: -0.0544337, 0.2252110, -0.0570538, 0.2254038, -0.0596556, 0.0642844
6: -4.1227875, -3.6127768, -4.1230063, -3.6123588, -0.1179512, 0.1180227
7: 1.2756864, 1.5145390, 1.2754140, 1.5171540, -0.0268653, 0.0246033
8: -6.2260742, -5.8324771, -6.2275534, -5.8322926, -0.1076199, 0.1045732
9: -5.3917699, -4.9298811, -5.3931584, -4.9297905, -0.1506443, 0.1478218

Time for backsubstitution: 5.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2476
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 3150
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2130
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 3448
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 3001
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 3006
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2038
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2516

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124351, upper bound: 0.0128825
time: 2.51 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B1_A2_A2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124351, upper bound: 0.0128939
time: 2.58 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -3.6810079, -3.2809808, -3.6821179, -3.2778647, -0.1857362, 0.1831678
1: -6.4922132, -5.8391733, -6.4944720, -5.8362036, -0.2361938, 0.2359213
2: -0.4303470, -0.2743701, -0.4305228, -0.2731943, -0.0525067, 0.0512371
3: -1.0976399, -0.8163966, -1.1010132, -0.8158531, -0.0571725, 0.0600644
4: -0.6259956, -0.4621032, -0.6261559, -0.4599935, -0.0399728, 0.0380008
5: -0.0530661, 0.2246784, -0.0560901, 0.2253706, -0.0602619, 0.0626566
6: -4.1226516, -3.6128542, -4.1229897, -3.6123972, -0.1180107, 0.1179849
7: 1.2757856, 1.5145295, 1.2753873, 1.5171628, -0.0269247, 0.0246646
8: -6.2252254, -5.8351097, -6.2275915, -5.8340588, -0.1042603, 0.1058147
9: -5.3908319, -4.9326596, -5.3931265, -4.9320021, -0.1466213, 0.1490891

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2476
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 3150
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2130
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 3448
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 3001
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3039
type: A, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 3006
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2038
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2516

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124461, upper bound: 0.0128825
time: 2.67 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124460, upper bound: 0.0128938
time: 2.65 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -3.6819625, -3.2782650, -3.6821179, -3.2756793, -0.1896381, 0.1819583
1: -6.4940100, -5.8343458, -6.4944730, -5.8321600, -0.2429606, 0.2338244
2: -0.4303994, -0.2742228, -0.4305228, -0.2730722, -0.0527116, 0.0511811
3: -1.0987093, -0.8160427, -1.1018716, -0.8158531, -0.0566743, 0.0616572
4: -0.6259944, -0.4620938, -0.6261559, -0.4599861, -0.0399926, 0.0379973
5: -0.0544336, 0.2251841, -0.0571829, 0.2253706, -0.0596619, 0.0645878
6: -4.1227875, -3.6128087, -4.1231041, -3.6123972, -0.1179561, 0.1181822
7: 1.2756863, 1.5145448, 1.2753043, 1.5171628, -0.0268684, 0.0248412
8: -6.2261014, -5.8324771, -6.2275915, -5.8319454, -0.1081600, 0.1045957
9: -5.3917418, -4.9298811, -5.3931274, -4.9296808, -0.1507283, 0.1478220

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2476
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 3150
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2130
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 3448
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 3001
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3039
type: A, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 3006
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2038
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2516

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124483, upper bound: 0.0128826
time: 2.66 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124483, upper bound: 0.0128938
time: 2.69 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3.6818922, -3.2776847, -3.6828203, -3.2767820, -0.1842710, 0.1884078
1: -6.4940052, -5.8335266, -6.4961734, -5.8324442, -0.2347037, 0.2438319
2: -0.4304638, -0.2741750, -0.4305141, -0.2730784, -0.0522410, 0.0511828
3: -1.0989131, -0.8160442, -1.1016651, -0.8156884, -0.0590075, 0.0595353
4: -0.6260161, -0.4620575, -0.6262732, -0.4604650, -0.0400333, 0.0387378
5: -0.0545965, 0.2251695, -0.0546628, 0.2254180, -0.0649916, 0.0624890
6: -4.1229467, -3.6128244, -4.1223645, -3.6148920, -0.1187055, 0.1200331
7: 1.2753141, 1.5146269, 1.2784317, 1.5163034, -0.0272280, 0.0252751
8: -6.2261147, -5.8314466, -6.2283554, -5.8320303, -0.1043003, 0.1111808
9: -5.3918133, -4.9291420, -5.3919344, -4.9296799, -0.1473600, 0.1528264

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3150
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2130
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 3448
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 3001
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2847
type: A, layer: 1, pos: 3006
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2516

## Relational analysis of NS_A2_B2_A1_A2_B1_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124938, upper bound: 0.0128938
time: 2.67 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0125052, upper bound: 0.0128937
time: 2.62 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -3.6820047, -3.2781723, -3.6821725, -3.2761176, -0.1889866, 0.1821616
1: -6.4940982, -5.8340330, -6.4945011, -5.8324399, -0.2426292, 0.2341658
2: -0.4304355, -0.2741468, -0.4305174, -0.2730472, -0.0527348, 0.0512960
3: -1.0987623, -0.8159990, -1.1017357, -0.8158157, -0.0568301, 0.0613657
4: -0.6260155, -0.4620581, -0.6261708, -0.4599862, -0.0400019, 0.0380704
5: -0.0544643, 0.2252208, -0.0570580, 0.2254114, -0.0597226, 0.0642972
6: -4.1228476, -3.6127703, -4.1230302, -3.6123540, -0.1180261, 0.1180826
7: 1.2754250, 1.5146325, 1.2752302, 1.5171540, -0.0270962, 0.0249197
8: -6.2261214, -5.8318291, -6.2275529, -5.8318434, -0.1088009, 0.1054628
9: -5.3918619, -4.9292636, -5.3931584, -4.9293447, -0.1512506, 0.1483917

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2476
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 3150
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2130
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 3448
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 3001
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 3006
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2038
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 2271
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2516

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124919, upper bound: 0.0128824
time: 2.66 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B1_A2_A2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124919, upper bound: 0.0128944
time: 3.07 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -3.6810045, -3.2808900, -3.6821179, -3.2778335, -0.1859143, 0.1833845
1: -6.4922562, -5.8389106, -6.4944725, -5.8359966, -0.2366306, 0.2362871
2: -0.4303867, -0.2743127, -0.4305457, -0.2731947, -0.0525283, 0.0513615
3: -1.0977054, -0.8163614, -1.1010129, -0.8158408, -0.0573027, 0.0600799
4: -0.6260170, -0.4620680, -0.6261712, -0.4599937, -0.0399805, 0.0380693
5: -0.0531018, 0.2246938, -0.0560943, 0.2253784, -0.0603166, 0.0626694
6: -4.1227117, -3.6128464, -4.1230135, -3.6123934, -0.1180763, 0.1180444
7: 1.2755384, 1.5146132, 1.2752130, 1.5171628, -0.0271546, 0.0249810
8: -6.2252722, -5.8344626, -6.2275915, -5.8336091, -0.1054399, 0.1067044
9: -5.3908949, -4.9321241, -5.3931246, -4.9316325, -0.1472155, 0.1496556

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2476
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 3150
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2130
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 3448
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 3001
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3039
type: A, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 3006
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2038
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2516

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124966, upper bound: 0.0128823
time: 2.53 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124967, upper bound: 0.0128939
time: 16.32 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -3.6819587, -3.2781737, -3.6821179, -3.2756484, -0.1898165, 0.1821752
1: -6.4940701, -5.8340330, -6.4944739, -5.8319530, -0.2433991, 0.2341917
2: -0.4304355, -0.2741691, -0.4305457, -0.2730724, -0.0527313, 0.0513188
3: -1.0987623, -0.8160205, -1.1018715, -0.8158408, -0.0568357, 0.0616707
4: -0.6260155, -0.4620590, -0.6261712, -0.4599860, -0.0400002, 0.0380705
5: -0.0544643, 0.2251937, -0.0571871, 0.2253784, -0.0597290, 0.0646006
6: -4.1228476, -3.6128027, -4.1231265, -3.6123927, -0.1180311, 0.1182422
7: 1.2754247, 1.5146383, 1.2751205, 1.5171628, -0.0270993, 0.0251577
8: -6.2261486, -5.8318291, -6.2275915, -5.8314958, -0.1093411, 0.1054855
9: -5.3918347, -4.9292636, -5.3931260, -4.9292350, -0.1513346, 0.1483919

Time for backsubstitution: 5.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 3180
type: A, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 371
type: B, layer: 1, pos: 371
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2299
type: A, layer: 1, pos: 2299
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2476
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 3150
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3258
type: B, layer: 1, pos: 3258
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3573
type: A, layer: 1, pos: 3573
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2995
type: B, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2130
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 3448
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3221
type: A, layer: 1, pos: 3221
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: A, layer: 1, pos: 2089
type: B, layer: 1, pos: 3001
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3039
type: A, layer: 1, pos: 3039
type: B, layer: 1, pos: 2364
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2847
type: B, layer: 1, pos: 3006
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2038
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: A, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 269
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3372
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3599
type: B, layer: 1, pos: 269
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3372
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2516

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0125051, upper bound: 0.0128825
time: 2.68 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0125051, upper bound: 0.0128944
time: 2.55 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 10.82 seconds
NS_A2_B2_A1_A1_B1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0124369, upper bound: 0.0128939
NS_A2_B2_A1_A1_B1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0124484, upper bound: 0.0128936
NS_A2_B2_A1_A1_B2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0124329, upper bound: 0.0128833
NS_A2_B2_A1_A1_B2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0124330, upper bound: 0.0128942
NS_A2_B2_A1_A1_B2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0124351, upper bound: 0.0128825
NS_A2_B2_A1_A1_B2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0124351, upper bound: 0.0128939
NS_A2_B2_A1_A1_B2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0124461, upper bound: 0.0128825
NS_A2_B2_A1_A1_B2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0124460, upper bound: 0.0128938
NS_A2_B2_A1_A1_B2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0124483, upper bound: 0.0128826
NS_A2_B2_A1_A1_B2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0124483, upper bound: 0.0128938
NS_A2_B2_A1_A2_B1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0124938, upper bound: 0.0128938
NS_A2_B2_A1_A2_B1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0125052, upper bound: 0.0128937
NS_A2_B2_A1_A2_B2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0124919, upper bound: 0.0128824
NS_A2_B2_A1_A2_B2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0124919, upper bound: 0.0128944
NS_A2_B2_A1_A2_B2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0124966, upper bound: 0.0128823
NS_A2_B2_A1_A2_B2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0124967, upper bound: 0.0128939
NS_A2_B2_A1_A2_B2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0125051, upper bound: 0.0128825
NS_A2_B2_A1_A2_B2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 9, time: 10.82
Output dim: 7, lower bound: -0.0125051, upper bound: 0.0128944

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 25.71 + 1268.96 = 1294.67 seconds
