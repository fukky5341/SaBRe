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
execution time: IAR + RelationalAnalysis = 7.15 + 18.24 = 25.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0129078, upper bound: 0.0129076

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 278
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 278

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129078, upper bound: 0.0126452
time: 2.51 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129078, upper bound: 0.0129081
time: 6.98 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.55 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.55
Output dim: 7, lower bound: -0.0129078, upper bound: 0.0126452
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.55
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

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 278
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129052, upper bound: 0.0125162
time: 2.54 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129052, upper bound: 0.0126424
time: 2.50 seconds

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

Time for backsubstitution: 5.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 278
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 347

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129051, upper bound: 0.0127796
time: 4.97 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129051, upper bound: 0.0129053
time: 36.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 47.13 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 47.13
Output dim: 7, lower bound: -0.0129052, upper bound: 0.0125162
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 47.13
Output dim: 7, lower bound: -0.0129052, upper bound: 0.0126424
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 47.13
Output dim: 7, lower bound: -0.0129051, upper bound: 0.0127796
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 47.13
Output dim: 7, lower bound: -0.0129051, upper bound: 0.0129053

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3.6830173, -3.2785783, -3.6829205, -3.2786314, -0.1864946, 0.1864113
1: -6.4960155, -5.8339076, -6.4959116, -5.8339081, -0.2418170, 0.2418195
2: -0.4305240, -0.2741677, -0.4304495, -0.2741846, -0.0511574, 0.0510631
3: -1.0987196, -0.8155928, -1.0987195, -0.8157519, -0.0584335, 0.0585874
4: -0.6262456, -0.4625608, -0.6261564, -0.4625645, -0.0378223, 0.0377650
5: -0.0520843, 0.2254663, -0.0520843, 0.2254889, -0.0618076, 0.0617865
6: -4.1221995, -3.6151738, -4.1223702, -3.6151986, -0.1176407, 0.1177855
7: 1.2785143, 1.5137215, 1.2786201, 1.5139666, -0.0251016, 0.0247977
8: -6.2269721, -5.8316512, -6.2269821, -5.8317099, -0.1084756, 0.1085787
9: -5.3906417, -4.9294834, -5.3906240, -4.9294453, -0.1508124, 0.1508514

Time for backsubstitution: 5.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3577

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128760, upper bound: 0.0125140
time: 2.58 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0125143
time: 2.42 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.6830573, -3.2785523, -3.6830878, -3.2765720, -0.1897561, 0.1865425
1: -6.4960632, -5.8339100, -6.4962816, -5.8322825, -0.2428420, 0.2425961
2: -0.4305448, -0.2741566, -0.4305344, -0.2730291, -0.0533397, 0.0510623
3: -1.0987203, -0.8155407, -1.1016771, -0.8155471, -0.0585943, 0.0616596
4: -0.6262842, -0.4625589, -0.6263036, -0.4604604, -0.0400482, 0.0379114
5: -0.0520844, 0.2255305, -0.0546564, 0.2256963, -0.0619480, 0.0644242
6: -4.1222095, -3.6151710, -4.1224051, -3.6147664, -0.1192315, 0.1176348
7: 1.2784808, 1.5137215, 1.2783897, 1.5162885, -0.0270857, 0.0253207
8: -6.2269735, -5.8317323, -6.2285070, -5.8317394, -0.1085480, 0.1103091
9: -5.3906503, -4.9295454, -5.3918419, -4.9294686, -0.1506349, 0.1543254

Time for backsubstitution: 5.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3577

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128760, upper bound: 0.0126409
time: 2.84 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0126402
time: 2.47 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.6823816, -3.2779386, -3.6824362, -3.2786102, -0.1864285, 0.1874958
1: -6.4943838, -5.8339214, -6.4946003, -5.8339081, -0.2418305, 0.2428811
2: -0.4305274, -0.2741534, -0.4304495, -0.2742306, -0.0510415, 0.0513450
3: -1.0987762, -0.8157216, -1.0987197, -0.8158530, -0.0585463, 0.0585824
4: -0.6261433, -0.4620855, -0.6260455, -0.4625633, -0.0378577, 0.0387420
5: -0.0544764, 0.2254514, -0.0520842, 0.2253929, -0.0651081, 0.0619191
6: -4.1228628, -3.6126518, -4.1228466, -3.6151931, -0.1178381, 0.1208164
7: 1.2753137, 1.5145922, 1.2786193, 1.5145928, -0.0283820, 0.0250042
8: -6.2262440, -5.8314996, -6.2263708, -5.8317099, -0.1085169, 0.1093315
9: -5.3918552, -4.9291592, -5.3906307, -4.9292135, -0.1522816, 0.1510107

Time for backsubstitution: 5.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3577

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128760, upper bound: 0.0127769
time: 72.72 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0127774
time: 3.14 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.6824219, -3.2779124, -3.6826031, -3.2765505, -0.1896895, 0.1876270
1: -6.4944286, -5.8339243, -6.4949703, -5.8322825, -0.2428545, 0.2436575
2: -0.4305480, -0.2741421, -0.4305344, -0.2730748, -0.0532211, 0.0513441
3: -1.0987765, -0.8156697, -1.1016772, -0.8156483, -0.0587073, 0.0616547
4: -0.6261819, -0.4620836, -0.6261926, -0.4604590, -0.0400836, 0.0388884
5: -0.0544764, 0.2255154, -0.0546565, 0.2256004, -0.0652484, 0.0645567
6: -4.1228733, -3.6126490, -4.1228814, -3.6147594, -0.1194289, 0.1206658
7: 1.2752805, 1.5145922, 1.2783883, 1.5169146, -0.0284538, 0.0255272
8: -6.2262459, -5.8315802, -6.2278953, -5.8317394, -0.1085891, 0.1110611
9: -5.3918638, -4.9292216, -5.3918490, -4.9292355, -0.1521040, 0.1544848

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3577

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128760, upper bound: 0.0129030
time: 2.45 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0129035
time: 4.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 12.27 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 12.27
Output dim: 7, lower bound: -0.0128760, upper bound: 0.0125140
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 12.27
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0125143
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 12.27
Output dim: 7, lower bound: -0.0128760, upper bound: 0.0126409
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 12.27
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0126402
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 12.27
Output dim: 7, lower bound: -0.0128760, upper bound: 0.0127769
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 12.27
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0127774
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 12.27
Output dim: 7, lower bound: -0.0128760, upper bound: 0.0129030
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 12.27
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0129035

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.6830721, -3.2785871, -3.6829205, -3.2786388, -0.1865638, 0.1863606
1: -6.4961777, -5.8339353, -6.4959116, -5.8339300, -0.2420331, 0.2416641
2: -0.4305239, -0.2741649, -0.4304489, -0.2741846, -0.0511546, 0.0510672
3: -1.0987740, -0.8156034, -1.0987195, -0.8157609, -0.0585175, 0.0585266
4: -0.6262542, -0.4625609, -0.6261564, -0.4625644, -0.0378301, 0.0377594
5: -0.0521281, 0.2254617, -0.0520843, 0.2254851, -0.0618660, 0.0617447
6: -4.1223741, -3.6151872, -4.1223702, -3.6152108, -0.1178452, 0.1176398
7: 1.2785391, 1.5140288, 1.2786407, 1.5139666, -0.0248489, 0.0251528
8: -6.2269721, -5.8316202, -6.2269821, -5.8317099, -0.1084538, 0.1086095
9: -5.3908291, -4.9295135, -5.3906236, -4.9294701, -0.1510573, 0.1506770

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 278
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 278

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126994, upper bound: 0.0125141
time: 2.55 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126994, upper bound: 0.0125141
time: 2.57 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.6831126, -3.2785606, -3.6830878, -3.2765789, -0.1898252, 0.1864916
1: -6.4962244, -5.8339376, -6.4962816, -5.8323040, -0.2430578, 0.2424407
2: -0.4305446, -0.2741537, -0.4305337, -0.2730291, -0.0533370, 0.0510664
3: -1.0987746, -0.8155514, -1.1016771, -0.8155563, -0.0586784, 0.0615989
4: -0.6262928, -0.4625588, -0.6263036, -0.4604604, -0.0400559, 0.0379058
5: -0.0521281, 0.2255260, -0.0546564, 0.2256925, -0.0620063, 0.0643823
6: -4.1223841, -3.6151843, -4.1224051, -3.6147776, -0.1194360, 0.1174889
7: 1.2785057, 1.5140288, 1.2784106, 1.5162885, -0.0268330, 0.0256758
8: -6.2269740, -5.8317013, -6.2285075, -5.8317394, -0.1085261, 0.1103399
9: -5.3908386, -4.9295759, -5.3918419, -4.9294930, -0.1508797, 0.1541510

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 278
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 278

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126994, upper bound: 0.0126404
time: 2.57 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126994, upper bound: 0.0126404
time: 2.59 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.6824365, -3.2779479, -3.6824362, -3.2786176, -0.1864977, 0.1874450
1: -6.4945459, -5.8339491, -6.4946003, -5.8339300, -0.2420464, 0.2427255
2: -0.4305272, -0.2741506, -0.4304489, -0.2742306, -0.0510388, 0.0513490
3: -1.0988305, -0.8157322, -1.0987197, -0.8158618, -0.0586304, 0.0585217
4: -0.6261518, -0.4620855, -0.6260455, -0.4625633, -0.0378655, 0.0387363
5: -0.0545201, 0.2254466, -0.0520842, 0.2253889, -0.0651664, 0.0618773
6: -4.1230373, -3.6126657, -4.1228466, -3.6152048, -0.1180425, 0.1206706
7: 1.2753385, 1.5148994, 1.2786398, 1.5145928, -0.0281334, 0.0253593
8: -6.2262440, -5.8314681, -6.2263703, -5.8317099, -0.1084951, 0.1093624
9: -5.3920441, -4.9291897, -5.3906302, -4.9292378, -0.1525263, 0.1508364

Time for backsubstitution: 5.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 278
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 278

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126403, upper bound: 0.0127767
time: 32.53 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0126403, upper bound: 0.0127768
time: 17.25 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.6824219, -3.2779601, -3.6826031, -3.2765913, -0.1896437, 0.1875715
1: -6.4944286, -5.8340654, -6.4949703, -5.8324022, -0.2427161, 0.2434884
2: -0.4305455, -0.2741420, -0.4305322, -0.2730747, -0.0532174, 0.0513410
3: -1.0987765, -0.8157166, -1.1016772, -0.8156891, -0.0586534, 0.0615885
4: -0.6261819, -0.4620901, -0.6261926, -0.4604645, -0.0400782, 0.0388820
5: -0.0544764, 0.2254794, -0.0546565, 0.2255695, -0.0652110, 0.0645108
6: -4.1228733, -3.6127930, -4.1228814, -3.6148782, -0.1192967, 0.1205038
7: 1.2755358, 1.5145922, 1.2785987, 1.5169146, -0.0281732, 0.0252994
8: -6.2262216, -5.8315802, -6.2278748, -5.8317394, -0.1085642, 0.1110404
9: -5.3918638, -4.9293880, -5.3918490, -4.9293752, -0.1519462, 0.1542941

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 278
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 278

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126133, upper bound: 0.0129028
time: 2.52 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126134, upper bound: 0.0129032
time: 3.83 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.6824768, -3.2779212, -3.6826031, -3.2765579, -0.1897585, 0.1875762
1: -6.4945903, -5.8339510, -6.4949703, -5.8323040, -0.2430701, 0.2435023
2: -0.4305479, -0.2741390, -0.4305337, -0.2730747, -0.0532184, 0.0513481
3: -1.0988308, -0.8156803, -1.1016772, -0.8156573, -0.0587914, 0.0615940
4: -0.6261907, -0.4620836, -0.6261926, -0.4604589, -0.0400913, 0.0388827
5: -0.0545201, 0.2255108, -0.0546565, 0.2255963, -0.0653067, 0.0645149
6: -4.1230483, -3.6126628, -4.1228814, -3.6147718, -0.1196333, 0.1205199
7: 1.2753053, 1.5148994, 1.2784088, 1.5169146, -0.0282052, 0.0258822
8: -6.2262459, -5.8315487, -6.2278948, -5.8317394, -0.1085674, 0.1110920
9: -5.3920527, -4.9292517, -5.3918490, -4.9292607, -0.1523488, 0.1543105

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 278
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 278

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126403, upper bound: 0.0129029
time: 9.64 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126401, upper bound: 0.0129025
time: 2.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 17.53 seconds
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 17.53
Output dim: 7, lower bound: -0.0126994, upper bound: 0.0125141
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 17.53
Output dim: 7, lower bound: -0.0126994, upper bound: 0.0125141
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 17.53
Output dim: 7, lower bound: -0.0126994, upper bound: 0.0126404
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 17.53
Output dim: 7, lower bound: -0.0126994, upper bound: 0.0126404
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 17.53
Output dim: 7, lower bound: -0.0126403, upper bound: 0.0127767
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 17.53
Output dim: 7, lower bound: -0.0126403, upper bound: 0.0127768
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -0.0126133, upper bound: 0.0129028
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -0.0126134, upper bound: 0.0129032
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -0.0126403, upper bound: 0.0129029
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.53
Output dim: 7, lower bound: -0.0126401, upper bound: 0.0129025

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.6824219, -3.2779601, -3.6830373, -3.2766228, -0.1895756, 0.1876408
1: -6.4944286, -5.8340654, -6.4962797, -5.8324022, -0.2423337, 0.2437270
2: -0.4305455, -0.2741420, -0.4305322, -0.2730374, -0.0533301, 0.0510428
3: -1.0987765, -0.8157166, -1.1016772, -0.8155937, -0.0586726, 0.0615535
4: -0.6261819, -0.4620901, -0.6262730, -0.4604664, -0.0402579, 0.0386641
5: -0.0544764, 0.2254794, -0.0546565, 0.2254997, -0.0647061, 0.0648638
6: -4.1228733, -3.6127930, -4.1222095, -3.6148882, -0.1197686, 0.1198271
7: 1.2755358, 1.5145922, 1.2786009, 1.5160433, -0.0272955, 0.0259042
8: -6.2262216, -5.8315802, -6.2284737, -5.8317394, -0.1084225, 0.1111136
9: -5.3918638, -4.9293880, -5.3918381, -4.9297028, -0.1516156, 0.1544610

Time for backsubstitution: 5.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2403

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125580, upper bound: 0.0129021
time: 12.54 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126121, upper bound: 0.0129022
time: 3.39 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.6824219, -3.2779601, -3.6824017, -3.2759752, -0.1899205, 0.1866102
1: -6.4944286, -5.8340654, -6.4946442, -5.8324170, -0.2427160, 0.2423970
2: -0.4305455, -0.2741420, -0.4305355, -0.2730252, -0.0536299, 0.0513391
3: -1.0987765, -0.8157166, -1.1017330, -0.8157228, -0.0585280, 0.0615939
4: -0.6261819, -0.4620901, -0.6261706, -0.4599889, -0.0402061, 0.0380048
5: -0.0544764, 0.2254794, -0.0570486, 0.2254846, -0.0618803, 0.0645124
6: -4.1228733, -3.6127930, -4.1228733, -3.6123657, -0.1194711, 0.1176488
7: 1.2755358, 1.5145922, 1.2754004, 1.5169141, -0.0268184, 0.0253376
8: -6.2262216, -5.8315802, -6.2277441, -5.8315878, -0.1085640, 0.1103154
9: -5.3918638, -4.9293880, -5.3930511, -4.9293795, -0.1509239, 0.1546778

Time for backsubstitution: 5.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2403

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125580, upper bound: 0.0129018
time: 2.52 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126122, upper bound: 0.0129015
time: 43.76 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.6824768, -3.2779212, -3.6830373, -3.2765889, -0.1896904, 0.1876456
1: -6.4945903, -5.8339510, -6.4962797, -5.8323040, -0.2426877, 0.2437407
2: -0.4305479, -0.2741390, -0.4305337, -0.2730374, -0.0533311, 0.0510499
3: -1.0988308, -0.8156803, -1.1016773, -0.8155620, -0.0588106, 0.0615589
4: -0.6261907, -0.4620836, -0.6262730, -0.4604608, -0.0402710, 0.0386649
5: -0.0545201, 0.2255108, -0.0546565, 0.2255263, -0.0648019, 0.0648679
6: -4.1230483, -3.6126628, -4.1222095, -3.6147814, -0.1201054, 0.1198432
7: 1.2753053, 1.5148994, 1.2784109, 1.5160433, -0.0273275, 0.0264871
8: -6.2262459, -5.8315487, -6.2284932, -5.8317394, -0.1084257, 0.1111652
9: -5.3920527, -4.9292517, -5.3918381, -4.9295888, -0.1520182, 0.1544773

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2403

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125849, upper bound: 0.0129010
time: 6.82 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126391, upper bound: 0.0129020
time: 7.45 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.6824768, -3.2779212, -3.6824017, -3.2759418, -0.1900353, 0.1866150
1: -6.4945903, -5.8339510, -6.4946442, -5.8323183, -0.2430700, 0.2424108
2: -0.4305479, -0.2741390, -0.4305370, -0.2730252, -0.0536309, 0.0513462
3: -1.0988308, -0.8156803, -1.1017330, -0.8156906, -0.0586661, 0.0615994
4: -0.6261907, -0.4620836, -0.6261706, -0.4599835, -0.0402192, 0.0380056
5: -0.0545201, 0.2255108, -0.0570486, 0.2255114, -0.0619761, 0.0645165
6: -4.1230483, -3.6126628, -4.1228733, -3.6122594, -0.1198078, 0.1176649
7: 1.2753053, 1.5148994, 1.2752106, 1.5169141, -0.0268453, 0.0259205
8: -6.2262459, -5.8315487, -6.2277641, -5.8315878, -0.1085673, 0.1103669
9: -5.3920527, -4.9292517, -5.3930511, -4.9292650, -0.1513266, 0.1546942

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2403

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125849, upper bound: 0.0129015
time: 2.46 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126390, upper bound: 0.0129017
time: 2.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 10.40 seconds
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 10.40
Output dim: 7, lower bound: -0.0125580, upper bound: 0.0129021
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 10.40
Output dim: 7, lower bound: -0.0126121, upper bound: 0.0129022
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 10.40
Output dim: 7, lower bound: -0.0125580, upper bound: 0.0129018
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 10.40
Output dim: 7, lower bound: -0.0126122, upper bound: 0.0129015
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 10.40
Output dim: 7, lower bound: -0.0125849, upper bound: 0.0129010
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 10.40
Output dim: 7, lower bound: -0.0126391, upper bound: 0.0129020
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 10.40
Output dim: 7, lower bound: -0.0125849, upper bound: 0.0129015
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 10.40
Output dim: 7, lower bound: -0.0126390, upper bound: 0.0129017

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.6824219, -3.2781348, -3.6830373, -3.2767625, -0.1893701, 0.1873959
1: -6.4944286, -5.8343844, -6.4962792, -5.8326621, -0.2420040, 0.2433255
2: -0.4304929, -0.2741421, -0.4304892, -0.2730377, -0.0532685, 0.0509923
3: -1.0987766, -0.8157672, -1.1016771, -0.8156347, -0.0586255, 0.0614954
4: -0.6261531, -0.4620903, -0.6262494, -0.4604666, -0.0402259, 0.0386380
5: -0.0544711, 0.2254586, -0.0546522, 0.2254829, -0.0646762, 0.0648321
6: -4.1228294, -3.6128132, -4.1221738, -3.6149044, -0.1196888, 0.1197598
7: 1.2757747, 1.5145922, 1.2787956, 1.5160433, -0.0269972, 0.0256667
8: -6.2262211, -5.8323340, -6.2284741, -5.8323498, -0.1075536, 0.1100496
9: -5.3918638, -4.9299684, -5.3918381, -4.9301810, -0.1511707, 0.1539170

Time for backsubstitution: 5.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2531

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125418, upper bound: 0.0128970
time: 52.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125535, upper bound: 0.0128972
time: 2.57 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.6824183, -3.2780435, -3.6830373, -3.2767315, -0.1895485, 0.1876126
1: -6.4944892, -5.8340707, -6.4962792, -5.8324552, -0.2424426, 0.2436920
2: -0.4305289, -0.2740873, -0.4305120, -0.2730379, -0.0532883, 0.0511010
3: -1.0988317, -0.8157451, -1.1016769, -0.8156222, -0.0587232, 0.0615088
4: -0.6261740, -0.4620551, -0.6262645, -0.4604667, -0.0402335, 0.0387022
5: -0.0545021, 0.2254684, -0.0546565, 0.2254909, -0.0647186, 0.0648448
6: -4.1228905, -3.6128068, -4.1221972, -3.6149001, -0.1197545, 0.1198193
7: 1.2755134, 1.5146855, 1.2786118, 1.5160433, -0.0272385, 0.0259848
8: -6.2262688, -5.8316875, -6.2284732, -5.8319001, -0.1087365, 0.1109378
9: -5.3919568, -4.9293528, -5.3918371, -4.9297342, -0.1517767, 0.1544923

Time for backsubstitution: 5.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2531

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125961, upper bound: 0.0128971
time: 2.50 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126077, upper bound: 0.0128976
time: 3.54 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.6824219, -3.2781348, -3.6824017, -3.2761164, -0.1897153, 0.1863654
1: -6.4944286, -5.8343844, -6.4946432, -5.8326759, -0.2423862, 0.2419955
2: -0.4304929, -0.2741421, -0.4304925, -0.2730254, -0.0535683, 0.0512886
3: -1.0987766, -0.8157672, -1.1017331, -0.8157636, -0.0584809, 0.0615359
4: -0.6261531, -0.4620903, -0.6261470, -0.4599890, -0.0401741, 0.0379787
5: -0.0544711, 0.2254586, -0.0570444, 0.2254677, -0.0618504, 0.0644806
6: -4.1228294, -3.6128132, -4.1228380, -3.6123824, -0.1193913, 0.1175815
7: 1.2757747, 1.5145922, 1.2755952, 1.5169141, -0.0265344, 0.0251001
8: -6.2262211, -5.8323340, -6.2277436, -5.8321977, -0.1076951, 0.1092514
9: -5.3918638, -4.9299684, -5.3930516, -4.9298573, -0.1504789, 0.1541337

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2531

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125419, upper bound: 0.0128973
time: 2.48 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125534, upper bound: 0.0128974
time: 5.68 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.6824183, -3.2780435, -3.6824017, -3.2760851, -0.1898937, 0.1865820
1: -6.4944892, -5.8340707, -6.4946446, -5.8324690, -0.2428250, 0.2423617
2: -0.4305289, -0.2740873, -0.4305153, -0.2730257, -0.0535881, 0.0513973
3: -1.0988317, -0.8157451, -1.1017331, -0.8157508, -0.0585787, 0.0615492
4: -0.6261740, -0.4620551, -0.6261622, -0.4599892, -0.0401817, 0.0380429
5: -0.0545021, 0.2254684, -0.0570487, 0.2254757, -0.0618928, 0.0644934
6: -4.1228905, -3.6128068, -4.1228604, -3.6123776, -0.1194570, 0.1176410
7: 1.2755134, 1.5146855, 1.2754114, 1.5169141, -0.0267635, 0.0254183
8: -6.2262688, -5.8316875, -6.2277436, -5.8317480, -0.1088778, 0.1101395
9: -5.3919568, -4.9293528, -5.3930507, -4.9294100, -0.1510850, 0.1547093

Time for backsubstitution: 5.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2531

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125961, upper bound: 0.0128974
time: 2.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126077, upper bound: 0.0128978
time: 2.49 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.6824768, -3.2780962, -3.6830373, -3.2767289, -0.1894849, 0.1874006
1: -6.4945898, -5.8342690, -6.4962792, -5.8325644, -0.2423580, 0.2433393
2: -0.4304952, -0.2741392, -0.4304909, -0.2730377, -0.0532695, 0.0509995
3: -1.0988308, -0.8157307, -1.1016771, -0.8156025, -0.0587636, 0.0615009
4: -0.6261617, -0.4620836, -0.6262494, -0.4604610, -0.0402390, 0.0386388
5: -0.0545148, 0.2254902, -0.0546522, 0.2255098, -0.0647720, 0.0648361
6: -4.1230040, -3.6126831, -4.1221738, -3.6147974, -0.1200255, 0.1197760
7: 1.2755439, 1.5148994, 1.2786057, 1.5160433, -0.0270292, 0.0262496
8: -6.2262459, -5.8323021, -6.2284937, -5.8323493, -0.1075568, 0.1101011
9: -5.3920517, -4.9298329, -5.3918381, -4.9300666, -0.1515733, 0.1539333

Time for backsubstitution: 5.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2531

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125687, upper bound: 0.0128978
time: 2.86 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125804, upper bound: 0.0128970
time: 2.53 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.6824737, -3.2780042, -3.6830373, -3.2766986, -0.1896634, 0.1876172
1: -6.4946508, -5.8339562, -6.4962792, -5.8323574, -0.2427968, 0.2437060
2: -0.4305312, -0.2740843, -0.4305137, -0.2730379, -0.0532893, 0.0511082
3: -1.0988860, -0.8157087, -1.1016769, -0.8155901, -0.0588613, 0.0615143
4: -0.6261827, -0.4620485, -0.6262645, -0.4604613, -0.0402465, 0.0387030
5: -0.0545458, 0.2255000, -0.0546565, 0.2255175, -0.0648143, 0.0648489
6: -4.1230640, -3.6126766, -4.1221972, -3.6147928, -0.1200912, 0.1198354
7: 1.2752829, 1.5149928, 1.2784219, 1.5160433, -0.0272705, 0.0265677
8: -6.2262936, -5.8316555, -6.2284937, -5.8318996, -0.1087397, 0.1109892
9: -5.3921447, -4.9292164, -5.3918371, -4.9296198, -0.1521793, 0.1545088

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2531

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126230, upper bound: 0.0128973
time: 2.53 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126345, upper bound: 0.0128971
time: 2.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.6824768, -3.2780962, -3.6824017, -3.2760828, -0.1898299, 0.1863701
1: -6.4945898, -5.8342690, -6.4946432, -5.8325777, -0.2427404, 0.2420092
2: -0.4304952, -0.2741392, -0.4304941, -0.2730254, -0.0535693, 0.0512958
3: -1.0988308, -0.8157307, -1.1017331, -0.8157315, -0.0586190, 0.0615413
4: -0.6261617, -0.4620836, -0.6261470, -0.4599833, -0.0401872, 0.0379795
5: -0.0545148, 0.2254902, -0.0570444, 0.2254946, -0.0619462, 0.0644847
6: -4.1230040, -3.6126831, -4.1228380, -3.6122761, -0.1197281, 0.1175976
7: 1.2755439, 1.5148994, 1.2754053, 1.5169141, -0.0265613, 0.0256830
8: -6.2262459, -5.8323021, -6.2277632, -5.8321977, -0.1076983, 0.1093029
9: -5.3920517, -4.9298329, -5.3930516, -4.9297423, -0.1508817, 0.1541501

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2531

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125688, upper bound: 0.0128974
time: 7.25 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125805, upper bound: 0.0128972
time: 2.55 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.6824737, -3.2780042, -3.6824017, -3.2760518, -0.1900084, 0.1865867
1: -6.4946508, -5.8339562, -6.4946446, -5.8323708, -0.2431792, 0.2423756
2: -0.4305312, -0.2740843, -0.4305168, -0.2730257, -0.0535891, 0.0514044
3: -1.0988860, -0.8157087, -1.1017331, -0.8157189, -0.0587168, 0.0615547
4: -0.6261827, -0.4620485, -0.6261622, -0.4599838, -0.0401947, 0.0380437
5: -0.0545458, 0.2255000, -0.0570487, 0.2255025, -0.0619885, 0.0644975
6: -4.1230640, -3.6126766, -4.1228604, -3.6122713, -0.1197937, 0.1176571
7: 1.2752829, 1.5149928, 1.2752215, 1.5169141, -0.0267905, 0.0260011
8: -6.2262936, -5.8316555, -6.2277641, -5.8317480, -0.1088812, 0.1101912
9: -5.3921447, -4.9292164, -5.3930507, -4.9292955, -0.1514876, 0.1547257

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2531

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126230, upper bound: 0.0128971
time: 2.58 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0128978
time: 12.97 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 21.09 seconds
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0125418, upper bound: 0.0128970
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0125535, upper bound: 0.0128972
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0125961, upper bound: 0.0128971
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0126077, upper bound: 0.0128976
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0125419, upper bound: 0.0128973
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0125534, upper bound: 0.0128974
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0125961, upper bound: 0.0128974
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0126077, upper bound: 0.0128978
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0125687, upper bound: 0.0128978
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0125804, upper bound: 0.0128970
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0126230, upper bound: 0.0128973
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0126345, upper bound: 0.0128971
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0125688, upper bound: 0.0128974
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0125805, upper bound: 0.0128972
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0126230, upper bound: 0.0128971
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -0.0126346, upper bound: 0.0128978

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.6821961, -3.2781355, -3.6827531, -3.2767637, -0.1891236, 0.1870962
1: -6.4941893, -5.8343844, -6.4959745, -5.8326621, -0.2417301, 0.2429818
2: -0.4304929, -0.2741581, -0.4304892, -0.2730579, -0.0532448, 0.0509736
3: -1.0987763, -0.8158420, -1.1016769, -0.8157295, -0.0585112, 0.0614047
4: -0.6261531, -0.4620911, -0.6262494, -0.4604673, -0.0402224, 0.0386346
5: -0.0544713, 0.2253863, -0.0546523, 0.2253928, -0.0645612, 0.0647408
6: -4.1228294, -3.6128783, -4.1221738, -3.6149852, -0.1195936, 0.1196835
7: 1.2757747, 1.5145390, 1.2787956, 1.5159761, -0.0268958, 0.0255989
8: -6.2260551, -5.8323340, -6.2282639, -5.8323498, -0.1073853, 0.1098365
9: -5.3918004, -4.9299684, -5.3917556, -4.9301810, -0.1511081, 0.1538376

Time for backsubstitution: 5.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124157, upper bound: 0.0128970
time: 2.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124157, upper bound: 0.0127711
time: 2.44 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.6821501, -3.2781360, -3.6826982, -3.2762942, -0.1899536, 0.1871094
1: -6.4941626, -5.8343844, -6.4959474, -5.8321738, -0.2425005, 0.2430074
2: -0.4304929, -0.2741804, -0.4305176, -0.2730831, -0.0532413, 0.0509964
3: -1.0987763, -0.8158636, -1.1018133, -0.8157548, -0.0585167, 0.0617096
4: -0.6261531, -0.4620919, -0.6262499, -0.4604673, -0.0402208, 0.0386347
5: -0.0544713, 0.2253597, -0.0547813, 0.2253596, -0.0645675, 0.0650443
6: -4.1228294, -3.6129105, -4.1222706, -3.6150234, -0.1195984, 0.1198434
7: 1.2757750, 1.5145448, 1.2786859, 1.5159848, -0.0269115, 0.0258369
8: -6.2260823, -5.8323340, -6.2283020, -5.8320017, -0.1079254, 0.1098592
9: -5.3917718, -4.9299684, -5.3917246, -4.9300714, -0.1511921, 0.1538379

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124272, upper bound: 0.0128975
time: 12.46 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124273, upper bound: 0.0127710
time: 2.57 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.6821921, -3.2780442, -3.6827531, -3.2767327, -0.1893020, 0.1873130
1: -6.4942493, -5.8340707, -6.4959750, -5.8324552, -0.2421691, 0.2433481
2: -0.4305289, -0.2741033, -0.4305120, -0.2730583, -0.0532646, 0.0510822
3: -1.0988318, -0.8158202, -1.1016771, -0.8157170, -0.0586089, 0.0614182
4: -0.6261740, -0.4620560, -0.6262645, -0.4604676, -0.0402300, 0.0386988
5: -0.0545020, 0.2253960, -0.0546565, 0.2254004, -0.0646036, 0.0647537
6: -4.1228905, -3.6128721, -4.1221972, -3.6149807, -0.1196593, 0.1197430
7: 1.2755134, 1.5146325, 1.2786119, 1.5159761, -0.0271370, 0.0259171
8: -6.2261019, -5.8316875, -6.2282634, -5.8319001, -0.1085682, 0.1107247
9: -5.3918924, -4.9293528, -5.3917561, -4.9297342, -0.1517141, 0.1544132

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124698, upper bound: 0.0128973
time: 2.62 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124698, upper bound: 0.0127714
time: 2.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.6821461, -3.2780449, -3.6826982, -3.2762635, -0.1901323, 0.1873261
1: -6.4942231, -5.8340707, -6.4959469, -5.8319674, -0.2429391, 0.2433740
2: -0.4305289, -0.2741256, -0.4305404, -0.2730834, -0.0532610, 0.0511051
3: -1.0988317, -0.8158417, -1.1018132, -0.8157423, -0.0586145, 0.0617229
4: -0.6261740, -0.4620567, -0.6262648, -0.4604675, -0.0402284, 0.0386990
5: -0.0545020, 0.2253693, -0.0547856, 0.2253673, -0.0646098, 0.0650571
6: -4.1228905, -3.6129041, -4.1222944, -3.6150188, -0.1196643, 0.1199028
7: 1.2755139, 1.5146383, 1.2785023, 1.5159848, -0.0271527, 0.0261550
8: -6.2261291, -5.8316875, -6.2283010, -5.8315525, -0.1091083, 0.1107473
9: -5.3918648, -4.9293528, -5.3917236, -4.9296246, -0.1517982, 0.1544134

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124814, upper bound: 0.0128968
time: 3.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124814, upper bound: 0.0127708
time: 2.48 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.6821961, -3.2781355, -3.6821175, -3.2761173, -0.1894689, 0.1860657
1: -6.4941893, -5.8343844, -6.4943399, -5.8326759, -0.2421125, 0.2416515
2: -0.4304929, -0.2741581, -0.4304925, -0.2730458, -0.0535446, 0.0512699
3: -1.0987763, -0.8158420, -1.1017333, -0.8158586, -0.0583666, 0.0614452
4: -0.6261531, -0.4620911, -0.6261470, -0.4599898, -0.0401706, 0.0379753
5: -0.0544713, 0.2253863, -0.0570444, 0.2253776, -0.0617354, 0.0643894
6: -4.1228294, -3.6128783, -4.1228380, -3.6124630, -0.1192962, 0.1175052
7: 1.2757747, 1.5145390, 1.2755953, 1.5168469, -0.0264493, 0.0250324
8: -6.2260551, -5.8323340, -6.2275329, -5.8321977, -0.1075269, 0.1090382
9: -5.3918004, -4.9299684, -5.3929701, -4.9298573, -0.1504164, 0.1540545

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124157, upper bound: 0.0128970
time: 2.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124157, upper bound: 0.0127714
time: 24.55 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.6821501, -3.2781360, -3.6820629, -3.2756481, -0.1902986, 0.1860788
1: -6.4941626, -5.8343844, -6.4943113, -5.8321881, -0.2428827, 0.2416775
2: -0.4304929, -0.2741804, -0.4305209, -0.2730710, -0.0535411, 0.0512927
3: -1.0987763, -0.8158636, -1.1018691, -0.8158838, -0.0583721, 0.0617500
4: -0.6261531, -0.4620919, -0.6261474, -0.4599897, -0.0401690, 0.0379755
5: -0.0544713, 0.2253597, -0.0571735, 0.2253444, -0.0617417, 0.0646928
6: -4.1228294, -3.6129105, -4.1229343, -3.6125021, -0.1193011, 0.1176649
7: 1.2757750, 1.5145448, 1.2754859, 1.5168555, -0.0264524, 0.0252702
8: -6.2260823, -5.8323340, -6.2275720, -5.8318510, -0.1080670, 0.1090609
9: -5.3917718, -4.9299684, -5.3929386, -4.9297476, -0.1505004, 0.1540548

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124273, upper bound: 0.0128972
time: 2.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124272, upper bound: 0.0127716
time: 2.84 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.6821921, -3.2780442, -3.6821175, -3.2760868, -0.1896471, 0.1862825
1: -6.4942493, -5.8340707, -6.4943399, -5.8324690, -0.2425512, 0.2420180
2: -0.4305289, -0.2741033, -0.4305153, -0.2730459, -0.0535644, 0.0513785
3: -1.0988318, -0.8158202, -1.1017333, -0.8158458, -0.0584643, 0.0614586
4: -0.6261740, -0.4620560, -0.6261622, -0.4599899, -0.0401782, 0.0380395
5: -0.0545020, 0.2253960, -0.0570486, 0.2253852, -0.0617778, 0.0644022
6: -4.1228905, -3.6128721, -4.1228604, -3.6124587, -0.1193619, 0.1175647
7: 1.2755134, 1.5146325, 1.2754115, 1.5168469, -0.0266784, 0.0253505
8: -6.2261019, -5.8316875, -6.2275333, -5.8317480, -0.1087095, 0.1099263
9: -5.3918924, -4.9293528, -5.3929691, -4.9294100, -0.1510225, 0.1546301

Time for backsubstitution: 5.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124699, upper bound: 0.0128971
time: 2.60 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124698, upper bound: 0.0127708
time: 2.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.6821461, -3.2780449, -3.6820629, -3.2756176, -0.1904773, 0.1862954
1: -6.4942231, -5.8340707, -6.4943123, -5.8319807, -0.2433213, 0.2420440
2: -0.4305289, -0.2741256, -0.4305437, -0.2730713, -0.0535609, 0.0514013
3: -1.0988317, -0.8158417, -1.1018691, -0.8158711, -0.0584699, 0.0617634
4: -0.6261740, -0.4620567, -0.6261625, -0.4599897, -0.0401766, 0.0380397
5: -0.0545020, 0.2253693, -0.0571774, 0.2253520, -0.0617840, 0.0647056
6: -4.1228905, -3.6129041, -4.1229587, -3.6124964, -0.1193668, 0.1177245
7: 1.2755139, 1.5146383, 1.2753018, 1.5168555, -0.0266816, 0.0255884
8: -6.2261291, -5.8316875, -6.2275724, -5.8314013, -0.1092498, 0.1099490
9: -5.3918648, -4.9293528, -5.3929377, -4.9293008, -0.1511065, 0.1546303

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124814, upper bound: 0.0128972
time: 2.64 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124815, upper bound: 0.0127708
time: 2.41 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.6822507, -3.2780964, -3.6827531, -3.2767303, -0.1892383, 0.1871011
1: -6.4943504, -5.8342690, -6.4959745, -5.8325644, -0.2420843, 0.2429956
2: -0.4304952, -0.2741554, -0.4304909, -0.2730579, -0.0532459, 0.0509807
3: -1.0988306, -0.8158056, -1.1016768, -0.8156976, -0.0586492, 0.0614102
4: -0.6261617, -0.4620844, -0.6262494, -0.4604620, -0.0402355, 0.0386354
5: -0.0545148, 0.2254177, -0.0546523, 0.2254194, -0.0646569, 0.0647449
6: -4.1230040, -3.6127484, -4.1221738, -3.6148791, -0.1199302, 0.1196995
7: 1.2755440, 1.5148463, 1.2786058, 1.5159761, -0.0269278, 0.0261818
8: -6.2260795, -5.8323021, -6.2282839, -5.8323493, -0.1073885, 0.1098880
9: -5.3919888, -4.9298329, -5.3917556, -4.9300666, -0.1515108, 0.1538540

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124425, upper bound: 0.0128971
time: 2.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124427, upper bound: 0.0127705
time: 2.37 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.6822052, -3.2780969, -3.6826982, -3.2762604, -0.1900685, 0.1871142
1: -6.4943237, -5.8342690, -6.4959474, -5.8320756, -0.2428544, 0.2430212
2: -0.4304952, -0.2741775, -0.4305193, -0.2730831, -0.0532423, 0.0510036
3: -1.0988305, -0.8158271, -1.1018133, -0.8157228, -0.0586548, 0.0617151
4: -0.6261617, -0.4620851, -0.6262499, -0.4604617, -0.0402339, 0.0386355
5: -0.0545149, 0.2253909, -0.0547813, 0.2253863, -0.0646632, 0.0650484
6: -4.1230040, -3.6127796, -4.1222706, -3.6149168, -0.1199352, 0.1198593
7: 1.2755444, 1.5148520, 1.2784961, 1.5159848, -0.0269435, 0.0264198
8: -6.2261071, -5.8323021, -6.2283211, -5.8320017, -0.1079287, 0.1099107
9: -5.3919601, -4.9298329, -5.3917246, -4.9299574, -0.1515948, 0.1538543

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124542, upper bound: 0.0128972
time: 14.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124541, upper bound: 0.0127709
time: 2.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.6822474, -3.2780049, -3.6827531, -3.2766991, -0.1894169, 0.1873178
1: -6.4944105, -5.8339562, -6.4959750, -5.8323574, -0.2425231, 0.2433621
2: -0.4305312, -0.2741003, -0.4305137, -0.2730583, -0.0532656, 0.0510894
3: -1.0988859, -0.8157835, -1.1016771, -0.8156850, -0.0587470, 0.0614236
4: -0.6261827, -0.4620494, -0.6262645, -0.4604621, -0.0402431, 0.0386996
5: -0.0545456, 0.2254276, -0.0546565, 0.2254273, -0.0646993, 0.0647577
6: -4.1230640, -3.6127419, -4.1221972, -3.6148746, -0.1199960, 0.1197592
7: 1.2752831, 1.5149395, 1.2784220, 1.5159761, -0.0271690, 0.0264999
8: -6.2261257, -5.8316555, -6.2282829, -5.8318996, -0.1085714, 0.1107761
9: -5.3920813, -4.9292164, -5.3917561, -4.9296198, -0.1521168, 0.1544295

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124967, upper bound: 0.0128972
time: 2.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124968, upper bound: 0.0127708
time: 2.51 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.6822009, -3.2780058, -3.6826982, -3.2762296, -0.1902471, 0.1873310
1: -6.4943848, -5.8339562, -6.4959469, -5.8318691, -0.2432933, 0.2433879
2: -0.4305312, -0.2741227, -0.4305421, -0.2730834, -0.0532620, 0.0511122
3: -1.0988861, -0.8158052, -1.1018132, -0.8157103, -0.0587525, 0.0617284
4: -0.6261827, -0.4620501, -0.6262648, -0.4604620, -0.0402414, 0.0386997
5: -0.0545458, 0.2254008, -0.0547856, 0.2253940, -0.0647056, 0.0650612
6: -4.1230640, -3.6127739, -4.1222944, -3.6149118, -0.1200010, 0.1199190
7: 1.2752831, 1.5149454, 1.2783124, 1.5159848, -0.0271847, 0.0267379
8: -6.2261543, -5.8316555, -6.2283211, -5.8315525, -0.1091115, 0.1107988
9: -5.3920527, -4.9292164, -5.3917236, -4.9295101, -0.1522008, 0.1544297

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125084, upper bound: 0.0128969
time: 2.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0125083, upper bound: 0.0127710
time: 2.56 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.6822507, -3.2780964, -3.6821175, -3.2760837, -0.1895836, 0.1860705
1: -6.4943504, -5.8342690, -6.4943399, -5.8325777, -0.2424664, 0.2416655
2: -0.4304952, -0.2741554, -0.4304941, -0.2730458, -0.0535456, 0.0512770
3: -1.0988306, -0.8158056, -1.1017333, -0.8158263, -0.0585047, 0.0614507
4: -0.6261617, -0.4620844, -0.6261470, -0.4599843, -0.0401837, 0.0379761
5: -0.0545148, 0.2254177, -0.0570444, 0.2254043, -0.0618311, 0.0643935
6: -4.1230040, -3.6127484, -4.1228380, -3.6123569, -0.1196329, 0.1175213
7: 1.2755440, 1.5148463, 1.2754056, 1.5168469, -0.0264762, 0.0256152
8: -6.2260795, -5.8323021, -6.2275534, -5.8321977, -0.1075301, 0.1090897
9: -5.3919888, -4.9298329, -5.3929701, -4.9297423, -0.1508192, 0.1540710

Time for backsubstitution: 5.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124426, upper bound: 0.0128969
time: 2.53 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124425, upper bound: 0.0127712
time: 5.45 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.6822052, -3.2780969, -3.6820629, -3.2756150, -0.1904135, 0.1860836
1: -6.4943237, -5.8342690, -6.4943113, -5.8320894, -0.2432367, 0.2416912
2: -0.4304952, -0.2741775, -0.4305224, -0.2730710, -0.0535421, 0.0512998
3: -1.0988305, -0.8158271, -1.1018691, -0.8158518, -0.0585102, 0.0617555
4: -0.6261617, -0.4620851, -0.6261474, -0.4599842, -0.0401820, 0.0379763
5: -0.0545149, 0.2253909, -0.0571735, 0.2253712, -0.0618374, 0.0646969
6: -4.1230040, -3.6127796, -4.1229343, -3.6123958, -0.1196378, 0.1176810
7: 1.2755444, 1.5148520, 1.2752960, 1.5168555, -0.0264793, 0.0258531
8: -6.2261071, -5.8323021, -6.2275915, -5.8318510, -0.1080702, 0.1091123
9: -5.3919601, -4.9298329, -5.3929386, -4.9296331, -0.1509031, 0.1540712

Time for backsubstitution: 5.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124542, upper bound: 0.0128975
time: 11.78 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124542, upper bound: 0.0127715
time: 6.26 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.6822474, -3.2780049, -3.6821175, -3.2760527, -0.1897619, 0.1862872
1: -6.4944105, -5.8339562, -6.4943399, -5.8323708, -0.2429054, 0.2420321
2: -0.4305312, -0.2741003, -0.4305168, -0.2730459, -0.0535654, 0.0513857
3: -1.0988859, -0.8157835, -1.1017333, -0.8158140, -0.0586024, 0.0614640
4: -0.6261827, -0.4620494, -0.6261622, -0.4599845, -0.0401913, 0.0380403
5: -0.0545456, 0.2254276, -0.0570486, 0.2254121, -0.0618735, 0.0644062
6: -4.1230640, -3.6127419, -4.1228604, -3.6123524, -0.1196986, 0.1175807
7: 1.2752831, 1.5149395, 1.2752217, 1.5168469, -0.0267054, 0.0259334
8: -6.2261257, -5.8316555, -6.2275534, -5.8317480, -0.1087129, 0.1099780
9: -5.3920813, -4.9292164, -5.3929691, -4.9292955, -0.1514252, 0.1546466

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0124967, upper bound: 0.0128968
time: 2.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124968, upper bound: 0.0127710
time: 2.52 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.6822009, -3.2780058, -3.6820629, -3.2755837, -0.1905919, 0.1863004
1: -6.4943848, -5.8339562, -6.4943123, -5.8318830, -0.2436755, 0.2420580
2: -0.4305312, -0.2741227, -0.4305452, -0.2730713, -0.0535619, 0.0514085
3: -1.0988861, -0.8158052, -1.1018691, -0.8158392, -0.0586080, 0.0617689
4: -0.6261827, -0.4620501, -0.6261625, -0.4599842, -0.0401896, 0.0380405
5: -0.0545458, 0.2254008, -0.0571774, 0.2253789, -0.0618798, 0.0647097
6: -4.1230640, -3.6127739, -4.1229587, -3.6123900, -0.1197035, 0.1177407
7: 1.2752831, 1.5149454, 1.2751122, 1.5168555, -0.0267085, 0.0261712
8: -6.2261543, -5.8316555, -6.2275915, -5.8314009, -0.1092530, 0.1100006
9: -5.3920527, -4.9292164, -5.3929377, -4.9291863, -0.1515092, 0.1546467

Time for backsubstitution: 5.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2125
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 347

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0125083, upper bound: 0.0128970
time: 2.63 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0125084, upper bound: 0.0127708
time: 2.42 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 10.62 seconds
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124157, upper bound: 0.0128970
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124157, upper bound: 0.0127711
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124272, upper bound: 0.0128975
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124273, upper bound: 0.0127710
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124698, upper bound: 0.0128973
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124698, upper bound: 0.0127714
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124814, upper bound: 0.0128968
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124814, upper bound: 0.0127708
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124157, upper bound: 0.0128970
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124157, upper bound: 0.0127714
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124273, upper bound: 0.0128972
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124272, upper bound: 0.0127716
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124699, upper bound: 0.0128971
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124698, upper bound: 0.0127708
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124814, upper bound: 0.0128972
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124815, upper bound: 0.0127708
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124425, upper bound: 0.0128971
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124427, upper bound: 0.0127705
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124542, upper bound: 0.0128972
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124541, upper bound: 0.0127709
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124967, upper bound: 0.0128972
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124968, upper bound: 0.0127708
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0125084, upper bound: 0.0128969
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0125083, upper bound: 0.0127710
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124426, upper bound: 0.0128969
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124425, upper bound: 0.0127712
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124542, upper bound: 0.0128975
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124542, upper bound: 0.0127715
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124967, upper bound: 0.0128968
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0124968, upper bound: 0.0127710
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0125083, upper bound: 0.0128970
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 10.62
Output dim: 7, lower bound: -0.0125084, upper bound: 0.0127708

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.6820087, -3.2782242, -3.6827531, -3.2767637, -0.1884719, 0.1871856
1: -6.4940376, -5.8343821, -6.4959745, -5.8326621, -0.2418314, 0.2422470
2: -0.4303975, -0.2741953, -0.4304892, -0.2730579, -0.0524081, 0.0510835
3: -1.0987756, -0.8160590, -1.1016769, -0.8157295, -0.0585915, 0.0611631
4: -0.6259944, -0.4620975, -0.6262494, -0.4604673, -0.0400467, 0.0386630
5: -0.0544708, 0.2251788, -0.0546523, 0.2253928, -0.0646282, 0.0645334
6: -4.1227942, -3.6129088, -4.1221738, -3.6149852, -0.1184763, 0.1198363
7: 1.2759156, 1.5145390, 1.2787956, 1.5159761, -0.0266726, 0.0251962
8: -6.2260494, -5.8323154, -6.2282639, -5.8323498, -0.1074256, 0.1097116
9: -5.3917713, -4.9299679, -5.3917556, -4.9301810, -0.1513592, 0.1517210

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0123806, upper bound: 0.0128890
time: 2.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124129, upper bound: 0.0128888
time: 2.60 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.6819625, -3.2782252, -3.6826982, -3.2762942, -0.1893019, 0.1871990
1: -6.4940100, -5.8343821, -6.4959474, -5.8321738, -0.2426015, 0.2422727
2: -0.4303975, -0.2742175, -0.4305176, -0.2730831, -0.0524045, 0.0511064
3: -1.0987756, -0.8160806, -1.1018133, -0.8157548, -0.0585971, 0.0614680
4: -0.6259944, -0.4620983, -0.6262499, -0.4604673, -0.0400450, 0.0386631
5: -0.0544709, 0.2251521, -0.0547813, 0.2253596, -0.0646345, 0.0648369
6: -4.1227942, -3.6129408, -4.1222706, -3.6150234, -0.1184814, 0.1199960
7: 1.2759157, 1.5145448, 1.2786859, 1.5159848, -0.0266883, 0.0254342
8: -6.2260766, -5.8323154, -6.2283020, -5.8320017, -0.1079656, 0.1097344
9: -5.3917427, -4.9299679, -5.3917246, -4.9300714, -0.1514431, 0.1517214

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0123922, upper bound: 0.0128891
time: 2.58 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124245, upper bound: 0.0128893
time: 2.63 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.6820047, -3.2781332, -3.6827531, -3.2767327, -0.1886505, 0.1874025
1: -6.4940982, -5.8340693, -6.4959750, -5.8324552, -0.2422701, 0.2426136
2: -0.4304336, -0.2741406, -0.4305120, -0.2730583, -0.0524278, 0.0511922
3: -1.0988314, -0.8160372, -1.1016771, -0.8157170, -0.0586893, 0.0611765
4: -0.6260155, -0.4620622, -0.6262645, -0.4604676, -0.0400542, 0.0387272
5: -0.0545017, 0.2251885, -0.0546565, 0.2254004, -0.0646705, 0.0645462
6: -4.1228552, -3.6129024, -4.1221972, -3.6149807, -0.1185421, 0.1198958
7: 1.2756541, 1.5146325, 1.2786119, 1.5159761, -0.0269138, 0.0255126
8: -6.2260971, -5.8316679, -6.2282634, -5.8319001, -0.1086068, 0.1106012
9: -5.3918629, -4.9293509, -5.3917561, -4.9297342, -0.1519651, 0.1522971

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124349, upper bound: 0.0128891
time: 2.63 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124671, upper bound: 0.0128894
time: 2.55 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.6819587, -3.2781336, -3.6826982, -3.2762635, -0.1894807, 0.1874159
1: -6.4940710, -5.8340693, -6.4959469, -5.8319674, -0.2430402, 0.2426393
2: -0.4304336, -0.2741627, -0.4305404, -0.2730834, -0.0524243, 0.0512151
3: -1.0988312, -0.8160588, -1.1018132, -0.8157423, -0.0586949, 0.0614813
4: -0.6260155, -0.4620631, -0.6262648, -0.4604675, -0.0400526, 0.0387274
5: -0.0545018, 0.2251617, -0.0547856, 0.2253673, -0.0646768, 0.0648497
6: -4.1228552, -3.6129348, -4.1222944, -3.6150188, -0.1185473, 0.1200554
7: 1.2756542, 1.5146383, 1.2785023, 1.5159848, -0.0269295, 0.0257506
8: -6.2261238, -5.8316679, -6.2283010, -5.8315525, -0.1091469, 0.1106238
9: -5.3918352, -4.9293509, -5.3917236, -4.9296246, -0.1520490, 0.1522973

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124465, upper bound: 0.0128891
time: 2.63 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124787, upper bound: 0.0128894
time: 2.50 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.6820087, -3.2782242, -3.6821175, -3.2761173, -0.1888173, 0.1861547
1: -6.4940376, -5.8343821, -6.4943399, -5.8326759, -0.2422140, 0.2409147
2: -0.4303975, -0.2741953, -0.4304925, -0.2730458, -0.0527078, 0.0513803
3: -1.0987756, -0.8160590, -1.1017333, -0.8158586, -0.0584469, 0.0612036
4: -0.6259944, -0.4620975, -0.6261470, -0.4599898, -0.0399949, 0.0380037
5: -0.0544708, 0.2251788, -0.0570444, 0.2253776, -0.0618024, 0.0641819
6: -4.1227942, -3.6129088, -4.1228380, -3.6124630, -0.1181771, 0.1176580
7: 1.2759156, 1.5145390, 1.2755953, 1.5168469, -0.0264291, 0.0246284
8: -6.2260494, -5.8323154, -6.2275329, -5.8321977, -0.1075676, 0.1089134
9: -5.3917713, -4.9299679, -5.3929701, -4.9298573, -0.1506674, 0.1519379

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0123806, upper bound: 0.0128890
time: 2.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124129, upper bound: 0.0128889
time: 2.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.6819625, -3.2782252, -3.6820629, -3.2756481, -0.1896470, 0.1861681
1: -6.4940100, -5.8343821, -6.4943113, -5.8321881, -0.2429843, 0.2409406
2: -0.4303975, -0.2742175, -0.4305209, -0.2730710, -0.0527043, 0.0514031
3: -1.0987756, -0.8160806, -1.1018691, -0.8158838, -0.0584525, 0.0615084
4: -0.6259944, -0.4620983, -0.6261474, -0.4599897, -0.0399932, 0.0380039
5: -0.0544709, 0.2251521, -0.0571735, 0.2253444, -0.0618087, 0.0644854
6: -4.1227942, -3.6129408, -4.1229343, -3.6125021, -0.1181821, 0.1178176
7: 1.2759157, 1.5145448, 1.2754859, 1.5168555, -0.0264322, 0.0248663
8: -6.2260766, -5.8323154, -6.2275720, -5.8318510, -0.1081077, 0.1089361
9: -5.3917427, -4.9299679, -5.3929386, -4.9297476, -0.1507514, 0.1519382

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0123922, upper bound: 0.0128886
time: 2.66 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124244, upper bound: 0.0128893
time: 2.62 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.6820047, -3.2781332, -3.6821175, -3.2760868, -0.1889956, 0.1863715
1: -6.4940982, -5.8340693, -6.4943399, -5.8324690, -0.2426529, 0.2412814
2: -0.4304336, -0.2741406, -0.4305153, -0.2730459, -0.0527276, 0.0514889
3: -1.0988314, -0.8160372, -1.1017333, -0.8158458, -0.0585447, 0.0612169
4: -0.6260155, -0.4620622, -0.6261622, -0.4599899, -0.0400025, 0.0380679
5: -0.0545017, 0.2251885, -0.0570486, 0.2253852, -0.0618447, 0.0641948
6: -4.1228552, -3.6129024, -4.1228604, -3.6124587, -0.1182429, 0.1177174
7: 1.2756541, 1.5146325, 1.2754115, 1.5168469, -0.0266590, 0.0249448
8: -6.2260971, -5.8316679, -6.2275333, -5.8317480, -0.1087487, 0.1098029
9: -5.3918629, -4.9293509, -5.3929691, -4.9294100, -0.1512735, 0.1525140

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124348, upper bound: 0.0128888
time: 2.60 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124671, upper bound: 0.0128889
time: 2.54 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.6819587, -3.2781336, -3.6820629, -3.2756176, -0.1898255, 0.1863849
1: -6.4940710, -5.8340693, -6.4943123, -5.8319807, -0.2434229, 0.2413074
2: -0.4304336, -0.2741627, -0.4305437, -0.2730713, -0.0527241, 0.0515118
3: -1.0988312, -0.8160588, -1.1018691, -0.8158711, -0.0585503, 0.0615218
4: -0.6260155, -0.4620631, -0.6261625, -0.4599897, -0.0400008, 0.0380681
5: -0.0545018, 0.2251617, -0.0571774, 0.2253520, -0.0618511, 0.0644982
6: -4.1228552, -3.6129348, -4.1229587, -3.6124964, -0.1182479, 0.1178771
7: 1.2756542, 1.5146383, 1.2753018, 1.5168555, -0.0266621, 0.0251828
8: -6.2261238, -5.8316679, -6.2275724, -5.8314013, -0.1092888, 0.1098256
9: -5.3918352, -4.9293509, -5.3929377, -4.9293008, -0.1513574, 0.1525142

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124464, upper bound: 0.0128895
time: 2.49 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124786, upper bound: 0.0128887
time: 2.73 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.6820636, -3.2781847, -3.6827531, -3.2767303, -0.1885866, 0.1871904
1: -6.4941988, -5.8342676, -6.4959745, -5.8325644, -0.2421855, 0.2422609
2: -0.4303999, -0.2741924, -0.4304909, -0.2730579, -0.0524091, 0.0510907
3: -1.0988299, -0.8160225, -1.1016768, -0.8156976, -0.0587296, 0.0611686
4: -0.6260031, -0.4620908, -0.6262494, -0.4604620, -0.0400597, 0.0386638
5: -0.0545145, 0.2252102, -0.0546523, 0.2254194, -0.0647239, 0.0645374
6: -4.1229692, -3.6127787, -4.1221738, -3.6148791, -0.1188131, 0.1198524
7: 1.2756851, 1.5148463, 1.2786058, 1.5159761, -0.0267046, 0.0257790
8: -6.2260737, -5.8322835, -6.2282839, -5.8323493, -0.1074289, 0.1097633
9: -5.3919592, -4.9298315, -5.3917556, -4.9300666, -0.1517619, 0.1517374

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124076, upper bound: 0.0126852
time: 42.13 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124399, upper bound: 0.0128888
time: 2.64 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.6820176, -3.2781856, -3.6826982, -3.2762604, -0.1894169, 0.1872039
1: -6.4941711, -5.8342676, -6.4959474, -5.8320756, -0.2429554, 0.2422867
2: -0.4303999, -0.2742144, -0.4305193, -0.2730831, -0.0524056, 0.0511136
3: -1.0988299, -0.8160440, -1.1018133, -0.8157228, -0.0587352, 0.0614734
4: -0.6260031, -0.4620914, -0.6262499, -0.4604617, -0.0400581, 0.0386640
5: -0.0545144, 0.2251834, -0.0547813, 0.2253863, -0.0647302, 0.0648410
6: -4.1229692, -3.6128104, -4.1222706, -3.6149168, -0.1188180, 0.1200120
7: 1.2756851, 1.5148520, 1.2784961, 1.5159848, -0.0267203, 0.0260171
8: -6.2261019, -5.8322835, -6.2283211, -5.8320017, -0.1079690, 0.1097859
9: -5.3919315, -4.9298315, -5.3917246, -4.9299574, -0.1518457, 0.1517377

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124191, upper bound: 0.0128889
time: 2.61 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124515, upper bound: 0.0128888
time: 2.58 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.6820595, -3.2780936, -3.6827531, -3.2766991, -0.1887653, 0.1874073
1: -6.4942589, -5.8339543, -6.4959750, -5.8323574, -0.2426243, 0.2426273
2: -0.4304359, -0.2741377, -0.4305137, -0.2730583, -0.0524289, 0.0511994
3: -1.0988855, -0.8160003, -1.1016771, -0.8156850, -0.0588274, 0.0611819
4: -0.6260241, -0.4620555, -0.6262645, -0.4604621, -0.0400673, 0.0387280
5: -0.0545454, 0.2252202, -0.0546565, 0.2254273, -0.0647663, 0.0645503
6: -4.1230292, -3.6127725, -4.1221972, -3.6148746, -0.1188788, 0.1199119
7: 1.2754234, 1.5149395, 1.2784220, 1.5159761, -0.0269458, 0.0260955
8: -6.2261209, -5.8316355, -6.2282829, -5.8318996, -0.1086100, 0.1106527
9: -5.3920517, -4.9292154, -5.3917561, -4.9296198, -0.1523678, 0.1523134

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124618, upper bound: 0.0128889
time: 31.40 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124940, upper bound: 0.0128887
time: 2.54 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.6820138, -3.2780945, -3.6826982, -3.2762296, -0.1895955, 0.1874208
1: -6.4942317, -5.8339543, -6.4959469, -5.8318691, -0.2433941, 0.2426533
2: -0.4304359, -0.2741599, -0.4305421, -0.2730834, -0.0524253, 0.0512222
3: -1.0988854, -0.8160220, -1.1018132, -0.8157103, -0.0588330, 0.0614867
4: -0.6260241, -0.4620566, -0.6262648, -0.4604620, -0.0400657, 0.0387281
5: -0.0545453, 0.2251934, -0.0547856, 0.2253940, -0.0647725, 0.0648538
6: -4.1230292, -3.6128039, -4.1222944, -3.6149118, -0.1188839, 0.1200716
7: 1.2754235, 1.5149454, 1.2783124, 1.5159848, -0.0269615, 0.0263335
8: -6.2261486, -5.8316355, -6.2283211, -5.8315525, -0.1091501, 0.1106753
9: -5.3920236, -4.9292154, -5.3917236, -4.9295101, -0.1524517, 0.1523135

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124733, upper bound: 0.0128889
time: 2.59 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0125056, upper bound: 0.0128888
time: 2.83 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.6820636, -3.2781847, -3.6821175, -3.2760837, -0.1889321, 0.1861594
1: -6.4941988, -5.8342676, -6.4943399, -5.8325777, -0.2425681, 0.2409285
2: -0.4303999, -0.2741924, -0.4304941, -0.2730458, -0.0527089, 0.0513874
3: -1.0988299, -0.8160225, -1.1017333, -0.8158263, -0.0585851, 0.0612091
4: -0.6260031, -0.4620908, -0.6261470, -0.4599843, -0.0400079, 0.0380045
5: -0.0545145, 0.2252102, -0.0570444, 0.2254043, -0.0618981, 0.0641860
6: -4.1229692, -3.6127787, -4.1228380, -3.6123569, -0.1185138, 0.1176741
7: 1.2756851, 1.5148463, 1.2754056, 1.5168469, -0.0264561, 0.0252112
8: -6.2260737, -5.8322835, -6.2275534, -5.8321977, -0.1075708, 0.1089649
9: -5.3919592, -4.9298315, -5.3929701, -4.9297423, -0.1510701, 0.1519543

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124076, upper bound: 0.0128889
time: 2.54 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124399, upper bound: 0.0128891
time: 2.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.6820176, -3.2781856, -3.6820629, -3.2756150, -0.1897618, 0.1861730
1: -6.4941711, -5.8342676, -6.4943113, -5.8320894, -0.2433382, 0.2409544
2: -0.4303999, -0.2742144, -0.4305224, -0.2730710, -0.0527053, 0.0514103
3: -1.0988299, -0.8160440, -1.1018691, -0.8158518, -0.0585907, 0.0615139
4: -0.6260031, -0.4620914, -0.6261474, -0.4599842, -0.0400063, 0.0380047
5: -0.0545144, 0.2251834, -0.0571735, 0.2253712, -0.0619044, 0.0644894
6: -4.1229692, -3.6128104, -4.1229343, -3.6123958, -0.1185188, 0.1178337
7: 1.2756851, 1.5148520, 1.2752960, 1.5168555, -0.0264592, 0.0254492
8: -6.2261019, -5.8322835, -6.2275915, -5.8318510, -0.1081110, 0.1089876
9: -5.3919315, -4.9298315, -5.3929386, -4.9296331, -0.1511540, 0.1519545

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124191, upper bound: 0.0128889
time: 2.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124514, upper bound: 0.0128888
time: 2.59 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.6820595, -3.2780936, -3.6821175, -3.2760527, -0.1891104, 0.1863763
1: -6.4942589, -5.8339543, -6.4943399, -5.8323708, -0.2430068, 0.2412953
2: -0.4304359, -0.2741377, -0.4305168, -0.2730459, -0.0527287, 0.0514961
3: -1.0988855, -0.8160003, -1.1017333, -0.8158140, -0.0586828, 0.0612224
4: -0.6260241, -0.4620555, -0.6261622, -0.4599845, -0.0400155, 0.0380687
5: -0.0545454, 0.2252202, -0.0570486, 0.2254121, -0.0619405, 0.0641988
6: -4.1230292, -3.6127725, -4.1228604, -3.6123524, -0.1185796, 0.1177335
7: 1.2754234, 1.5149395, 1.2752217, 1.5168469, -0.0266859, 0.0255277
8: -6.2261209, -5.8316355, -6.2275534, -5.8317480, -0.1087520, 0.1098545
9: -5.3920517, -4.9292154, -5.3929691, -4.9292955, -0.1516761, 0.1525303

Time for backsubstitution: 5.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124618, upper bound: 0.0128895
time: 2.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124941, upper bound: 0.0128887
time: 2.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.6820138, -3.2780945, -3.6820629, -3.2755837, -0.1899403, 0.1863897
1: -6.4942317, -5.8339543, -6.4943123, -5.8318830, -0.2437770, 0.2413214
2: -0.4304359, -0.2741599, -0.4305452, -0.2730713, -0.0527251, 0.0515189
3: -1.0988854, -0.8160220, -1.1018691, -0.8158392, -0.0586884, 0.0615273
4: -0.6260241, -0.4620566, -0.6261625, -0.4599842, -0.0400139, 0.0380689
5: -0.0545453, 0.2251934, -0.0571774, 0.2253789, -0.0619468, 0.0645023
6: -4.1230292, -3.6128039, -4.1229587, -3.6123900, -0.1185845, 0.1178932
7: 1.2754235, 1.5149454, 1.2751122, 1.5168555, -0.0266890, 0.0257656
8: -6.2261486, -5.8316355, -6.2275915, -5.8314009, -0.1092921, 0.1098771
9: -5.3920236, -4.9292154, -5.3929377, -4.9291863, -0.1517600, 0.1525304

Time for backsubstitution: 5.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 807
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
type: B, layer: 1, pos: 346

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0124734, upper bound: 0.0128889
time: 2.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0125055, upper bound: 0.0128888
time: 2.70 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 11.00 seconds
NS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0123806, upper bound: 0.0128890
NS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124129, upper bound: 0.0128888
NS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0123922, upper bound: 0.0128891
NS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124245, upper bound: 0.0128893
NS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124349, upper bound: 0.0128891
NS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124671, upper bound: 0.0128894
NS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124465, upper bound: 0.0128891
NS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124787, upper bound: 0.0128894
NS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0123806, upper bound: 0.0128890
NS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124129, upper bound: 0.0128889
NS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0123922, upper bound: 0.0128886
NS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124244, upper bound: 0.0128893
NS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124348, upper bound: 0.0128888
NS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124671, upper bound: 0.0128889
NS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124464, upper bound: 0.0128895
NS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124786, upper bound: 0.0128887
NS_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124076, upper bound: 0.0126852
NS_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124399, upper bound: 0.0128888
NS_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124191, upper bound: 0.0128889
NS_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124515, upper bound: 0.0128888
NS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124618, upper bound: 0.0128889
NS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124940, upper bound: 0.0128887
NS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124733, upper bound: 0.0128889
NS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0125056, upper bound: 0.0128888
NS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124076, upper bound: 0.0128889
NS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124399, upper bound: 0.0128891
NS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124191, upper bound: 0.0128889
NS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124514, upper bound: 0.0128888
NS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124618, upper bound: 0.0128895
NS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124941, upper bound: 0.0128887
NS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0124734, upper bound: 0.0128889
NS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.00
Output dim: 7, lower bound: -0.0125055, upper bound: 0.0128888

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 25.40 + 1020.11 = 1045.51 seconds
