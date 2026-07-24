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
execution time: IAR + RelationalAnalysis = 8.24 + 17.71 = 25.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0129078, upper bound: 0.0129076

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3020

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3595

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129077, upper bound: 0.0129074
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129077, upper bound: 0.0129074
time: 4.17 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.69 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.69
Output dim: 7, lower bound: -0.0129077, upper bound: 0.0129074
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.69
Output dim: 7, lower bound: -0.0129077, upper bound: 0.0129074

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1869113, 0.1869113
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2422895, 0.2422895
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513033, 0.0513033
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0587083, 0.0587083
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0381045, 0.0381045
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0624915, 0.0624915
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1184303, 0.1184303
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257999, 0.0257999
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1088091, 0.1088091
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1513630, 0.1513630

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2330

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129067, upper bound: 0.0128991
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128993, upper bound: 0.0129065
time: 2.48 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1869113, 0.1869113
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2422895, 0.2422895
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513033, 0.0513033
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0587083, 0.0587083
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0381045, 0.0381045
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0624915, 0.0624915
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1184303, 0.1184303
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257999, 0.0257999
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1088091, 0.1088091
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1513630, 0.1513630

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 670

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2835

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129007, upper bound: 0.0129050
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129051, upper bound: 0.0129006
time: 2.68 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 11.79 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 11.79
Output dim: 7, lower bound: -0.0129067, upper bound: 0.0128991
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 11.79
Output dim: 7, lower bound: -0.0128993, upper bound: 0.0129065
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 11.79
Output dim: 7, lower bound: -0.0129007, upper bound: 0.0129050
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 11.79
Output dim: 7, lower bound: -0.0129051, upper bound: 0.0129006

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1868531, 0.1867725
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2421421, 0.2419653
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513028, 0.0513027
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0586408, 0.0586514
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0381010, 0.0381021
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0624108, 0.0623909
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1183822, 0.1184204
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257989, 0.0257986
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1086794, 0.1087101
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1512541, 0.1512110

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3349

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2566

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129063, upper bound: 0.0128987
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129062, upper bound: 0.0128989
time: 2.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1867725, 0.1868531
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2419653, 0.2421421
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513027, 0.0513028
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0586514, 0.0586408
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0381021, 0.0381010
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0623909, 0.0624108
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1184204, 0.1183822
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257986, 0.0257989
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1087101, 0.1086794
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1512110, 0.1512541

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2984

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2524

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128971, upper bound: 0.0129022
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128946, upper bound: 0.0129047
time: 2.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1868808, 0.1868751
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2422316, 0.2422130
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512971, 0.0512973
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0586902, 0.0586885
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0381031, 0.0381032
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0624726, 0.0624704
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1184162, 0.1184161
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257876, 0.0257901
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1087572, 0.1087315
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1512976, 0.1512806

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128982, upper bound: 0.0129019
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128975, upper bound: 0.0129025
time: 2.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1868751, 0.1868808
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2422131, 0.2422316
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512973, 0.0512971
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0586885, 0.0586902
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0381032, 0.0381031
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0624704, 0.0624726
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1184161, 0.1184162
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257901, 0.0257876
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1087315, 0.1087572
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1512807, 0.1512976

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3020

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 99

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129038, upper bound: 0.0128998
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129040, upper bound: 0.0128995
time: 2.66 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 11.73 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 11.73
Output dim: 7, lower bound: -0.0129063, upper bound: 0.0128987
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 11.73
Output dim: 7, lower bound: -0.0129062, upper bound: 0.0128989
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 11.73
Output dim: 7, lower bound: -0.0128971, upper bound: 0.0129022
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 11.73
Output dim: 7, lower bound: -0.0128946, upper bound: 0.0129047
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 11.73
Output dim: 7, lower bound: -0.0128982, upper bound: 0.0129019
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 11.73
Output dim: 7, lower bound: -0.0128975, upper bound: 0.0129025
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 11.73
Output dim: 7, lower bound: -0.0129038, upper bound: 0.0128998
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 11.73
Output dim: 7, lower bound: -0.0129040, upper bound: 0.0128995

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859318, 0.1858660
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2413584, 0.2411996
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511115, 0.0511064
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585745, 0.0585864
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380728, 0.0380721
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622810, 0.0622624
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179283, 0.1179622
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257911, 0.0257906
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1057898, 0.1058999
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497157, 0.1497145

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3038

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3372

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129062, upper bound: 0.0128989
time: 35.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129064, upper bound: 0.0128985
time: 2.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859466, 0.1858512
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2413765, 0.2411816
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511065, 0.0511114
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585757, 0.0585851
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380711, 0.0380738
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622823, 0.0622611
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179240, 0.1179665
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257909, 0.0257908
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1058692, 0.1058204
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497575, 0.1496726

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 666

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 661

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129063, upper bound: 0.0128992
time: 25.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129063, upper bound: 0.0128992
time: 3.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1867663, 0.1868510
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2419440, 0.2421200
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513027, 0.0513028
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0586413, 0.0586279
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0381020, 0.0381009
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0623783, 0.0623960
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1184201, 0.1183821
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257973, 0.0257960
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1087021, 0.1086694
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1511974, 0.1512406

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2540

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128955, upper bound: 0.0129013
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128959, upper bound: 0.0129007
time: 17.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1867725, 0.1868469
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2419653, 0.2421207
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513027, 0.0513028
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0586385, 0.0586408
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0381020, 0.0381010
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0623761, 0.0624108
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1184204, 0.1183819
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257986, 0.0257976
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1087101, 0.1086714
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1512110, 0.1512405

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 838

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128934, upper bound: 0.0128936
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128839, upper bound: 0.0129031
time: 2.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1866929, 0.1866864
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2421105, 0.2420833
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512936, 0.0512943
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0586594, 0.0586577
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380516, 0.0380514
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0624409, 0.0624390
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1182603, 0.1182617
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257429, 0.0257448
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1085048, 0.1084776
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1512862, 0.1512673

Time for backsubstitution: 6.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 269

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3020

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128972, upper bound: 0.0129010
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128975, upper bound: 0.0129008
time: 2.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1866921, 0.1866872
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2421018, 0.2420920
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512941, 0.0512938
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0586595, 0.0586577
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380514, 0.0380516
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0624412, 0.0624387
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1182617, 0.1182602
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257424, 0.0257453
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1085032, 0.1084791
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1512843, 0.1512693

Time for backsubstitution: 6.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3001

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2125

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0127721, upper bound: 0.0127771
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0127721, upper bound: 0.0127771
time: 2.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1864696, 0.1863979
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2412697, 0.2411065
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512799, 0.0512787
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585402, 0.0585665
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380563, 0.0380556
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0623100, 0.0623369
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179876, 0.1180449
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257816, 0.0257794
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1085638, 0.1085653
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1510343, 0.1510059

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2847

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129036, upper bound: 0.0128990
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0128996
time: 5.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1863922, 0.1864752
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2410880, 0.2412882
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512788, 0.0512798
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585648, 0.0585420
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380556, 0.0380563
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0623347, 0.0623122
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1180448, 0.1179877
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257818, 0.0257791
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1085397, 0.1085894
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1509890, 0.1510512

Time for backsubstitution: 6.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3359

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 371

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128934, upper bound: 0.0128891
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128933, upper bound: 0.0128892
time: 4.05 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 14.84 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0129062, upper bound: 0.0128989
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0129064, upper bound: 0.0128985
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0129063, upper bound: 0.0128992
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0129063, upper bound: 0.0128992
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0128955, upper bound: 0.0129013
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0128959, upper bound: 0.0129007
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0128934, upper bound: 0.0128936
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0128839, upper bound: 0.0129031
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0128972, upper bound: 0.0129010
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0128975, upper bound: 0.0129008
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0127721, upper bound: 0.0127771
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0127721, upper bound: 0.0127771
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0129036, upper bound: 0.0128990
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0128996
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0128934, upper bound: 0.0128891
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 14.84
Output dim: 7, lower bound: -0.0128933, upper bound: 0.0128892

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859318, 0.1858660
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2413584, 0.2411996
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511115, 0.0511064
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585745, 0.0585864
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380728, 0.0380721
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622810, 0.0622624
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179283, 0.1179622
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257911, 0.0257906
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1057898, 0.1058999
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497157, 0.1497145

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2283

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129046, upper bound: 0.0128882
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128959, upper bound: 0.0128976
time: 4.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859318, 0.1858660
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2413584, 0.2411996
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511115, 0.0511064
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585745, 0.0585864
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380728, 0.0380721
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622810, 0.0622624
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179283, 0.1179622
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257911, 0.0257906
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1057898, 0.1058999
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497157, 0.1497145

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 768

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2984

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129063, upper bound: 0.0128988
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129062, upper bound: 0.0128985
time: 17.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859466, 0.1858512
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2413765, 0.2411816
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511065, 0.0511114
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585757, 0.0585851
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380711, 0.0380738
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622823, 0.0622611
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179240, 0.1179665
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257909, 0.0257908
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1058692, 0.1058204
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497575, 0.1496726

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2531

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129018, upper bound: 0.0128828
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128902, upper bound: 0.0128945
time: 2.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859466, 0.1858512
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2413765, 0.2411816
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511065, 0.0511114
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585757, 0.0585851
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380711, 0.0380738
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622823, 0.0622611
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179240, 0.1179665
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257909, 0.0257908
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1058692, 0.1058204
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497575, 0.1496726

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3372

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2634

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129035, upper bound: 0.0128973
time: 5.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129040, upper bound: 0.0128968
time: 3.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1860579, 0.1862097
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2401993, 0.2404659
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513025, 0.0513027
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582761, 0.0582467
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380855, 0.0380857
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0619451, 0.0619435
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1182137, 0.1181727
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257909, 0.0257898
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1081345, 0.1081307
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505049, 0.1505893

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2634

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128931, upper bound: 0.0128977
time: 5.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128924, upper bound: 0.0128986
time: 9.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1861172, 0.1861426
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2402822, 0.2403754
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513026, 0.0513027
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582601, 0.0582627
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380868, 0.0380844
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0619258, 0.0619637
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1182108, 0.1181738
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257911, 0.0257895
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1081616, 0.1081017
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505469, 0.1505482

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 667

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128959, upper bound: 0.0129006
time: 23.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128959, upper bound: 0.0129003
time: 13.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1865892, 0.1867180
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2415795, 0.2418735
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512988, 0.0513013
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0586130, 0.0585731
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380926, 0.0380851
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0623448, 0.0623366
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1183327, 0.1182516
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257977, 0.0257967
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1086285, 0.1085776
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1511265, 0.1511461

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2995

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 667

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128840, upper bound: 0.0129038
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128840, upper bound: 0.0129038
time: 2.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1800234, 0.1797109
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2290857, 0.2285907
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512689, 0.0512663
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0561073, 0.0561894
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378669, 0.0378741
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0594767, 0.0595288
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1161128, 0.1162151
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257169, 0.0257188
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1043810, 0.1042832
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1461934, 0.1460316

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2847

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128972, upper bound: 0.0129007
time: 12.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128966, upper bound: 0.0129014
time: 11.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1797173, 0.1800169
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2286179, 0.2290585
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512657, 0.0512696
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0561911, 0.0561056
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378743, 0.0378667
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0595306, 0.0594748
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1162137, 0.1161142
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257169, 0.0257188
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1043104, 0.1043538
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1460504, 0.1461746

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2995

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128932, upper bound: 0.0129007
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128970, upper bound: 0.0128967
time: 3.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1864606, 0.1863887
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2412624, 0.2410973
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512749, 0.0512729
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585226, 0.0585483
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380536, 0.0380529
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622899, 0.0623162
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179829, 0.1180392
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257809, 0.0257783
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1085611, 0.1085628
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1510324, 0.1510030

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 269

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2271

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129030, upper bound: 0.0128984
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129030, upper bound: 0.0128984
time: 2.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1864603, 0.1863889
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2412605, 0.2410994
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512741, 0.0512736
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585220, 0.0585488
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380536, 0.0380529
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622892, 0.0623168
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179820, 0.1180402
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257805, 0.0257787
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1085611, 0.1085628
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1510314, 0.1510041

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3448

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2566

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129025, upper bound: 0.0128989
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129024, upper bound: 0.0128991
time: 2.74 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 11.98 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0129046, upper bound: 0.0128882
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0128959, upper bound: 0.0128976
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0129063, upper bound: 0.0128988
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0129062, upper bound: 0.0128985
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0129018, upper bound: 0.0128828
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0128902, upper bound: 0.0128945
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0129035, upper bound: 0.0128973
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0129040, upper bound: 0.0128968
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0128931, upper bound: 0.0128977
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0128924, upper bound: 0.0128986
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0128959, upper bound: 0.0129006
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0128959, upper bound: 0.0129003
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0128840, upper bound: 0.0129038
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0128840, upper bound: 0.0129038
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0128972, upper bound: 0.0129007
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0128966, upper bound: 0.0129014
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0128932, upper bound: 0.0129007
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0128970, upper bound: 0.0128967
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0129030, upper bound: 0.0128984
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0129030, upper bound: 0.0128984
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0129025, upper bound: 0.0128989
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.98
Output dim: 7, lower bound: -0.0129024, upper bound: 0.0128991

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1825505, 0.1824663
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2335020, 0.2334203
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511116, 0.0511064
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0573994, 0.0574488
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0376978, 0.0377006
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610240, 0.0610401
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1150333, 0.1151242
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257746, 0.0257735
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1043761, 0.1045160
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1479048, 0.1479660

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2038

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3448

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129046, upper bound: 0.0128876
time: 4.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129033, upper bound: 0.0128884
time: 2.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1825321, 0.1824847
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2335790, 0.2333432
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511116, 0.0511065
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0574370, 0.0574113
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0377013, 0.0376972
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610587, 0.0610053
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1150903, 0.1150672
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257741, 0.0257741
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1044059, 0.1044862
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1479672, 0.1479036

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 670

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2275

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128954, upper bound: 0.0128886
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128878, upper bound: 0.0128972
time: 3.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859318, 0.1858660
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2413584, 0.2411996
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511115, 0.0511064
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585745, 0.0585864
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380728, 0.0380721
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622810, 0.0622624
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179283, 0.1179622
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257911, 0.0257906
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1057898, 0.1058999
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497157, 0.1497145

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3577

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 670

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0128989
time: 13.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0128981
time: 19.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859318, 0.1858660
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2413584, 0.2411996
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511115, 0.0511064
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585745, 0.0585864
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380728, 0.0380721
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622810, 0.0622624
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179283, 0.1179622
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257911, 0.0257906
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1057898, 0.1058999
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497157, 0.1497145

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2690

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2130

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129054, upper bound: 0.0128984
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129054, upper bound: 0.0128979
time: 6.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1856509, 0.1855708
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2410343, 0.2408655
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510877, 0.0510920
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0584673, 0.0584708
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380693, 0.0380720
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0621737, 0.0621461
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1178378, 0.1178736
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257113, 0.0257060
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1056572, 0.1056314
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496856, 0.1496055

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 670

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129016, upper bound: 0.0128832
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129016, upper bound: 0.0128832
time: 4.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1852294, 0.1851597
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2394158, 0.2393069
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0508008, 0.0507883
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585056, 0.0585147
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378821, 0.0378720
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0621047, 0.0620785
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179117, 0.1179540
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257158, 0.0257160
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1039620, 0.1039889
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1483721, 0.1483474

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3180

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2524

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129013, upper bound: 0.0128926
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128989, upper bound: 0.0128948
time: 6.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1852551, 0.1851341
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2395018, 0.2392210
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0507833, 0.0508057
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585053, 0.0585150
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378692, 0.0378848
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0620997, 0.0620835
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179116, 0.1179541
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257161, 0.0257158
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1040376, 0.1039133
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1484323, 0.1482871

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2516

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3072

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129040, upper bound: 0.0128955
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129034, upper bound: 0.0128961
time: 2.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1858701, 0.1860210
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2400783, 0.2403363
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512991, 0.0512998
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582453, 0.0582159
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380339, 0.0380340
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0619135, 0.0619122
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1180577, 0.1180182
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257461, 0.0257445
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1078820, 0.1078768
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1504935, 0.1505759

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 806

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128917, upper bound: 0.0128968
time: 2.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128920, upper bound: 0.0128964
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1858692, 0.1860218
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2400696, 0.2403450
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512996, 0.0512992
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582453, 0.0582159
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380337, 0.0380342
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0619138, 0.0619119
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1180592, 0.1180167
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257456, 0.0257450
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1078805, 0.1078783
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1504915, 0.1505778

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2996

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 347

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0127634, upper bound: 0.0128959
time: 26.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128896, upper bound: 0.0127698
time: 8.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1861172, 0.1861426
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2402822, 0.2403754
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513026, 0.0513027
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582601, 0.0582627
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380868, 0.0380844
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0619258, 0.0619637
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1182108, 0.1181738
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257911, 0.0257895
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1081616, 0.1081017
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505469, 0.1505482

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 269

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2690

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128956, upper bound: 0.0129004
time: 2.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128955, upper bound: 0.0129006
time: 2.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1861172, 0.1861426
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2402822, 0.2403754
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513026, 0.0513027
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582601, 0.0582627
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380868, 0.0380844
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0619258, 0.0619637
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1182108, 0.1181738
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257911, 0.0257895
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1081616, 0.1081017
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505469, 0.1505482

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2531

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2476

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128940, upper bound: 0.0128980
time: 46.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128939, upper bound: 0.0128991
time: 2.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1865892, 0.1867180
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2415795, 0.2418735
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512988, 0.0513013
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0586130, 0.0585731
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380926, 0.0380851
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0623448, 0.0623366
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1183327, 0.1182516
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257977, 0.0257967
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1086285, 0.1085776
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1511265, 0.1511461

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 269

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 838

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128832, upper bound: 0.0129030
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128835, upper bound: 0.0129025
time: 2.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1865892, 0.1867180
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2415795, 0.2418735
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512988, 0.0513013
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0586130, 0.0585731
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380926, 0.0380851
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0623448, 0.0623366
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1183327, 0.1182516
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257977, 0.0257967
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1086285, 0.1085776
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1511265, 0.1511461

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2299

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3020

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128828, upper bound: 0.0129025
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128832, upper bound: 0.0129022
time: 2.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1798923, 0.1795666
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2289174, 0.2284188
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512296, 0.0512272
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0560820, 0.0561641
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378346, 0.0378423
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0594227, 0.0594750
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1160620, 0.1161642
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257181, 0.0257199
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1041784, 0.1040794
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1460481, 0.1458807

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2346

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2690

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128968, upper bound: 0.0129009
time: 6.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128968, upper bound: 0.0129003
time: 2.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1798792, 0.1795797
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2289139, 0.2284223
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512297, 0.0512271
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0560819, 0.0561642
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378351, 0.0378418
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0594229, 0.0594748
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1160619, 0.1161644
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257179, 0.0257200
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1041773, 0.1040806
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1460426, 0.1458862

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 278

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128964, upper bound: 0.0126387
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126338, upper bound: 0.0129014
time: 2.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1797178, 0.1797039
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2286171, 0.2283887
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512646, 0.0512652
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0560772, 0.0561074
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378532, 0.0378583
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0594036, 0.0594756
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1160270, 0.1160682
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257134, 0.0257176
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1043101, 0.1042128
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1460500, 0.1459947

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2164

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 806

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128917, upper bound: 0.0129007
time: 75.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128931, upper bound: 0.0128993
time: 7.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1794043, 0.1800169
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2279481, 0.2290585
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512613, 0.0512696
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0561911, 0.0559917
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378743, 0.0378456
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0595306, 0.0593478
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1162137, 0.1159274
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257169, 0.0257153
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1041693, 0.1043538
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1458706, 0.1461746

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2847

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128951, upper bound: 0.0128973
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128969, upper bound: 0.0128955
time: 2.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1864610, 0.1863863
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2412672, 0.2410889
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512745, 0.0512726
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585220, 0.0585480
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380535, 0.0380529
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622892, 0.0623162
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179821, 0.1180407
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257808, 0.0257783
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1085631, 0.1085610
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1510455, 0.1509972

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3001

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128989, upper bound: 0.0128975
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129027, upper bound: 0.0128939
time: 2.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1864582, 0.1863887
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2412540, 0.2410973
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512749, 0.0512725
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585222, 0.0585483
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380535, 0.0380529
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622899, 0.0623155
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179829, 0.1180384
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257809, 0.0257783
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1085594, 0.1085628
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1510266, 0.1510030

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2071

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3258

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129024, upper bound: 0.0128934
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128981, upper bound: 0.0128977
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1855392, 0.1854825
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2404767, 0.2403337
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510828, 0.0510773
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0584557, 0.0584837
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380254, 0.0380229
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0621595, 0.0621883
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1175281, 0.1175820
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257727, 0.0257707
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1056715, 0.1057526
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1494929, 0.1495075

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 269

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0128988
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0128989
time: 14.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1855540, 0.1854677
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2404948, 0.2403157
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510778, 0.0510823
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0584569, 0.0584825
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380237, 0.0380246
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0621608, 0.0621870
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1175238, 0.1175863
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257724, 0.0257709
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1057509, 0.1056731
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1495348, 0.1494656

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2283

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2089

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129012, upper bound: 0.0128987
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129020, upper bound: 0.0128979
time: 2.63 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 11.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129046, upper bound: 0.0128876
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129033, upper bound: 0.0128884
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128954, upper bound: 0.0128886
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128878, upper bound: 0.0128972
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0128989
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0128981
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129054, upper bound: 0.0128984
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129054, upper bound: 0.0128979
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129016, upper bound: 0.0128832
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129016, upper bound: 0.0128832
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129013, upper bound: 0.0128926
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128989, upper bound: 0.0128948
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129040, upper bound: 0.0128955
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129034, upper bound: 0.0128961
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128917, upper bound: 0.0128968
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128920, upper bound: 0.0128964
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0127634, upper bound: 0.0128959
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128896, upper bound: 0.0127698
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128956, upper bound: 0.0129004
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128955, upper bound: 0.0129006
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128940, upper bound: 0.0128980
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128939, upper bound: 0.0128991
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128832, upper bound: 0.0129030
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128835, upper bound: 0.0129025
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128828, upper bound: 0.0129025
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128832, upper bound: 0.0129022
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128968, upper bound: 0.0129009
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128968, upper bound: 0.0129003
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128964, upper bound: 0.0126387
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0126338, upper bound: 0.0129014
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128917, upper bound: 0.0129007
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128931, upper bound: 0.0128993
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128951, upper bound: 0.0128973
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128969, upper bound: 0.0128955
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128989, upper bound: 0.0128975
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129027, upper bound: 0.0128939
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129024, upper bound: 0.0128934
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0128981, upper bound: 0.0128977
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0128988
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0128989
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129012, upper bound: 0.0128987
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.70
Output dim: 7, lower bound: -0.0129020, upper bound: 0.0128979

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1824243, 0.1823491
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2333906, 0.2332937
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0509377, 0.0509530
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0572715, 0.0573363
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0370997, 0.0370249
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0606979, 0.0607533
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1149466, 0.1150480
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257121, 0.0257029
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1037320, 0.1037861
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1477619, 0.1478052

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 799

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129043, upper bound: 0.0128865
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129037, upper bound: 0.0128871
time: 2.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1824333, 0.1823401
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2333753, 0.2333087
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0509582, 0.0509325
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0572869, 0.0573209
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0370221, 0.0371025
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0607371, 0.0607141
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1149571, 0.1150375
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257040, 0.0257110
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1036462, 0.1038719
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1477439, 0.1478232

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 670

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129032, upper bound: 0.0128883
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129032, upper bound: 0.0128883
time: 2.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1824759, 0.1824294
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2335699, 0.2333338
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510859, 0.0510888
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0574277, 0.0574032
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0376931, 0.0376875
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610471, 0.0609941
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1150617, 0.1150458
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257705, 0.0257702
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1043499, 0.1044283
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1479551, 0.1478882

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3006

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 670

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128954, upper bound: 0.0128883
time: 7.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128954, upper bound: 0.0128894
time: 15.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1824769, 0.1824285
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2335696, 0.2333342
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510939, 0.0510808
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0574289, 0.0574019
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0376916, 0.0376890
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610475, 0.0609937
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1150689, 0.1150386
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257702, 0.0257705
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1043480, 0.1044302
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1479518, 0.1478914

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2271

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2125

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0127623, upper bound: 0.0127718
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0127623, upper bound: 0.0127718
time: 2.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859318, 0.1858660
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2413584, 0.2411996
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511115, 0.0511064
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585745, 0.0585864
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380728, 0.0380721
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622810, 0.0622624
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179283, 0.1179622
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257911, 0.0257906
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1057898, 0.1058999
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497157, 0.1497145

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2164

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3356

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128972, upper bound: 0.0128990
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129062, upper bound: 0.0128900
time: 3.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859318, 0.1858660
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2413584, 0.2411996
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511115, 0.0511064
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585745, 0.0585864
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380728, 0.0380721
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622810, 0.0622624
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179283, 0.1179622
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257911, 0.0257906
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1057898, 0.1058999
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497157, 0.1497145

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2180

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129016, upper bound: 0.0128978
time: 5.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129054, upper bound: 0.0128933
time: 89.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1852110, 0.1851131
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2406971, 0.2405026
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0509630, 0.0509675
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585468, 0.0585588
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380452, 0.0380457
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622015, 0.0621847
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1175754, 0.1176219
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257797, 0.0257798
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1033781, 0.1033697
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1485194, 0.1484514

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3001

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2346

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129032, upper bound: 0.0128861
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128937, upper bound: 0.0128959
time: 2.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1851788, 0.1851376
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2406614, 0.2405330
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0509709, 0.0509579
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585466, 0.0585586
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380460, 0.0380445
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622025, 0.0621829
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1175845, 0.1176094
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257803, 0.0257792
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1032596, 0.1034610
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1484527, 0.1485046

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2271

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2516

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129031, upper bound: 0.0128858
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0128954
time: 2.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1856509, 0.1855708
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2410343, 0.2408655
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510877, 0.0510920
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0584673, 0.0584708
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380693, 0.0380720
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0621737, 0.0621461
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1178378, 0.1178736
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257113, 0.0257060
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1056572, 0.1056314
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496856, 0.1496055

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2540

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2361

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128991, upper bound: 0.0128811
time: 4.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128993, upper bound: 0.0128809
time: 3.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1856509, 0.1855708
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2410343, 0.2408655
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510877, 0.0510920
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0584673, 0.0584708
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380693, 0.0380720
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0621737, 0.0621461
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1178378, 0.1178736
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257113, 0.0257060
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1056572, 0.1056314
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496856, 0.1496055

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2364

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3573

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128855, upper bound: 0.0128833
time: 4.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129012, upper bound: 0.0128664
time: 42.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1852233, 0.1851577
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2393947, 0.2392850
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0508008, 0.0507882
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0584954, 0.0585018
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378820, 0.0378719
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0620921, 0.0620636
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179115, 0.1179539
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257145, 0.0257131
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1039540, 0.1039788
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1483585, 0.1483340

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3359

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 666

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129013, upper bound: 0.0128919
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129013, upper bound: 0.0128919
time: 2.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1852294, 0.1851536
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2394158, 0.2392858
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0508008, 0.0507883
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0584927, 0.0585147
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378820, 0.0378720
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0620899, 0.0620785
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179117, 0.1179538
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257158, 0.0257147
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1039620, 0.1039808
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1483721, 0.1483338

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2348

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 371

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128882, upper bound: 0.0128831
time: 15.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128880, upper bound: 0.0128843
time: 2.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1852546, 0.1851336
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2395005, 0.2392197
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0507827, 0.0508051
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585019, 0.0585116
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378682, 0.0378838
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0620971, 0.0620809
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179114, 0.1179540
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257156, 0.0257153
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1040351, 0.1039109
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1484309, 0.1482857

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3349

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 666

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129040, upper bound: 0.0128957
time: 2.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129040, upper bound: 0.0128957
time: 2.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1852546, 0.1851336
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2395005, 0.2392195
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0507827, 0.0508051
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585019, 0.0585116
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378682, 0.0378838
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0620971, 0.0620809
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179114, 0.1179539
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257155, 0.0257153
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1040352, 0.1039107
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1484309, 0.1482857

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129020, upper bound: 0.0128963
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129033, upper bound: 0.0128944
time: 2.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1841261, 0.1840805
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2360810, 0.2361159
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512872, 0.0512856
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575489, 0.0576128
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378413, 0.0378624
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0611601, 0.0612656
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1165408, 0.1167020
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257410, 0.0257396
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1071028, 0.1070542
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496290, 0.1496616

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3180

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 278

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128917, upper bound: 0.0126339
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126291, upper bound: 0.0128965
time: 2.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1839296, 0.1842771
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2358580, 0.2363390
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512849, 0.0512878
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0576421, 0.0575195
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378624, 0.0378413
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0612669, 0.0611587
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1167415, 0.1165013
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257412, 0.0257395
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1070594, 0.1070976
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1495791, 0.1497114

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2690

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128918, upper bound: 0.0128963
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128919, upper bound: 0.0128960
time: 2.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1858625, 0.1860161
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2399511, 0.2403506
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513205, 0.0512858
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0581646, 0.0579794
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0379474, 0.0380044
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0618467, 0.0617059
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1180647, 0.1179671
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0256065, 0.0257494
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1078339, 0.1077797
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1504145, 0.1505960

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 670

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 667

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0127635, upper bound: 0.0128958
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0127635, upper bound: 0.0128958
time: 2.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1861172, 0.1861426
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2402822, 0.2403754
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513026, 0.0513027
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582601, 0.0582627
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380868, 0.0380844
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0619258, 0.0619637
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1182108, 0.1181738
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257911, 0.0257895
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1081616, 0.1081017
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505469, 0.1505482

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3577

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128930, upper bound: 0.0128977
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128924, upper bound: 0.0128981
time: 2.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1861172, 0.1861426
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2402822, 0.2403754
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513026, 0.0513027
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582601, 0.0582627
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380868, 0.0380844
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0619258, 0.0619637
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1182108, 0.1181738
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257911, 0.0257895
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1081616, 0.1081017
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505469, 0.1505482

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128937, upper bound: 0.0129000
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128955, upper bound: 0.0128981
time: 2.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859844, 0.1860023
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2401974, 0.2402899
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512981, 0.0512982
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582462, 0.0582496
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380473, 0.0380427
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0619212, 0.0619595
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1181433, 0.1181094
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257795, 0.0257776
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1080532, 0.1079869
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505342, 0.1505355

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2105

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2136

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128924, upper bound: 0.0128977
time: 6.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128933, upper bound: 0.0128972
time: 40.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859769, 0.1860098
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2401967, 0.2402905
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512982, 0.0512982
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582471, 0.0582487
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380451, 0.0380450
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0619216, 0.0619591
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1181465, 0.1181062
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257791, 0.0257780
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1080467, 0.1079934
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505343, 0.1505354

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3150

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 677

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0128975
time: 6.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128928, upper bound: 0.0128981
time: 8.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1865889, 0.1867174
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2415789, 0.2418727
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512987, 0.0513011
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0586104, 0.0585702
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380922, 0.0380847
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0623426, 0.0623339
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1183321, 0.1182508
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257967, 0.0257959
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1086276, 0.1085765
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1511262, 0.1511458

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2347

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2566

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128829, upper bound: 0.0129027
time: 2.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128827, upper bound: 0.0129028
time: 2.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1865886, 0.1867177
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2415787, 0.2418729
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512986, 0.0513012
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0586100, 0.0585706
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380922, 0.0380847
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0623421, 0.0623344
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1183319, 0.1182510
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257969, 0.0257957
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1086274, 0.1085768
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1511263, 0.1511457

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3372

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3349

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128830, upper bound: 0.0129021
time: 6.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128830, upper bound: 0.0129017
time: 89.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1799175, 0.1797402
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2285491, 0.2283757
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512741, 0.0512734
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0560596, 0.0561036
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0379079, 0.0379078
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0593792, 0.0594249
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1161850, 0.1162048
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257717, 0.0257707
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1045028, 0.1043812
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1460328, 0.1459093

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2847

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128827, upper bound: 0.0129017
time: 23.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128819, upper bound: 0.0129029
time: 2.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1796113, 0.1800458
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2280813, 0.2288427
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512708, 0.0512766
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0561433, 0.0560197
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0379153, 0.0379004
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0594330, 0.0593710
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1162862, 0.1161039
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257717, 0.0257707
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1044321, 0.1044519
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1458898, 0.1460519

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2164

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128745, upper bound: 0.0129014
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128822, upper bound: 0.0128937
time: 2.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1798923, 0.1795666
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2289174, 0.2284188
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512296, 0.0512272
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0560820, 0.0561641
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378346, 0.0378423
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0594227, 0.0594750
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1160620, 0.1161642
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257181, 0.0257199
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1041784, 0.1040794
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1460481, 0.1458807

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3356

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2531

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128923, upper bound: 0.0128847
time: 15.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128807, upper bound: 0.0128959
time: 28.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1798923, 0.1795666
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2289174, 0.2284188
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512296, 0.0512272
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0560820, 0.0561641
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378346, 0.0378423
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0594227, 0.0594750
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1160620, 0.1161642
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257181, 0.0257199
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1041784, 0.1040794
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1460481, 0.1458807

Time for backsubstitution: 6.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3001

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 807

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128962, upper bound: 0.0129000
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128966, upper bound: 0.0128998
time: 2.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1796317, 0.1793251
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2285525, 0.2279826
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512188, 0.0512112
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0560337, 0.0561198
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0376449, 0.0376526
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0589157, 0.0588347
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1154118, 0.1156641
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0250768, 0.0248766
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1039872, 0.1038655
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1458632, 0.1456532

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 735

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3573

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128802, upper bound: 0.0126386
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128960, upper bound: 0.0126226
time: 2.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1796246, 0.1793323
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2284741, 0.2280610
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512139, 0.0512162
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0560375, 0.0561161
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0376458, 0.0376517
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0587828, 0.0589676
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1155617, 0.1155142
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0248745, 0.0250789
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1039622, 0.1038905
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1458095, 0.1457069

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 666

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126319, upper bound: 0.0129012
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126338, upper bound: 0.0128994
time: 2.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1797163, 0.1797019
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2286159, 0.2283865
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512640, 0.0512646
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0560769, 0.0561072
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378529, 0.0378580
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0594033, 0.0594754
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1160266, 0.1160678
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257128, 0.0257171
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1043098, 0.1042126
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1460496, 0.1459941

Time for backsubstitution: 6.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2690

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 278

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128918, upper bound: 0.0126380
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0126292, upper bound: 0.0129007
time: 2.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1797158, 0.1797024
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2286148, 0.2283875
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512640, 0.0512646
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0560770, 0.0561072
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378529, 0.0378580
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0594034, 0.0594753
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1160266, 0.1160678
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257129, 0.0257170
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1043098, 0.1042126
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1460494, 0.1459944

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 99

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3258

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128925, upper bound: 0.0128942
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128882, upper bound: 0.0128985
time: 2.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1793922, 0.1799999
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2279154, 0.2290188
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512536, 0.0512602
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0561169, 0.0559127
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378618, 0.0378342
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0594539, 0.0592664
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1162089, 0.1159218
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257155, 0.0257141
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1041531, 0.1043353
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1458670, 0.1461701

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 768

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128915, upper bound: 0.0128961
time: 9.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128943, upper bound: 0.0128932
time: 4.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1793872, 0.1800048
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2279085, 0.2290257
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512520, 0.0512619
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0561122, 0.0559175
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0378629, 0.0378331
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0594492, 0.0592711
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1162081, 0.1159225
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257157, 0.0257139
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1041509, 0.1043375
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1458662, 0.1461711

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3021

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 799

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128918, upper bound: 0.0128940
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128958, upper bound: 0.0128901
time: 2.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1864609, 0.1860727
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2412664, 0.2404190
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512734, 0.0512683
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0584081, 0.0585499
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380324, 0.0380444
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0621622, 0.0623170
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1177953, 0.1179947
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257773, 0.0257770
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1085629, 0.1084201
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1510452, 0.1508174

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128977, upper bound: 0.0128872
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128882, upper bound: 0.0128967
time: 2.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1861473, 0.1863863
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2405974, 0.2410889
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512701, 0.0512726
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585220, 0.0584341
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380535, 0.0380318
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622892, 0.0621892
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179821, 0.1178539
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257808, 0.0257748
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1084221, 0.1085610
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1508658, 0.1509972

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2275

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2986

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129024, upper bound: 0.0128935
time: 29.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129020, upper bound: 0.0128940
time: 15.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1864051, 0.1863423
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2411578, 0.2410326
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512733, 0.0512702
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585060, 0.0585273
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380535, 0.0380529
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622691, 0.0622861
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179786, 0.1180351
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257713, 0.0257657
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1084989, 0.1085201
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1509808, 0.1509815

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 807

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3577

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128733, upper bound: 0.0128915
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129002, upper bound: 0.0128645
time: 2.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1864119, 0.1863356
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2411894, 0.2410011
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512726, 0.0512709
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0585012, 0.0585321
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380535, 0.0380529
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0622605, 0.0622948
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1179796, 0.1180340
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257683, 0.0257687
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1085168, 0.1085023
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1510051, 0.1509573

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3599
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3072

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3021

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128966, upper bound: 0.0128966
time: 18.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128966, upper bound: 0.0128961
time: 2.72 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 28.21 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129043, upper bound: 0.0128865
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129037, upper bound: 0.0128871
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129032, upper bound: 0.0128883
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129032, upper bound: 0.0128883
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128954, upper bound: 0.0128883
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128954, upper bound: 0.0128894
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0127623, upper bound: 0.0127718
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0127623, upper bound: 0.0127718
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128972, upper bound: 0.0128990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129062, upper bound: 0.0128900
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129016, upper bound: 0.0128978
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129054, upper bound: 0.0128933
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129032, upper bound: 0.0128861
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128937, upper bound: 0.0128959
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129031, upper bound: 0.0128858
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0128954
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128991, upper bound: 0.0128811
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128993, upper bound: 0.0128809
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128855, upper bound: 0.0128833
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129012, upper bound: 0.0128664
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129013, upper bound: 0.0128919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129013, upper bound: 0.0128919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128882, upper bound: 0.0128831
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128880, upper bound: 0.0128843
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129040, upper bound: 0.0128957
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129040, upper bound: 0.0128957
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129020, upper bound: 0.0128963
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129033, upper bound: 0.0128944
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128917, upper bound: 0.0126339
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0126291, upper bound: 0.0128965
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128918, upper bound: 0.0128963
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128919, upper bound: 0.0128960
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0127635, upper bound: 0.0128958
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0127635, upper bound: 0.0128958
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128930, upper bound: 0.0128977
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128924, upper bound: 0.0128981
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128937, upper bound: 0.0129000
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128955, upper bound: 0.0128981
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128924, upper bound: 0.0128977
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128933, upper bound: 0.0128972
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0128975
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128928, upper bound: 0.0128981
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128829, upper bound: 0.0129027
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128827, upper bound: 0.0129028
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128830, upper bound: 0.0129021
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128830, upper bound: 0.0129017
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128827, upper bound: 0.0129017
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128819, upper bound: 0.0129029
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128745, upper bound: 0.0129014
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128822, upper bound: 0.0128937
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128923, upper bound: 0.0128847
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128807, upper bound: 0.0128959
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128962, upper bound: 0.0129000
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128966, upper bound: 0.0128998
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128802, upper bound: 0.0126386
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128960, upper bound: 0.0126226
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0126319, upper bound: 0.0129012
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0126338, upper bound: 0.0128994
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128918, upper bound: 0.0126380
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0126292, upper bound: 0.0129007
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128925, upper bound: 0.0128942
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128882, upper bound: 0.0128985
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128915, upper bound: 0.0128961
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128943, upper bound: 0.0128932
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128918, upper bound: 0.0128940
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128958, upper bound: 0.0128901
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128977, upper bound: 0.0128872
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128882, upper bound: 0.0128967
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129024, upper bound: 0.0128935
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129020, upper bound: 0.0128940
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128733, upper bound: 0.0128915
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0129002, upper bound: 0.0128645
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128966, upper bound: 0.0128966
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 28.21
Output dim: 7, lower bound: -0.0128966, upper bound: 0.0128961
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 28.21
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0128988
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 28.21
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0128989
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 28.21
Output dim: 7, lower bound: -0.0129012, upper bound: 0.0128987
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 28.21
Output dim: 7, lower bound: -0.0129020, upper bound: 0.0128979

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 25.95 + 1786.28 = 1812.23 seconds
