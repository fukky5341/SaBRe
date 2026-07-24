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
execution time: IAR + RelationalAnalysis = 7.11 + 18.63 = 25.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0129078, upper bound: 0.0129076

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3349

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129072, upper bound: 0.0129072
time: 8.01 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129072, upper bound: 0.0129073
time: 4.25 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 12.33 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 12.33
Output dim: 7, lower bound: -0.0129072, upper bound: 0.0129072
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 12.33
Output dim: 7, lower bound: -0.0129072, upper bound: 0.0129073

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1869136, 0.1869131
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2422413, 0.2422428
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513047, 0.0513046
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0586919, 0.0586920
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380964, 0.0380965
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0624747, 0.0624746
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1184087, 0.1184090
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0258027, 0.0258025
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1088281, 0.1088286
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1513607, 0.1513633

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3024

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129070, upper bound: 0.0129040
time: 10.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129049, upper bound: 0.0129073
time: 21.59 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1869131, 0.1869136
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2422428, 0.2422413
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0513046, 0.0513047
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0586920, 0.0586919
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0380965, 0.0380964
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0624746, 0.0624747
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1184090, 0.1184087
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0258025, 0.0258027
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1088286, 0.1088281
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1513632, 0.1513606

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3024

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129070, upper bound: 0.0129052
time: 4.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129050, upper bound: 0.0129061
time: 8.91 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 18.80 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 18.80
Output dim: 7, lower bound: -0.0129070, upper bound: 0.0129040
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 18.80
Output dim: 7, lower bound: -0.0129049, upper bound: 0.0129073
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 18.80
Output dim: 7, lower bound: -0.0129070, upper bound: 0.0129052
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 18.80
Output dim: 7, lower bound: -0.0129050, upper bound: 0.0129061

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859514, 0.1860466
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2398866, 0.2400974
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512790, 0.0512791
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0583960, 0.0584033
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0379824, 0.0379871
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0621706, 0.0621379
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1175441, 0.1175716
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257916, 0.0257905
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1084625, 0.1085052
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1508214, 0.1508768

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3039

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0129036
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0129038
time: 44.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1860471, 0.1859508
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2400957, 0.2398882
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512792, 0.0512789
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0584032, 0.0583961
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0379871, 0.0379824
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0621380, 0.0621705
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1175713, 0.1175444
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257906, 0.0257915
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1085046, 0.1084630
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1508742, 0.1508240

Time for backsubstitution: 5.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3039

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129036, upper bound: 0.0129059
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129032, upper bound: 0.0129064
time: 51.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859508, 0.1860471
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2398882, 0.2400959
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512789, 0.0512792
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0583961, 0.0584032
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0379824, 0.0379871
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0621705, 0.0621379
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1175444, 0.1175713
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257915, 0.0257906
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1084630, 0.1085046
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1508240, 0.1508741

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3039

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0129030
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0129038
time: 24.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1860466, 0.1859514
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2400974, 0.2398865
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512791, 0.0512790
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0584033, 0.0583960
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0379871, 0.0379824
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0621379, 0.0621706
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1175716, 0.1175441
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257905, 0.0257916
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1085052, 0.1084625
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1508767, 0.1508214

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3039

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129036, upper bound: 0.0129065
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129032, upper bound: 0.0129067
time: 3.01 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 12.02 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0129036
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0129038
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 7, lower bound: -0.0129036, upper bound: 0.0129059
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 7, lower bound: -0.0129032, upper bound: 0.0129064
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0129030
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0129038
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 7, lower bound: -0.0129036, upper bound: 0.0129065
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 7, lower bound: -0.0129032, upper bound: 0.0129067

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1857770, 0.1858577
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2394768, 0.2396635
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512784, 0.0512778
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0583276, 0.0583751
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0379612, 0.0379724
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0620968, 0.0621103
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1173814, 0.1174874
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257915, 0.0257903
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1083958, 0.1084250
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1507233, 0.1507741

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3448

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129060, upper bound: 0.0129023
time: 18.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129047, upper bound: 0.0129030
time: 10.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1857624, 0.1859277
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2394527, 0.2398213
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512777, 0.0512785
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0583873, 0.0583348
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0379746, 0.0379659
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0621611, 0.0620641
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1175147, 0.1174089
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257915, 0.0257903
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1083823, 0.1084529
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1507187, 0.1508064

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3448

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0129024
time: 4.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129047, upper bound: 0.0129030
time: 11.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859282, 0.1857619
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2398198, 0.2394543
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512786, 0.0512776
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0583348, 0.0583874
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0379659, 0.0379747
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0620642, 0.0621610
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1174086, 0.1175149
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257904, 0.0257913
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1084523, 0.1083828
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1508039, 0.1507213

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3448

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129036, upper bound: 0.0129053
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129022, upper bound: 0.0129066
time: 4.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1858582, 0.1857765
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2396619, 0.2394785
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512779, 0.0512783
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0583750, 0.0583277
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0379723, 0.0379612
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0621104, 0.0620967
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1174871, 0.1173817
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257904, 0.0257914
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1084244, 0.1083964
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1507715, 0.1507259

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3448

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129032, upper bound: 0.0129016
time: 13.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129018, upper bound: 0.0129058
time: 2.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1857765, 0.1858582
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2394785, 0.2396619
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512783, 0.0512779
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0583277, 0.0583750
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0379612, 0.0379723
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0620967, 0.0621104
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1173817, 0.1174871
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257914, 0.0257904
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1083964, 0.1084244
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1507259, 0.1507715

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3448

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0129019
time: 29.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129047, upper bound: 0.0129036
time: 3.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1857619, 0.1859282
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2394543, 0.2398198
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512776, 0.0512786
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0583874, 0.0583348
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0379747, 0.0379659
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0621610, 0.0620642
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1175149, 0.1174086
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257913, 0.0257904
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1083828, 0.1084523
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1507213, 0.1508039

Time for backsubstitution: 5.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3448

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0129026
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129048, upper bound: 0.0129038
time: 4.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1859277, 0.1857625
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2398213, 0.2394527
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512785, 0.0512777
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0583348, 0.0583873
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0379659, 0.0379746
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0620641, 0.0621611
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1174089, 0.1175146
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257903, 0.0257915
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1084529, 0.1083823
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1508064, 0.1507187

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3448

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129036, upper bound: 0.0129015
time: 6.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129023, upper bound: 0.0129065
time: 2.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1858577, 0.1857770
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2396635, 0.2394768
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0512778, 0.0512784
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0583751, 0.0583276
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0379724, 0.0379611
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0621103, 0.0620968
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1174874, 0.1173814
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257903, 0.0257915
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1084250, 0.1083958
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1507740, 0.1507233

Time for backsubstitution: 5.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3448

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129032, upper bound: 0.0129039
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129019, upper bound: 0.0129057
time: 2.64 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 10.94 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129060, upper bound: 0.0129023
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129047, upper bound: 0.0129030
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0129024
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129047, upper bound: 0.0129030
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129036, upper bound: 0.0129053
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129022, upper bound: 0.0129066
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129032, upper bound: 0.0129016
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129018, upper bound: 0.0129058
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0129019
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129047, upper bound: 0.0129036
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129061, upper bound: 0.0129026
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129048, upper bound: 0.0129038
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129036, upper bound: 0.0129015
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129023, upper bound: 0.0129065
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129032, upper bound: 0.0129039
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.94
Output dim: 7, lower bound: -0.0129019, upper bound: 0.0129057

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1856509, 0.1857405
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2393653, 0.2395369
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511045, 0.0511244
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0581996, 0.0582626
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0373630, 0.0372966
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0617708, 0.0618234
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1172946, 0.1174111
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257290, 0.0257197
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1077517, 0.1076950
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505805, 0.1506132

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129047, upper bound: 0.0129010
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129052, upper bound: 0.0129013
time: 2.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1856599, 0.1857315
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2393502, 0.2395519
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511250, 0.0511039
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582151, 0.0582471
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0372854, 0.0373742
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0618100, 0.0617842
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1173051, 0.1174006
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257208, 0.0257278
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1076659, 0.1077808
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505625, 0.1506312

Time for backsubstitution: 5.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129034, upper bound: 0.0129022
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129038, upper bound: 0.0129022
time: 2.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1856363, 0.1858106
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2393411, 0.2396947
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511038, 0.0511250
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582594, 0.0582223
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0373764, 0.0372901
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0618350, 0.0617772
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1174279, 0.1173326
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257289, 0.0257197
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1077382, 0.1077230
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505759, 0.1506456

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129047, upper bound: 0.0129013
time: 23.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129052, upper bound: 0.0129010
time: 2.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1856453, 0.1858016
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2393261, 0.2397097
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511243, 0.0511046
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582748, 0.0582069
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0372988, 0.0373678
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0618742, 0.0617381
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1174384, 0.1173221
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257208, 0.0257278
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1076524, 0.1078088
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505579, 0.1506636

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129033, upper bound: 0.0129026
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129038, upper bound: 0.0129024
time: 2.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1858021, 0.1856448
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2397082, 0.2393277
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511046, 0.0511242
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582068, 0.0582749
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0373677, 0.0372989
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0617381, 0.0618741
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1173218, 0.1174386
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257279, 0.0257207
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1078082, 0.1076530
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1506610, 0.1505604

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129024, upper bound: 0.0129037
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129024, upper bound: 0.0129035
time: 2.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1858111, 0.1856358
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2396932, 0.2393427
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511251, 0.0511037
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582222, 0.0582595
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0372901, 0.0373765
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0617773, 0.0618349
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1173323, 0.1174282
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257198, 0.0257288
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1077225, 0.1077387
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1506431, 0.1505784

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129011, upper bound: 0.0129050
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129011, upper bound: 0.0129044
time: 2.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1857321, 0.1856593
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2395504, 0.2393519
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511040, 0.0511249
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582471, 0.0582151
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0373741, 0.0372854
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0617843, 0.0618099
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1174003, 0.1173054
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257279, 0.0257207
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1077803, 0.1076664
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1506286, 0.1505651

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129019, upper bound: 0.0129047
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129022, upper bound: 0.0129036
time: 2.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1857411, 0.1856503
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2395353, 0.2393669
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511245, 0.0511044
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582625, 0.0581997
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0372965, 0.0373631
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0618235, 0.0617707
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1174108, 0.1172949
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257198, 0.0257288
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1076945, 0.1077523
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1506107, 0.1505830

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129005, upper bound: 0.0129051
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129009, upper bound: 0.0129049
time: 2.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1856503, 0.1857411
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2393669, 0.2395353
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511044, 0.0511245
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0581997, 0.0582625
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0373631, 0.0372965
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0617707, 0.0618235
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1172949, 0.1174108
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257288, 0.0257198
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1077523, 0.1076945
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505830, 0.1506107

Time for backsubstitution: 5.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129048, upper bound: 0.0129010
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129052, upper bound: 0.0129013
time: 2.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1856593, 0.1857321
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2393519, 0.2395504
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511249, 0.0511039
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582151, 0.0582471
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0372854, 0.0373741
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0618099, 0.0617843
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1173054, 0.1174003
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257207, 0.0257279
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1076665, 0.1077803
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505651, 0.1506287

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129034, upper bound: 0.0129021
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129039, upper bound: 0.0129023
time: 2.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1856358, 0.1858111
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2393427, 0.2396932
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511037, 0.0511251
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582595, 0.0582222
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0373765, 0.0372901
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0618349, 0.0617773
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1174282, 0.1173323
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257288, 0.0257198
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1077387, 0.1077225
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505784, 0.1506430

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129047, upper bound: 0.0129011
time: 13.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129052, upper bound: 0.0129018
time: 2.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1856448, 0.1858021
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2393277, 0.2397082
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511242, 0.0511046
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582749, 0.0582068
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0372989, 0.0373677
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0618741, 0.0617381
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1174387, 0.1173218
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257207, 0.0257279
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1076530, 0.1078082
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1505604, 0.1506610

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129034, upper bound: 0.0129029
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129039, upper bound: 0.0129026
time: 2.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1858016, 0.1856453
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2397097, 0.2393261
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511046, 0.0511243
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582069, 0.0582748
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0373678, 0.0372988
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0617380, 0.0618742
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1173221, 0.1174384
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257278, 0.0257208
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1078088, 0.1076524
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1506636, 0.1505579

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129025, upper bound: 0.0129042
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129025, upper bound: 0.0129031
time: 2.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1858106, 0.1856363
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2396947, 0.2393411
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511251, 0.0511038
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582223, 0.0582594
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0372901, 0.0373764
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0617772, 0.0618350
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1173326, 0.1174279
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257197, 0.0257289
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1077230, 0.1077381
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1506456, 0.1505759

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129011, upper bound: 0.0129054
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129011, upper bound: 0.0129049
time: 7.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1857315, 0.1856599
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2395519, 0.2393502
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511039, 0.0511250
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582472, 0.0582151
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0373742, 0.0372854
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0617842, 0.0618100
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1174006, 0.1173051
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257278, 0.0257208
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1077808, 0.1076659
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1506312, 0.1505625

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129019, upper bound: 0.0129044
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129023, upper bound: 0.0129036
time: 2.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1857405, 0.1856509
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2395369, 0.2393653
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511244, 0.0511045
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0582626, 0.0581996
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0372966, 0.0373630
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0618234, 0.0617708
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1174111, 0.1172946
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257197, 0.0257290
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1076950, 0.1077517
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1506132, 0.1505805

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3038

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129006, upper bound: 0.0129058
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129009, upper bound: 0.0129049
time: 2.72 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 10.91 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129047, upper bound: 0.0129010
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129052, upper bound: 0.0129013
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129034, upper bound: 0.0129022
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129038, upper bound: 0.0129022
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129047, upper bound: 0.0129013
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129052, upper bound: 0.0129010
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129033, upper bound: 0.0129026
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129038, upper bound: 0.0129024
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129024, upper bound: 0.0129037
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129024, upper bound: 0.0129035
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129011, upper bound: 0.0129050
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129011, upper bound: 0.0129044
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129019, upper bound: 0.0129047
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129022, upper bound: 0.0129036
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129005, upper bound: 0.0129051
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129009, upper bound: 0.0129049
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129048, upper bound: 0.0129010
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129052, upper bound: 0.0129013
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129034, upper bound: 0.0129021
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129039, upper bound: 0.0129023
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129047, upper bound: 0.0129011
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129052, upper bound: 0.0129018
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129034, upper bound: 0.0129029
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129039, upper bound: 0.0129026
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129025, upper bound: 0.0129042
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129025, upper bound: 0.0129031
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129011, upper bound: 0.0129054
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129011, upper bound: 0.0129049
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129019, upper bound: 0.0129044
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129023, upper bound: 0.0129036
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129006, upper bound: 0.0129058
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.91
Output dim: 7, lower bound: -0.0129009, upper bound: 0.0129049

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1838872, 0.1837221
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2352864, 0.2351277
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510932, 0.0511108
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0574774, 0.0576336
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371602, 0.0371154
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0609918, 0.0611512
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1156988, 0.1160178
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257239, 0.0257148
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1069436, 0.1068436
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496700, 0.1496529

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129030, upper bound: 0.0128905
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128942, upper bound: 0.0128991
time: 2.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1836324, 0.1839186
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2349560, 0.2353506
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510909, 0.0511131
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575706, 0.0575403
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371813, 0.0370938
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610986, 0.0610444
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1158995, 0.1158153
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257240, 0.0257146
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1069002, 0.1068870
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496202, 0.1497028

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129034, upper bound: 0.0128901
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128948, upper bound: 0.0128988
time: 2.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1838962, 0.1837131
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2352713, 0.2351427
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511137, 0.0510903
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0574928, 0.0576181
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0370826, 0.0371930
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610310, 0.0611120
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1157092, 0.1160074
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257158, 0.0257229
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068577, 0.1069293
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496520, 0.1496708

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129016, upper bound: 0.0128918
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128928, upper bound: 0.0129005
time: 2.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1836414, 0.1839096
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2349410, 0.2353657
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511114, 0.0510926
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575860, 0.0575249
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371037, 0.0371715
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0611378, 0.0610052
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1159100, 0.1158048
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257159, 0.0257227
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068144, 0.1069728
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496022, 0.1497207

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129021, upper bound: 0.0128915
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128934, upper bound: 0.0129001
time: 2.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1837849, 0.1837922
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2350867, 0.2352855
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510925, 0.0511115
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575371, 0.0575828
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371737, 0.0371048
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610560, 0.0610950
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1158320, 0.1159087
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257239, 0.0257148
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1069197, 0.1068715
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496501, 0.1496853

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0128907
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128941, upper bound: 0.0128993
time: 2.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1836179, 0.1840470
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2349319, 0.2356157
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510902, 0.0511138
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0576304, 0.0575000
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371952, 0.0370874
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0611628, 0.0609983
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1160346, 0.1157367
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257240, 0.0257146
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068867, 0.1069149
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496156, 0.1497351

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129034, upper bound: 0.0128906
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128948, upper bound: 0.0128993
time: 2.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1837939, 0.1837832
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2350717, 0.2353005
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511130, 0.0510910
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575525, 0.0575674
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0370961, 0.0371824
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610952, 0.0610558
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1158425, 0.1158982
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257158, 0.0257229
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068339, 0.1069573
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496321, 0.1497033

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129016, upper bound: 0.0128921
time: 2.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128928, upper bound: 0.0129007
time: 2.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1836269, 0.1840380
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2349168, 0.2356308
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511107, 0.0510933
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0576458, 0.0574846
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371176, 0.0371650
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0612020, 0.0609590
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1160451, 0.1157262
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257159, 0.0257227
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068009, 0.1070007
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1495976, 0.1497531

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129021, upper bound: 0.0128920
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0129007
time: 2.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1840385, 0.1836264
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2356292, 0.2349185
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510934, 0.0511107
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0574845, 0.0576459
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371649, 0.0371177
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0609591, 0.0612019
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1157259, 0.1160454
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257229, 0.0257158
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1070001, 0.1068014
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497506, 0.1496001

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129007, upper bound: 0.0128935
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128920, upper bound: 0.0129021
time: 2.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1837837, 0.1837934
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2352989, 0.2350733
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510911, 0.0511129
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575673, 0.0575526
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371824, 0.0370961
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610559, 0.0610951
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1158979, 0.1158428
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257230, 0.0257156
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1069568, 0.1068344
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497007, 0.1496346

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129007, upper bound: 0.0128928
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128921, upper bound: 0.0129016
time: 2.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1840475, 0.1836174
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2356142, 0.2349335
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511139, 0.0510902
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0574999, 0.0576304
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0370873, 0.0371953
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0609984, 0.0611627
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1157364, 0.1160349
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257147, 0.0257239
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1069144, 0.1068872
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497326, 0.1496181

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128994, upper bound: 0.0128948
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128907, upper bound: 0.0129034
time: 2.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1837927, 0.1837844
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2352839, 0.2350883
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511116, 0.0510924
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575828, 0.0575372
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371047, 0.0371738
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610951, 0.0610559
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1159084, 0.1158323
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257149, 0.0257238
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068709, 0.1069202
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496827, 0.1496526

Time for backsubstitution: 5.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128993, upper bound: 0.0128942
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128907, upper bound: 0.0129029
time: 2.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1839101, 0.1836410
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2353642, 0.2349426
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510927, 0.0511113
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575248, 0.0575861
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371714, 0.0371038
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610053, 0.0611377
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1158045, 0.1159103
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257229, 0.0257158
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1069722, 0.1068150
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497182, 0.1496048

Time for backsubstitution: 5.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129002, upper bound: 0.0128934
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128915, upper bound: 0.0129021
time: 2.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1837136, 0.1838958
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2351410, 0.2352729
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510904, 0.0511136
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0576181, 0.0574929
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371929, 0.0370827
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0611121, 0.0610309
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1160071, 0.1157095
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257230, 0.0257157
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1069288, 0.1068584
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496683, 0.1496546

Time for backsubstitution: 5.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129005, upper bound: 0.0128928
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128919, upper bound: 0.0129016
time: 2.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1839191, 0.1836319
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2353491, 0.2349577
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511132, 0.0510908
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575402, 0.0575706
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0370938, 0.0371814
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610445, 0.0610985
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1158150, 0.1158998
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257147, 0.0257239
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068864, 0.1069007
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497002, 0.1496227

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128988, upper bound: 0.0128948
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128901, upper bound: 0.0129034
time: 2.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1837226, 0.1838868
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2351260, 0.2352879
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511109, 0.0510931
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0576335, 0.0574774
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371153, 0.0371603
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0611513, 0.0609917
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1160176, 0.1156991
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257149, 0.0257238
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068430, 0.1069441
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496504, 0.1496726

Time for backsubstitution: 5.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128991, upper bound: 0.0128942
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128905, upper bound: 0.0129030
time: 2.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1838868, 0.1837226
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2352879, 0.2351260
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510931, 0.0511109
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0574774, 0.0576335
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371603, 0.0371153
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0609917, 0.0611513
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1156991, 0.1160176
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257238, 0.0257149
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1069441, 0.1068430
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496726, 0.1496503

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129030, upper bound: 0.0128904
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128942, upper bound: 0.0128990
time: 2.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1836320, 0.1839191
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2349577, 0.2353491
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510908, 0.0511132
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575707, 0.0575402
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371814, 0.0370938
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610985, 0.0610445
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1158998, 0.1158150
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257239, 0.0257147
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1069007, 0.1068864
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496227, 0.1497002

Time for backsubstitution: 5.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129035, upper bound: 0.0128900
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128948, upper bound: 0.0128987
time: 2.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1838958, 0.1837136
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2352729, 0.2351410
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511136, 0.0510904
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0574929, 0.0576181
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0370827, 0.0371929
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610309, 0.0611121
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1157095, 0.1160071
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257157, 0.0257230
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068583, 0.1069288
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496546, 0.1496683

Time for backsubstitution: 5.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0128918
time: 2.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128929, upper bound: 0.0129005
time: 2.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1836410, 0.1839101
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2349426, 0.2353642
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511113, 0.0510927
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575861, 0.0575248
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371038, 0.0371714
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0611377, 0.0610053
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1159103, 0.1158045
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257158, 0.0257229
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068150, 0.1069722
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496048, 0.1497182

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129021, upper bound: 0.0128914
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0129001
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1837844, 0.1837927
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2350883, 0.2352839
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510924, 0.0511116
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575372, 0.0575827
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371738, 0.0371047
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610559, 0.0610951
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1158323, 0.1159084
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257238, 0.0257149
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1069202, 0.1068709
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496526, 0.1496827

Time for backsubstitution: 5.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129030, upper bound: 0.0128906
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128942, upper bound: 0.0128993
time: 2.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1836174, 0.1840475
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2349335, 0.2356142
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510902, 0.0511139
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0576304, 0.0574999
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371953, 0.0370873
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0611627, 0.0609984
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1160349, 0.1157365
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257239, 0.0257147
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068872, 0.1069144
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496181, 0.1497326

Time for backsubstitution: 5.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129035, upper bound: 0.0128906
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128949, upper bound: 0.0128993
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1837934, 0.1837837
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2350733, 0.2352989
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511129, 0.0510911
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575526, 0.0575673
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0370961, 0.0371824
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610951, 0.0610559
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1158428, 0.1158979
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257156, 0.0257230
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068344, 0.1069567
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496346, 0.1497006

Time for backsubstitution: 5.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129016, upper bound: 0.0128920
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128928, upper bound: 0.0129006
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1836264, 0.1840385
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2349185, 0.2356292
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511107, 0.0510934
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0576459, 0.0574845
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371177, 0.0371649
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0612019, 0.0609592
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1160454, 0.1157260
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257158, 0.0257229
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068014, 0.1070001
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496001, 0.1497505

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129021, upper bound: 0.0128919
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0129006
time: 2.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1840380, 0.1836269
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2356308, 0.2349168
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510933, 0.0511107
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0574846, 0.0576458
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371650, 0.0371176
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0609591, 0.0612020
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1157262, 0.1160451
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257227, 0.0257159
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1070007, 0.1068009
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497531, 0.1495976

Time for backsubstitution: 5.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129008, upper bound: 0.0128934
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128921, upper bound: 0.0129021
time: 2.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1837831, 0.1837939
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2353005, 0.2350717
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510910, 0.0511130
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575674, 0.0575525
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371824, 0.0370961
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610558, 0.0610952
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1158982, 0.1158425
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257229, 0.0257158
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1069573, 0.1068339
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497032, 0.1496321

Time for backsubstitution: 5.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129007, upper bound: 0.0128927
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128921, upper bound: 0.0129015
time: 2.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1840470, 0.1836179
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2356157, 0.2349319
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511138, 0.0510902
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575000, 0.0576304
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0370874, 0.0371952
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0609982, 0.0611628
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1157367, 0.1160346
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257146, 0.0257240
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1069149, 0.1068866
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497351, 0.1496155

Time for backsubstitution: 5.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128994, upper bound: 0.0128947
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128907, upper bound: 0.0129033
time: 2.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1837922, 0.1837849
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2352855, 0.2350867
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511115, 0.0510925
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575828, 0.0575371
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371048, 0.0371737
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610950, 0.0610560
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1159087, 0.1158320
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257148, 0.0257239
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068715, 0.1069196
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496852, 0.1496501

Time for backsubstitution: 5.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128994, upper bound: 0.0128941
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128908, upper bound: 0.0129029
time: 2.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1839096, 0.1836414
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2353657, 0.2349410
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510926, 0.0511114
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575249, 0.0575860
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371715, 0.0371037
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610052, 0.0611378
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1158048, 0.1159100
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257227, 0.0257159
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1069728, 0.1068144
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497208, 0.1496022

Time for backsubstitution: 5.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129002, upper bound: 0.0128934
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128915, upper bound: 0.0129020
time: 2.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1837131, 0.1838962
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2351427, 0.2352713
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510903, 0.0511137
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0576181, 0.0574928
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371930, 0.0370826
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0611120, 0.0610310
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1160073, 0.1157092
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257229, 0.0257158
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1069293, 0.1068578
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496709, 0.1496521

Time for backsubstitution: 5.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129005, upper bound: 0.0128927
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128919, upper bound: 0.0129015
time: 2.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1839186, 0.1836324
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2353506, 0.2349560
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511131, 0.0510909
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0575403, 0.0575706
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0370938, 0.0371813
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0610444, 0.0610986
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1158153, 0.1158995
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257146, 0.0257240
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068870, 0.1069002
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1497028, 0.1496202

Time for backsubstitution: 5.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128989, upper bound: 0.0128947
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128902, upper bound: 0.0129033
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1837221, 0.1838872
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2351277, 0.2352864
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511108, 0.0510932
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0576336, 0.0574774
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0371154, 0.0371602
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0611512, 0.0609918
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1160178, 0.1156988
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257147, 0.0257239
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1068436, 0.1069436
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1496529, 0.1496700

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128992, upper bound: 0.0128941
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128906, upper bound: 0.0129029
time: 2.77 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 11.10 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129030, upper bound: 0.0128905
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128942, upper bound: 0.0128991
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129034, upper bound: 0.0128901
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128948, upper bound: 0.0128988
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129016, upper bound: 0.0128918
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128928, upper bound: 0.0129005
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129021, upper bound: 0.0128915
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128934, upper bound: 0.0129001
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129029, upper bound: 0.0128907
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128941, upper bound: 0.0128993
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129034, upper bound: 0.0128906
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128948, upper bound: 0.0128993
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129016, upper bound: 0.0128921
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128928, upper bound: 0.0129007
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129021, upper bound: 0.0128920
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0129007
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129007, upper bound: 0.0128935
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128920, upper bound: 0.0129021
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129007, upper bound: 0.0128928
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128921, upper bound: 0.0129016
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128994, upper bound: 0.0128948
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128907, upper bound: 0.0129034
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128993, upper bound: 0.0128942
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128907, upper bound: 0.0129029
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129002, upper bound: 0.0128934
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128915, upper bound: 0.0129021
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129005, upper bound: 0.0128928
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128919, upper bound: 0.0129016
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128988, upper bound: 0.0128948
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128901, upper bound: 0.0129034
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128991, upper bound: 0.0128942
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128905, upper bound: 0.0129030
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129030, upper bound: 0.0128904
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128942, upper bound: 0.0128990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129035, upper bound: 0.0128900
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128948, upper bound: 0.0128987
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0128918
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128929, upper bound: 0.0129005
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129021, upper bound: 0.0128914
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0129001
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129030, upper bound: 0.0128906
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128942, upper bound: 0.0128993
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129035, upper bound: 0.0128906
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128949, upper bound: 0.0128993
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129016, upper bound: 0.0128920
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128928, upper bound: 0.0129006
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129021, upper bound: 0.0128919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0129006
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129008, upper bound: 0.0128934
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128921, upper bound: 0.0129021
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129007, upper bound: 0.0128927
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128921, upper bound: 0.0129015
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128994, upper bound: 0.0128947
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128907, upper bound: 0.0129033
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128994, upper bound: 0.0128941
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128908, upper bound: 0.0129029
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129002, upper bound: 0.0128934
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128915, upper bound: 0.0129020
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0129005, upper bound: 0.0128927
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128919, upper bound: 0.0129015
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128989, upper bound: 0.0128947
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128902, upper bound: 0.0129033
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128992, upper bound: 0.0128941
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.10
Output dim: 7, lower bound: -0.0128906, upper bound: 0.0129029

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1805122, 0.1803292
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2274448, 0.2273639
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510933, 0.0511109
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0563046, 0.0564985
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0367862, 0.0367448
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0597372, 0.0599315
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1128098, 0.1131870
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257074, 0.0256977
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1055322, 0.1054619
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1478627, 0.1479079

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129018, upper bound: 0.0128786
time: 4.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128922, upper bound: 0.0128785
time: 65.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1804944, 0.1803476
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2275226, 0.2272869
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510933, 0.0511109
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0563422, 0.0564608
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0367896, 0.0367413
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0597720, 0.0598966
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1128668, 0.1131289
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257069, 0.0256982
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1055619, 0.1054321
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1479250, 0.1478454

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128929, upper bound: 0.0128871
time: 19.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128827, upper bound: 0.0128983
time: 10.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1802579, 0.1805257
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2271150, 0.2275869
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510910, 0.0511132
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0563978, 0.0564052
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0368072, 0.0367233
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0598441, 0.0598248
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1130106, 0.1129840
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257075, 0.0256976
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1054889, 0.1055053
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1478128, 0.1479577

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129022, upper bound: 0.0128787
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128927, upper bound: 0.0128888
time: 12.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1802396, 0.1805437
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2271923, 0.2275095
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510910, 0.0511132
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0564355, 0.0563676
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0368108, 0.0367197
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0598787, 0.0597899
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1130685, 0.1129263
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257070, 0.0256981
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1055185, 0.1054755
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1478752, 0.1478952

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0128879
time: 6.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128834, upper bound: 0.0128981
time: 3.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1805213, 0.1803202
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2274297, 0.2273790
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511138, 0.0510904
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0563200, 0.0564831
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0367085, 0.0368225
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0597764, 0.0598923
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1128202, 0.1131765
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0256993, 0.0257058
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1054465, 0.1055477
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1478447, 0.1479259

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129004, upper bound: 0.0128801
time: 4.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128908, upper bound: 0.0128912
time: 2.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1805034, 0.1803386
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2275076, 0.2273020
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511138, 0.0510904
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0563576, 0.0564454
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0367120, 0.0368189
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0598112, 0.0598575
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1128772, 0.1131184
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0256987, 0.0257064
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1054761, 0.1055179
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1479070, 0.1478634

Time for backsubstitution: 5.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128916, upper bound: 0.0128895
time: 22.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128814, upper bound: 0.0128995
time: 5.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1802669, 0.1805167
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2271000, 0.2276020
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511115, 0.0510927
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0564132, 0.0563898
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0367296, 0.0368009
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0598833, 0.0597856
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1130210, 0.1129735
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0256994, 0.0257057
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1054031, 0.1055911
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1477948, 0.1479757

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129009, upper bound: 0.0128803
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128913, upper bound: 0.0128906
time: 13.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1802486, 0.1805347
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2271773, 0.2275245
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511115, 0.0510927
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0564509, 0.0563521
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0367331, 0.0367974
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0599179, 0.0597507
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1130790, 0.1129158
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0256989, 0.0257062
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1054328, 0.1055613
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1478572, 0.1479132

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128922, upper bound: 0.0128895
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128820, upper bound: 0.0128993
time: 4.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1804100, 0.1803992
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2272457, 0.2275218
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510926, 0.0511116
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0563643, 0.0564477
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0367996, 0.0367343
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0598015, 0.0598753
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1129430, 0.1130778
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257074, 0.0256977
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1055084, 0.1054899
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1478427, 0.1479402

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0128792
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128921, upper bound: 0.0128892
time: 20.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1803920, 0.1804176
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2273229, 0.2274443
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510926, 0.0511116
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0564020, 0.0564101
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0368031, 0.0367307
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0598363, 0.0598405
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1130006, 0.1130197
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257068, 0.0256982
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1055380, 0.1054600
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1479051, 0.1478778

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128929, upper bound: 0.0128887
time: 8.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128826, upper bound: 0.0128979
time: 58.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1802428, 0.1806540
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2270899, 0.2278520
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510903, 0.0511139
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0564576, 0.0563647
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0368211, 0.0367168
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0599083, 0.0597784
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1131456, 0.1129049
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257075, 0.0256976
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1054751, 0.1055332
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1478078, 0.1479901

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129022, upper bound: 0.0128795
time: 4.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128926, upper bound: 0.0128897
time: 3.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1802250, 0.1806719
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2271682, 0.2277740
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510903, 0.0511139
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0564952, 0.0563273
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0368247, 0.0367133
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0599430, 0.0597437
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1132036, 0.1128477
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257070, 0.0256981
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1055050, 0.1055034
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1478705, 0.1479276

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0128884
time: 9.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128833, upper bound: 0.0128985
time: 10.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1804190, 0.1803902
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2272307, 0.2275368
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511131, 0.0510911
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0563798, 0.0564323
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0367220, 0.0368119
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0598407, 0.0598361
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1129535, 0.1130673
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0256992, 0.0257058
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1054226, 0.1055756
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1478247, 0.1479582

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129004, upper bound: 0.0128799
time: 32.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128907, upper bound: 0.0128911
time: 15.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1804010, 0.1804086
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2273079, 0.2274593
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511131, 0.0510911
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0564174, 0.0563946
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0367255, 0.0368083
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0598755, 0.0598013
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1130111, 0.1130092
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0256987, 0.0257064
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1054522, 0.1055458
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1478871, 0.1478957

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128915, upper bound: 0.0128896
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128813, upper bound: 0.0128995
time: 12.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1802518, 0.1806450
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2270749, 0.2278670
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511108, 0.0510934
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0564730, 0.0563492
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0367435, 0.0367944
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0599475, 0.0597392
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1131561, 0.1128944
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0256994, 0.0257057
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1053893, 0.1056190
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1477898, 0.1480080

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0129009, upper bound: 0.0128808
time: 8.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128913, upper bound: 0.0128911
time: 3.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1802340, 0.1806629
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2271531, 0.2277890
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0511108, 0.0510934
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0565107, 0.0563118
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0367470, 0.0367909
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0599822, 0.0597045
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1132141, 0.1128372
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0256989, 0.0257062
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1054192, 0.1055892
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1478525, 0.1479456

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128922, upper bound: 0.0128900
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128820, upper bound: 0.0129000
time: 2.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1806634, 0.1802335
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2277875, 0.2271547
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510934, 0.0511107
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0563118, 0.0565107
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0367909, 0.0367471
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0597046, 0.0599821
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1128370, 0.1132144
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257063, 0.0256987
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1055886, 0.1054198
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1479430, 0.1478551

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128994, upper bound: 0.0128824
time: 3.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128896, upper bound: 0.0128923
time: 8.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1806456, 0.1802512
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2278655, 0.2270764
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510934, 0.0511107
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0563491, 0.0564731
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0367943, 0.0367436
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0597393, 0.0599474
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1128941, 0.1131564
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257058, 0.0256993
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1056185, 0.1053898
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1480055, 0.1477924

Time for backsubstitution: 5.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128907, upper bound: 0.0128915
time: 134.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128803, upper bound: 0.0129008
time: 6.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1804091, 0.1804005
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2274578, 0.2273096
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510912, 0.0511130
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0563946, 0.0564175
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0368083, 0.0367256
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0598014, 0.0598754
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1130089, 0.1130114
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257065, 0.0256986
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1055452, 0.1054528
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1478932, 0.1478896

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128994, upper bound: 0.0128817
time: 4.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128896, upper bound: 0.0128920
time: 3.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.6832824, -3.2785172, -3.6832824, -3.2785172, -0.1803908, 0.1804186
1: -6.4961281, -5.8339081, -6.4961281, -5.8339081, -0.2275351, 0.2272322
2: -0.4305590, -0.2741258, -0.4305590, -0.2741258, -0.0510912, 0.0511130
3: -1.0987204, -0.8154984, -1.0987204, -0.8154984, -0.0564322, 0.0563799
4: -0.6264212, -0.4625564, -0.6264212, -0.4625564, -0.0368118, 0.0367220
5: -0.0520843, 0.2261229, -0.0520843, 0.2261229, -0.0598362, 0.0598406
6: -4.1229086, -3.6151509, -4.1229086, -3.6151509, -0.1130670, 0.1129538
7: 1.2784494, 1.5145966, 1.2784494, 1.5145966, -0.0257060, 0.0256991
8: -6.2270203, -5.8316193, -6.2270203, -5.8316193, -0.1055751, 0.1054231
9: -5.3906612, -4.9291134, -5.3906612, -4.9291134, -0.1479557, 0.1478273

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 269
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3599

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0128908, upper bound: 0.0128810
time: 198.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0128804, upper bound: 0.0129008
time: 3.79 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 207.97 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0129018, upper bound: 0.0128786
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128922, upper bound: 0.0128785
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128929, upper bound: 0.0128871
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128827, upper bound: 0.0128983
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0129022, upper bound: 0.0128787
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128927, upper bound: 0.0128888
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0128879
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128834, upper bound: 0.0128981
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0129004, upper bound: 0.0128801
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128908, upper bound: 0.0128912
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128916, upper bound: 0.0128895
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128814, upper bound: 0.0128995
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0129009, upper bound: 0.0128803
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128913, upper bound: 0.0128906
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128922, upper bound: 0.0128895
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128820, upper bound: 0.0128993
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0128792
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128921, upper bound: 0.0128892
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128929, upper bound: 0.0128887
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128826, upper bound: 0.0128979
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0129022, upper bound: 0.0128795
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128926, upper bound: 0.0128897
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0128884
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128833, upper bound: 0.0128985
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0129004, upper bound: 0.0128799
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128907, upper bound: 0.0128911
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128915, upper bound: 0.0128896
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128813, upper bound: 0.0128995
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0129009, upper bound: 0.0128808
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128913, upper bound: 0.0128911
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128922, upper bound: 0.0128900
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128820, upper bound: 0.0129000
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128994, upper bound: 0.0128824
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128896, upper bound: 0.0128923
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128907, upper bound: 0.0128915
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128803, upper bound: 0.0129008
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128994, upper bound: 0.0128817
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128896, upper bound: 0.0128920
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128908, upper bound: 0.0128810
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 207.97
Output dim: 7, lower bound: -0.0128804, upper bound: 0.0129008
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128994, upper bound: 0.0128948
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128907, upper bound: 0.0129034
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128993, upper bound: 0.0128942
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128907, upper bound: 0.0129029
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0129002, upper bound: 0.0128934
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128915, upper bound: 0.0129021
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0129005, upper bound: 0.0128928
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128919, upper bound: 0.0129016
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128988, upper bound: 0.0128948
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128901, upper bound: 0.0129034
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128991, upper bound: 0.0128942
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128905, upper bound: 0.0129030
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0129030, upper bound: 0.0128904
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128942, upper bound: 0.0128990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0129035, upper bound: 0.0128900
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128948, upper bound: 0.0128987
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0129017, upper bound: 0.0128918
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128929, upper bound: 0.0129005
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0129021, upper bound: 0.0128914
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0129001
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0129030, upper bound: 0.0128906
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128942, upper bound: 0.0128993
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0129035, upper bound: 0.0128906
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128949, upper bound: 0.0128993
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0129016, upper bound: 0.0128920
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128928, upper bound: 0.0129006
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0129021, upper bound: 0.0128919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128935, upper bound: 0.0129006
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0129008, upper bound: 0.0128934
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128921, upper bound: 0.0129021
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0129007, upper bound: 0.0128927
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128921, upper bound: 0.0129015
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128994, upper bound: 0.0128947
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128907, upper bound: 0.0129033
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128994, upper bound: 0.0128941
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128908, upper bound: 0.0129029
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0129002, upper bound: 0.0128934
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128915, upper bound: 0.0129020
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0129005, upper bound: 0.0128927
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128919, upper bound: 0.0129015
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128989, upper bound: 0.0128947
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128902, upper bound: 0.0129033
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128992, upper bound: 0.0128941
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 207.97
Output dim: 7, lower bound: -0.0128906, upper bound: 0.0129029

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 25.73 + 1823.80 = 1849.54 seconds
