## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 10)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.0344169486


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240826, 0.3240825)
1: (-1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3811505, 0.3811505)
2: (-1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2837726, 0.2837726)
3: (-4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6619647, 0.6619646)
4: (-2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4952256, 0.4952256)
5: (-4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7620035, 0.7620036)
6: (-4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3675027, 0.3675027)
7: (-3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9713706, 0.9713705)
8: (0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755523, 0.2755523)
9: (-0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4963282, 0.4963282)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.12 + 46.20 = 54.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0344514, upper bound: 0.0344565

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 3422
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 3474
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3257
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3197
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 3558
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 391
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 381
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 3225
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 3181
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 3234
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3015
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3160
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2505
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 3173
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3170
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 3171
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3434

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3443

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0340251, upper bound: 0.0344493
time: 152.56 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344518, upper bound: 0.0344601
time: 16.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 168.96 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 168.96
Output dim: 8, lower bound: -0.0340251, upper bound: 0.0344493
NS_A2, status: Status.UNKNOWN, split count: 1, time: 168.96
Output dim: 8, lower bound: -0.0344518, upper bound: 0.0344601

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.1597147, 0.2800326, -0.1600194, 0.2800364, -0.3224573, 0.3226214
1: -1.0757819, -0.2817194, -1.0757899, -0.2813299, -0.3792832, 0.3789729
2: -1.8377936, -1.0541399, -1.8378450, -1.0541389, -0.2834076, 0.2834470
3: -4.2605700, -2.6198556, -4.2606111, -2.6194129, -0.6589462, 0.6585987
4: -2.5287137, -1.2045258, -2.5288568, -1.2045175, -0.4944622, 0.4945704
5: -4.7281046, -2.8351574, -4.7281194, -2.8346114, -0.7579979, 0.7576597
6: -4.0861225, -2.5240543, -4.0864482, -2.5240533, -0.3655744, 0.3658858
7: -3.7077219, -1.7783970, -3.7077503, -1.7769399, -0.9647695, 0.9633690
8: 0.3490982, 0.7923821, 0.3484607, 0.7923832, -0.2720226, 0.2724778
9: -0.6294641, 0.1021355, -0.6295026, 0.1025922, -0.4940814, 0.4936677

Time for backsubstitution: 6.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3257
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3197
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 381
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 3225
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 3234
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3015
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3160
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 3173
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3170
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 3171
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3434

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3438

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340225, upper bound: 0.0340739
time: 157.98 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340237, upper bound: 0.0340729
time: 161.92 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.1622863, 0.2819964, -0.1613808, 0.2800514, -0.3242074, 0.3252211
1: -1.0779042, -0.2794812, -1.0756205, -0.2794940, -0.3826690, 0.3791013
2: -1.8381107, -1.0540831, -1.8380721, -1.0541377, -0.2836534, 0.2837125
3: -4.2646999, -2.6161749, -4.2607799, -2.6162174, -0.6649660, 0.6591177
4: -2.5295355, -1.2037852, -2.5294929, -1.2044823, -0.4947408, 0.4958369
5: -4.7327180, -2.8306766, -4.7281880, -2.8308146, -0.7655118, 0.7586811
6: -4.0876598, -2.5223813, -4.0877051, -2.5240474, -0.3658689, 0.3694640
7: -3.7174716, -1.7702518, -3.7078660, -1.7702255, -0.9788018, 0.9655492
8: 0.3454892, 0.7960958, 0.3456190, 0.7923887, -0.2729226, 0.2780188
9: -0.6323165, 0.1046413, -0.6296670, 0.1046185, -0.4989198, 0.4944202

Time for backsubstitution: 6.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3257
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3197
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 381
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 3225
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 3234
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3015
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3160
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 3173
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3170
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 3171
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3434

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3438

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344487, upper bound: 0.0340807
time: 43.29 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344506, upper bound: 0.0344573
time: 16.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 66.68 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 66.68
Output dim: 8, lower bound: -0.0340225, upper bound: 0.0340739
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 66.68
Output dim: 8, lower bound: -0.0340237, upper bound: 0.0340729
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 66.68
Output dim: 8, lower bound: -0.0344487, upper bound: 0.0340807
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 66.68
Output dim: 8, lower bound: -0.0344506, upper bound: 0.0344573

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.1615248, 0.2819790, -0.1604552, 0.2800303, -0.3233131, 0.3241361
1: -1.0778980, -0.2804149, -1.0756131, -0.2806292, -0.3819194, 0.3784341
2: -1.8378112, -1.0540847, -1.8377100, -1.0541394, -0.2833287, 0.2833175
3: -4.2644978, -2.6174579, -4.2605362, -2.6177945, -0.6637275, 0.6580481
4: -2.5291271, -1.2038329, -2.5290008, -1.2045406, -0.4942932, 0.4953501
5: -4.7326565, -2.8324771, -4.7281132, -2.8330321, -0.7638239, 0.7572658
6: -4.0874119, -2.5223892, -4.0874043, -2.5240579, -0.3655505, 0.3690864
7: -3.7174637, -1.7733909, -3.7078574, -1.7740693, -0.9750698, 0.9624413
8: 0.3477252, 0.7960943, 0.3483701, 0.7923868, -0.2708075, 0.2754709
9: -0.6322104, 0.1034796, -0.6295397, 0.1031939, -0.4973837, 0.4931218

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3422
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 3474
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3257
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3197
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 3558
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 391
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 381
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 3225
type: A, layer: 1, pos: 3181
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 3234
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3015
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3160
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2505
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 3173
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3170
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 3171
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3434

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3422

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340226, upper bound: 0.0340810
time: 18.99 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344469, upper bound: 0.0340801
time: 21.51 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.1622104, 0.2819929, -0.1613677, 0.2803294, -0.3244876, 0.3251253
1: -1.0779021, -0.2794885, -1.0766082, -0.2794066, -0.3822109, 0.3789641
2: -1.8381079, -1.0540833, -1.8382591, -1.0537496, -0.2839595, 0.2836790
3: -4.2646699, -2.6161830, -4.2625504, -2.6162212, -0.6640915, 0.6598440
4: -2.5295188, -1.2037895, -2.5295396, -1.2039171, -0.4952264, 0.4955854
5: -4.7327166, -2.8306861, -4.7310529, -2.8308251, -0.7641360, 0.7599850
6: -4.0875626, -2.5223825, -4.0876970, -2.5240114, -0.3660348, 0.3693609
7: -3.7174709, -1.7708459, -3.7136829, -1.7709519, -0.9756157, 0.9697488
8: 0.3455018, 0.7960957, 0.3456326, 0.7962884, -0.2760388, 0.2758603
9: -0.6322964, 0.1046344, -0.6317990, 0.1046103, -0.4980188, 0.4965226

Time for backsubstitution: 6.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3422
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 3474
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3257
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3197
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 3558
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 391
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 381
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 3225
type: A, layer: 1, pos: 3181
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 3234
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3015
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3160
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2505
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 3173
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3170
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 3171
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3434

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3422

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0340219, upper bound: 0.0344508
time: 168.75 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344498, upper bound: 0.0344580
time: 22.04 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 196.97 seconds
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 196.97
Output dim: 8, lower bound: -0.0340226, upper bound: 0.0340810
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 196.97
Output dim: 8, lower bound: -0.0344469, upper bound: 0.0340801
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 196.97
Output dim: 8, lower bound: -0.0340219, upper bound: 0.0344508
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 196.97
Output dim: 8, lower bound: -0.0344498, upper bound: 0.0344580

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1616590, 0.2824402, -0.1603559, 0.2800296, -0.3233703, 0.3246861
1: -1.0802116, -0.2802294, -1.0756125, -0.2806595, -0.3839730, 0.3780328
2: -1.8381763, -1.0531287, -1.8377063, -1.0541395, -0.2834795, 0.2842586
3: -4.2676802, -2.6174624, -4.2605300, -2.6178148, -0.6668089, 0.6574734
4: -2.5292256, -1.2020038, -2.5289843, -1.2045463, -0.4940414, 0.4971281
5: -4.7375693, -2.8324862, -4.7281108, -2.8330426, -0.7686121, 0.7559756
6: -4.0874510, -2.5222487, -4.0873375, -2.5240581, -0.3656044, 0.3693236
7: -3.7288268, -1.7734197, -3.7078555, -1.7740929, -0.9859653, 0.9597523
8: 0.3477222, 0.8021505, 0.3483735, 0.7923866, -0.2691232, 0.2813090
9: -0.6364608, 0.1034532, -0.6295345, 0.1031713, -0.5016032, 0.4923988

Time for backsubstitution: 6.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3257
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3197
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 381
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 3225
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 3234
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3015
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3160
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 3173
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3170
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 3171
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3434

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 386

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344455, upper bound: 0.0338206
time: 17.81 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344453, upper bound: 0.0340798
time: 163.99 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1610044, 0.2819690, -0.1604528, 0.2803111, -0.3230947, 0.3240673
1: -1.0778934, -0.2803975, -1.0766015, -0.2801273, -0.3815256, 0.3780937
2: -1.8376907, -1.0540860, -1.8379368, -1.0537517, -0.2835095, 0.2833380
3: -4.2643781, -2.6173601, -4.2623281, -2.6171155, -0.6630443, 0.6585938
4: -2.5288546, -1.2038532, -2.5290229, -1.2039642, -0.4945574, 0.4950202
5: -4.7326279, -2.8323562, -4.7309856, -2.8320849, -0.7628009, 0.7583072
6: -4.0873041, -2.5223944, -4.0875001, -2.5240207, -0.3657096, 0.3691100
7: -3.7174602, -1.7746054, -3.7136741, -1.7737875, -0.9729914, 0.9663109
8: 0.3475934, 0.7960933, 0.3472098, 0.7962865, -0.2739308, 0.2742458
9: -0.6321416, 0.1033193, -0.6316818, 0.1036121, -0.4968538, 0.4950909

Time for backsubstitution: 6.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3257
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3197
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 381
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 3225
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 3234
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3015
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3160
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 3173
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3170
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 3171
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3434

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 386

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340190, upper bound: 0.0341901
time: 30.40 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0340211, upper bound: 0.0344530
time: 20.78 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1623428, 0.2824543, -0.1612681, 0.2803288, -0.3245449, 0.3256751
1: -1.0802158, -0.2793012, -1.0766075, -0.2794349, -0.3842643, 0.3785623
2: -1.8384731, -1.0531275, -1.8382556, -1.0537493, -0.2841100, 0.2846201
3: -4.2678509, -2.6161880, -4.2625451, -2.6162415, -0.6671723, 0.6592697
4: -2.5296168, -1.2019608, -2.5295236, -1.2039226, -0.4949744, 0.4973633
5: -4.7376285, -2.8306949, -4.7310505, -2.8308361, -0.7689244, 0.7586903
6: -4.0876026, -2.5222423, -4.0876293, -2.5240123, -0.3660886, 0.3695981
7: -3.7288332, -1.7708747, -3.7136803, -1.7709763, -0.9865115, 0.9670596
8: 0.3454989, 0.8021519, 0.3456363, 0.7962884, -0.2743492, 0.2816982
9: -0.6365466, 0.1046078, -0.6317935, 0.1045878, -0.5022382, 0.4957997

Time for backsubstitution: 6.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3257
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3197
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 381
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 3225
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 3234
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3015
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3160
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 3173
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3170
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 3171
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3434

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 386

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344464, upper bound: 0.0341873
time: 39.29 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344494, upper bound: 0.0344597
time: 15.65 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 61.62 seconds
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 61.62
Output dim: 8, lower bound: -0.0344455, upper bound: 0.0338206
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 61.62
Output dim: 8, lower bound: -0.0344453, upper bound: 0.0340798
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 61.62
Output dim: 8, lower bound: -0.0340190, upper bound: 0.0341901
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 61.62
Output dim: 8, lower bound: -0.0340211, upper bound: 0.0344530
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 61.62
Output dim: 8, lower bound: -0.0344464, upper bound: 0.0341873
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 61.62
Output dim: 8, lower bound: -0.0344494, upper bound: 0.0344597

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1613068, 0.2823731, -0.1595184, 0.2797501, -0.3222633, 0.3235790
1: -1.0799916, -0.2802709, -1.0745307, -0.2810552, -0.3832831, 0.3759650
2: -1.8380899, -1.0550241, -1.8390493, -1.0561446, -0.2779620, 0.2762885
3: -4.2669301, -2.6191251, -4.2634425, -2.6195202, -0.6593583, 0.6480973
4: -2.5291939, -1.2021255, -2.5305347, -1.2046942, -0.4926079, 0.4957459
5: -4.7374692, -2.8355846, -4.7309561, -2.8362803, -0.7599318, 0.7450052
6: -4.0862131, -2.5222809, -4.0858035, -2.5241017, -0.3636965, 0.3668254
7: -3.7287261, -1.7742802, -3.7113509, -1.7749983, -0.9811777, 0.9539313
8: 0.3507001, 0.8021396, 0.3519618, 0.7906747, -0.2643448, 0.2777436
9: -0.6343259, 0.1034483, -0.6268445, 0.1031629, -0.4996468, 0.4897200

Time for backsubstitution: 6.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 3474
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3257
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3197
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 3558
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 391
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 381
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 3225
type: A, layer: 1, pos: 3181
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 3234
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3015
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3160
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2505
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 3173
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3170
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 3171
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3434

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3458

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0341462, upper bound: 0.0338223
time: 40.27 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344454, upper bound: 0.0338165
time: 138.94 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1615692, 0.2824395, -0.1602452, 0.2800286, -0.3233279, 0.3238541
1: -1.0802038, -0.2802789, -1.0756042, -0.2807231, -0.3837765, 0.3779941
2: -1.8381745, -1.0539939, -1.8377039, -1.0551522, -0.2755692, 0.2842445
3: -4.2676744, -2.6189117, -4.2605238, -2.6196468, -0.6578303, 0.6574256
4: -2.5292230, -1.2026188, -2.5289812, -1.2053041, -0.4927519, 0.4970320
5: -4.7375684, -2.8340073, -4.7281084, -2.8349156, -0.7564918, 0.7559579
6: -4.0874462, -2.5225143, -4.0873299, -2.5243938, -0.3656727, 0.3691821
7: -3.7288196, -1.7747521, -3.7078474, -1.7757794, -0.9798828, 0.9596733
8: 0.3477306, 0.8021500, 0.3483841, 0.7923859, -0.2690574, 0.2765281
9: -0.6364224, 0.1034499, -0.6294889, 0.1031671, -0.5014490, 0.4926594

Time for backsubstitution: 6.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 3474
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3257
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3197
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 3558
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 391
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 381
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 3225
type: A, layer: 1, pos: 3181
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 3234
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3015
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3160
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2505
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 3173
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3170
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 3171
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3434

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3458

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0341473, upper bound: 0.0340828
time: 11.58 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344450, upper bound: 0.0340761
time: 164.31 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1609141, 0.2819682, -0.1603424, 0.2803102, -0.3230525, 0.3232338
1: -1.0778855, -0.2804480, -1.0765932, -0.2801904, -0.3813297, 0.3780562
2: -1.8376884, -1.0549514, -1.8379340, -1.0547640, -0.2756020, 0.2833239
3: -4.2643728, -2.6188092, -4.2623219, -2.6189473, -0.6540658, 0.6585463
4: -2.5288515, -1.2044683, -2.5290198, -1.2047232, -0.4932701, 0.4949241
5: -4.7326269, -2.8338773, -4.7309842, -2.8339589, -0.7506820, 0.7582894
6: -4.0872984, -2.5226593, -4.0874939, -2.5243568, -0.3657782, 0.3689687
7: -3.7174547, -1.7759373, -3.7136664, -1.7754731, -0.9669113, 0.9662319
8: 0.3476017, 0.7960925, 0.3472204, 0.7962857, -0.2738650, 0.2694649
9: -0.6321033, 0.1033161, -0.6316365, 0.1036078, -0.4966981, 0.4953515

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 3474
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3257
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3197
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 3558
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 391
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 381
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 3225
type: A, layer: 1, pos: 3181
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 3234
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3015
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3160
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2505
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 3173
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3170
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 3171
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3434

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3458

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0337231, upper bound: 0.0344484
time: 104.37 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0340197, upper bound: 0.0344561
time: 13.17 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1619925, 0.2823873, -0.1604314, 0.2800496, -0.3234398, 0.3245695
1: -1.0799953, -0.2793428, -1.0755264, -0.2798303, -0.3835751, 0.3764954
2: -1.8383861, -1.0550232, -1.8395977, -1.0557544, -0.2785927, 0.2766473
3: -4.2671013, -2.6178505, -4.2654572, -2.6179464, -0.6597221, 0.6498959
4: -2.5295842, -1.2020819, -2.5310757, -1.2040703, -0.4935413, 0.4959802
5: -4.7375283, -2.8337936, -4.7338943, -2.8340740, -0.7602440, 0.7477173
6: -4.0863619, -2.5222745, -4.0860939, -2.5240562, -0.3641780, 0.3670954
7: -3.7287340, -1.7717359, -3.7171741, -1.7718815, -0.9817235, 0.9612368
8: 0.3484770, 0.8021410, 0.3492252, 0.7945762, -0.2695708, 0.2781324
9: -0.6344110, 0.1046028, -0.6291029, 0.1045794, -0.5002820, 0.4931217

Time for backsubstitution: 6.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 3474
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3257
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3197
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 3558
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 391
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 381
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 3225
type: A, layer: 1, pos: 3181
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 3234
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3015
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3160
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2505
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 3173
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3170
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 3171
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3434

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3458

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0341468, upper bound: 0.0341903
time: 23.41 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344451, upper bound: 0.0341956
time: 13.58 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1622528, 0.2824535, -0.1611574, 0.2803278, -0.3245027, 0.3248446
1: -1.0802079, -0.2793506, -1.0765992, -0.2794980, -0.3840689, 0.3785234
2: -1.8384709, -1.0539930, -1.8382533, -1.0547621, -0.2761993, 0.2846061
3: -4.2678461, -2.6176369, -4.2625375, -2.6180732, -0.6581937, 0.6592221
4: -2.5296142, -1.2025760, -2.5295203, -1.2046809, -0.4936847, 0.4972670
5: -4.7376270, -2.8322163, -4.7310491, -2.8327098, -0.7568040, 0.7586727
6: -4.0875978, -2.5225070, -4.0876226, -2.5243473, -0.3661577, 0.3694566
7: -3.7288270, -1.7722070, -3.7136722, -1.7726626, -0.9804290, 0.9669806
8: 0.3455072, 0.8021514, 0.3456468, 0.7962876, -0.2742834, 0.2769169
9: -0.6365082, 0.1046044, -0.6317481, 0.1045836, -0.5020845, 0.4960598

Time for backsubstitution: 6.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 3474
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3257
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3197
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 3558
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 391
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 381
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 3225
type: A, layer: 1, pos: 3181
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 3234
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3015
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3160
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2505
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 3173
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3170
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 3171
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3434

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3458

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0341523, upper bound: 0.0344517
time: 21.02 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344463, upper bound: 0.0341883
time: 151.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 179.05 seconds
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 179.05
Output dim: 8, lower bound: -0.0341462, upper bound: 0.0338223
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 179.05
Output dim: 8, lower bound: -0.0344454, upper bound: 0.0338165
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 179.05
Output dim: 8, lower bound: -0.0341473, upper bound: 0.0340828
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 179.05
Output dim: 8, lower bound: -0.0344450, upper bound: 0.0340761
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 179.05
Output dim: 8, lower bound: -0.0337231, upper bound: 0.0344484
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 179.05
Output dim: 8, lower bound: -0.0340197, upper bound: 0.0344561
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 179.05
Output dim: 8, lower bound: -0.0341468, upper bound: 0.0341903
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 179.05
Output dim: 8, lower bound: -0.0344451, upper bound: 0.0341956
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 179.05
Output dim: 8, lower bound: -0.0341523, upper bound: 0.0344517
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 179.05
Output dim: 8, lower bound: -0.0344463, upper bound: 0.0341883

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1623518, 0.2835804, -0.1595057, 0.2797492, -0.3238418, 0.3244645
1: -1.0815418, -0.2802828, -1.0742786, -0.2810691, -0.3843811, 0.3746936
2: -1.8380982, -1.0548050, -1.8390424, -1.0561447, -0.2778561, 0.2764753
3: -4.2692652, -2.6196785, -4.2634373, -2.6201246, -0.6607702, 0.6472377
4: -2.5291924, -1.2013533, -2.5305216, -1.2046974, -0.4922806, 0.4965181
5: -4.7404699, -2.8360820, -4.7309546, -2.8368068, -0.7626966, 0.7438359
6: -4.0858335, -2.5198135, -4.0854921, -2.5241020, -0.3622835, 0.3696794
7: -3.7328877, -1.7747402, -3.7113390, -1.7754016, -0.9843663, 0.9522488
8: 0.3506821, 0.8052886, 0.3519671, 0.7906736, -0.2629725, 0.2808012
9: -0.6362501, 0.1034449, -0.6268408, 0.1031558, -0.5015517, 0.4888555

Time for backsubstitution: 6.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3257
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3197
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 381
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 3225
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 3234
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3015
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3160
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 3173
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3170
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 3171
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3434

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 497

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344369, upper bound: 0.0333370
time: 163.35 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344416, upper bound: 0.0338179
time: 22.87 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1626109, 0.2836468, -0.1602325, 0.2800280, -0.3249018, 0.3247402
1: -1.0817543, -0.2802908, -1.0753517, -0.2807369, -0.3848750, 0.3767225
2: -1.8381829, -1.0537747, -1.8376969, -1.0551525, -0.2754634, 0.2844313
3: -4.2700095, -2.6194644, -4.2605200, -2.6202512, -0.6592419, 0.6565664
4: -2.5292206, -1.2018471, -2.5289679, -1.2053068, -0.4924247, 0.4978043
5: -4.7405691, -2.8345058, -4.7281075, -2.8354425, -0.7592565, 0.7547885
6: -4.0870676, -2.5200460, -4.0870190, -2.5243938, -0.3642586, 0.3720359
7: -3.7329817, -1.7752126, -3.7078347, -1.7761825, -0.9830719, 0.9579906
8: 0.3477124, 0.8052989, 0.3483895, 0.7923849, -0.2676850, 0.2795857
9: -0.6383463, 0.1034465, -0.6294855, 0.1031601, -0.5033531, 0.4917952

Time for backsubstitution: 6.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3257
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3197
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 381
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 3225
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 3234
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3015
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3160
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 3173
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3170
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 3171
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3434

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 497

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344388, upper bound: 0.0336002
time: 186.99 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344451, upper bound: 0.0340790
time: 99.45 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1604750, 0.2819651, -0.1599788, 0.2803074, -0.3227666, 0.3228532
1: -1.0777349, -0.2816120, -1.0764691, -0.2811213, -0.3802190, 0.3767627
2: -1.8375082, -1.0549564, -1.8377899, -1.0547677, -0.2754173, 0.2831631
3: -4.2643094, -2.6201537, -4.2622700, -2.6200335, -0.6530760, 0.6573775
4: -2.5283842, -1.2044868, -2.5286465, -1.2047383, -0.4927966, 0.4945417
5: -4.7325993, -2.8355255, -4.7309628, -2.8352833, -0.7493247, 0.7567707
6: -4.0856481, -2.5226600, -4.0861740, -2.5243573, -0.3640248, 0.3675542
7: -3.7173948, -1.7779454, -3.7136173, -1.7771388, -0.9651693, 0.9641681
8: 0.3494109, 0.7960905, 0.3486669, 0.7962841, -0.2721740, 0.2680080
9: -0.6320393, 0.1022034, -0.6315850, 0.1027184, -0.4957464, 0.4942055

Time for backsubstitution: 6.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3257
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3197
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 381
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 3225
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 3234
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3015
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3160
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 3173
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3170
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 3171
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3434

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 497

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0337094, upper bound: 0.0339813
time: 23.14 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0337137, upper bound: 0.0344521
time: 83.07 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1619565, 0.2831753, -0.1603297, 0.2803095, -0.3246264, 0.3241200
1: -1.0794358, -0.2804596, -1.0763402, -0.2802043, -0.3824279, 0.3767843
2: -1.8376967, -1.0547323, -1.8379272, -1.0547640, -0.2754963, 0.2835107
3: -4.2667084, -2.6193624, -4.2623167, -2.6195517, -0.6554775, 0.6576867
4: -2.5288501, -1.2036961, -2.5290065, -1.2047260, -0.4929428, 0.4956964
5: -4.7356267, -2.8343754, -4.7309823, -2.8344860, -0.7534468, 0.7571200
6: -4.0869198, -2.5201919, -4.0871816, -2.5243571, -0.3643639, 0.3718225
7: -3.7216167, -1.7763977, -3.7136543, -1.7758760, -0.9700996, 0.9645491
8: 0.3475834, 0.7992417, 0.3472257, 0.7962848, -0.2724926, 0.2725224
9: -0.6340269, 0.1033129, -0.6316328, 0.1036009, -0.4986021, 0.4944870

Time for backsubstitution: 6.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3257
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3197
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 381
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 3225
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 3234
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3015
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3160
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 3173
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3170
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 3171
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3434

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 497

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340059, upper bound: 0.0339762
time: 147.25 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0340100, upper bound: 0.0344498
time: 40.09 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1630390, 0.2835946, -0.1604187, 0.2800488, -0.3250193, 0.3254550
1: -1.0815455, -0.2793548, -1.0752740, -0.2798443, -0.3846731, 0.3752235
2: -1.8383945, -1.0548038, -1.8395907, -1.0557544, -0.2784866, 0.2768340
3: -4.2694368, -2.6184037, -4.2654524, -2.6185508, -0.6611341, 0.6490366
4: -2.5295825, -1.2013105, -2.5310619, -1.2040730, -0.4932140, 0.4967527
5: -4.7405286, -2.8342912, -4.7338929, -2.8346002, -0.7630088, 0.7465479
6: -4.0859833, -2.5198069, -4.0857821, -2.5240564, -0.3627648, 0.3699493
7: -3.7328954, -1.7721957, -3.7171614, -1.7722846, -0.9849122, 0.9595544
8: 0.3484589, 0.8052900, 0.3492305, 0.7945752, -0.2681985, 0.2811900
9: -0.6363350, 0.1045994, -0.6290990, 0.1045721, -0.5021867, 0.4922571

Time for backsubstitution: 6.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3257
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3197
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 381
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 3225
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 3234
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3015
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3160
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 3173
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3170
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 3171
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3434

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 497

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344404, upper bound: 0.0337147
time: 201.55 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344460, upper bound: 0.0341936
time: 14.09 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1618137, 0.2824503, -0.1607956, 0.2803251, -0.3242168, 0.3244645
1: -1.0800575, -0.2805151, -1.0764754, -0.2804289, -0.3829581, 0.3772301
2: -1.8382907, -1.0539975, -1.8381089, -1.0547658, -0.2760143, 0.2844451
3: -4.2677817, -2.6189816, -4.2624860, -2.6191597, -0.6572042, 0.6580536
4: -2.5291471, -1.2025943, -2.5291469, -1.2046962, -0.4932112, 0.4968849
5: -4.7376013, -2.8338640, -4.7310271, -2.8340340, -0.7554468, 0.7571541
6: -4.0859470, -2.5225084, -4.0863028, -2.5243487, -0.3644037, 0.3680413
7: -3.7287674, -1.7742158, -3.7136238, -1.7743286, -0.9786873, 0.9649171
8: 0.3473166, 0.8021493, 0.3470933, 0.7962859, -0.2725924, 0.2754599
9: -0.6364446, 0.1034916, -0.6316968, 0.1036940, -0.5011331, 0.4949137

Time for backsubstitution: 6.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3257
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3197
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 381
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 3225
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 3234
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3015
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3160
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 3173
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3170
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 3171
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3434

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 497

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0341410, upper bound: 0.0339818
time: 19.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0341477, upper bound: 0.0344476
time: 157.26 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1632972, 0.2836608, -0.1611449, 0.2803271, -0.3260775, 0.3257304
1: -1.0817583, -0.2793625, -1.0763466, -0.2795119, -0.3851672, 0.3772516
2: -1.8384793, -1.0537738, -1.8382463, -1.0547624, -0.2760935, 0.2847929
3: -4.2701807, -2.6181898, -4.2625332, -2.6186774, -0.6596053, 0.6583626
4: -2.5296116, -1.2018043, -2.5295069, -1.2046838, -0.4933574, 0.4980394
5: -4.7406282, -2.8327146, -4.7310476, -2.8332365, -0.7595687, 0.7575033
6: -4.0872183, -2.5200403, -4.0873113, -2.5243483, -0.3647434, 0.3723105
7: -3.7329888, -1.7726675, -3.7136600, -1.7730657, -0.9836181, 0.9652981
8: 0.3454892, 0.8053004, 0.3456522, 0.7962865, -0.2729109, 0.2799744
9: -0.6384318, 0.1046010, -0.6317447, 0.1045766, -0.5039886, 0.4951954

Time for backsubstitution: 6.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 3448
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3257
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 3197
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 3188
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 381
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 3405
type: B, layer: 1, pos: 2621
type: B, layer: 1, pos: 3225
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2281
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 3234
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3015
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3161
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3160
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 3173
type: B, layer: 1, pos: 3174
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3170
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 3172
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 3171
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3434

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 497

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344422, upper bound: 0.0339768
time: 149.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344467, upper bound: 0.0344554
time: 21.38 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 177.23 seconds
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 177.23
Output dim: 8, lower bound: -0.0344369, upper bound: 0.0333370
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 177.23
Output dim: 8, lower bound: -0.0344416, upper bound: 0.0338179
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 177.23
Output dim: 8, lower bound: -0.0344388, upper bound: 0.0336002
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 177.23
Output dim: 8, lower bound: -0.0344451, upper bound: 0.0340790
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 177.23
Output dim: 8, lower bound: -0.0337094, upper bound: 0.0339813
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 177.23
Output dim: 8, lower bound: -0.0337137, upper bound: 0.0344521
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 177.23
Output dim: 8, lower bound: -0.0340059, upper bound: 0.0339762
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 177.23
Output dim: 8, lower bound: -0.0340100, upper bound: 0.0344498
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 177.23
Output dim: 8, lower bound: -0.0344404, upper bound: 0.0337147
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 177.23
Output dim: 8, lower bound: -0.0344460, upper bound: 0.0341936
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 177.23
Output dim: 8, lower bound: -0.0341410, upper bound: 0.0339818
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 177.23
Output dim: 8, lower bound: -0.0341477, upper bound: 0.0344476
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 177.23
Output dim: 8, lower bound: -0.0344422, upper bound: 0.0339768
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 177.23
Output dim: 8, lower bound: -0.0344467, upper bound: 0.0344554

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1613554, 0.2835616, -0.1582289, 0.2796440, -0.3225507, 0.3230050
1: -1.0815351, -0.2808342, -1.0748850, -0.2817942, -0.3833081, 0.3737831
2: -1.8377291, -1.0548069, -1.8385073, -1.0563681, -0.2773235, 0.2759208
3: -4.2690296, -2.6205800, -4.2627158, -2.6211987, -0.6595701, 0.6457398
4: -2.5287974, -1.2014047, -2.5300117, -1.2047608, -0.4915159, 0.4958217
5: -4.7403955, -2.8374324, -4.7298622, -2.8384576, -0.7609494, 0.7413697
6: -4.0855904, -2.5198238, -4.0851688, -2.5241473, -0.3619883, 0.3693153
7: -3.7328789, -1.7779037, -3.7085471, -1.7792743, -0.9808757, 0.9471045
8: 0.3524681, 0.8052864, 0.3541546, 0.7891302, -0.2601357, 0.2787829
9: -0.6361262, 0.1024220, -0.6261003, 0.1019294, -0.5001607, 0.4870398

Time for backsubstitution: 6.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 3474
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3257
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3197
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 3558
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 391
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 381
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 3225
type: A, layer: 1, pos: 3181
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 3234
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3015
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3160
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2505
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 3173
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3170
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 3171
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3434

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3459

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0342483, upper bound: 0.0333401
time: 147.05 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344387, upper bound: 0.0333384
time: 169.13 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1623001, 0.2835788, -0.1594423, 0.2797472, -0.3238261, 0.3243494
1: -1.0815407, -0.2805628, -1.0742778, -0.2814145, -0.3831816, 0.3746022
2: -1.8380963, -1.0548052, -1.8390400, -1.0561445, -0.2778524, 0.2760443
3: -4.2692528, -2.6197338, -4.2634230, -2.6201930, -0.6596792, 0.6471924
4: -2.5290537, -1.2013562, -2.5303502, -1.2047007, -0.4922156, 0.4958797
5: -4.7404690, -2.8361335, -4.7309532, -2.8368695, -0.7601270, 0.7438245
6: -4.0858259, -2.5198140, -4.0854816, -2.5241027, -0.3622747, 0.3695946
7: -3.7328875, -1.7747773, -3.7113380, -1.7754475, -0.9788448, 0.9519948
8: 0.3506844, 0.8052885, 0.3519700, 0.7906736, -0.2629078, 0.2777076
9: -0.6362429, 0.1033917, -0.6268322, 0.1030902, -0.5001705, 0.4888061

Time for backsubstitution: 6.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 3474
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 3448
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3257
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 3197
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 3558
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 391
type: A, layer: 1, pos: 3188
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 381
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 3405
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 3225
type: A, layer: 1, pos: 3181
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2281
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 3234
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3015
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3161
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3160
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2505
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 3173
type: A, layer: 1, pos: 3174
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3170
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 3172
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 3171
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3434

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3459

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0342525, upper bound: 0.0333442
time: 84.32 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344395, upper bound: 0.0338122
time: 154.22 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 54.32 + 3770.34 = 3824.66 seconds
