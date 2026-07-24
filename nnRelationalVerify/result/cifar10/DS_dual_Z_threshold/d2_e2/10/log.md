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
execution time: IAR + RelationalAnalysis = 8.10 + 45.80 = 53.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0344514, upper bound: 0.0344565

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 396

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344491, upper bound: 0.0344292
time: 30.34 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344238, upper bound: 0.0344234
time: 162.99 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 193.42 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 193.42
Output dim: 8, lower bound: -0.0344491, upper bound: 0.0344292
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 193.42
Output dim: 8, lower bound: -0.0344238, upper bound: 0.0344234

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240577, 0.3240572
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3811290, 0.3811313
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2835687, 0.2835945
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6616439, 0.6616960
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4949414, 0.4950353
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7617610, 0.7618003
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674836, 0.3674879
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9712781, 0.9712962
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755523, 0.2755514
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4962867, 0.4962908

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 381

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344303, upper bound: 0.0344003
time: 222.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344340, upper bound: 0.0344039
time: 142.35 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240572, 0.3240577
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3811312, 0.3811291
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2835945, 0.2835687
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6616960, 0.6616438
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4950354, 0.4949414
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7618003, 0.7617610
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674879, 0.3674836
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9712961, 0.9712781
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755515, 0.2755523
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4962908, 0.4962867

Time for backsubstitution: 5.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 381

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344009, upper bound: 0.0344358
time: 16.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344144, upper bound: 0.0344312
time: 228.46 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 251.10 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 251.10
Output dim: 8, lower bound: -0.0344303, upper bound: 0.0344003
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 251.10
Output dim: 8, lower bound: -0.0344340, upper bound: 0.0344039
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 251.10
Output dim: 8, lower bound: -0.0344009, upper bound: 0.0344358
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 251.10
Output dim: 8, lower bound: -0.0344144, upper bound: 0.0344312

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240481, 0.3240641
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3811338, 0.3811190
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2834974, 0.2835470
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6614333, 0.6616020
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4948579, 0.4950203
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7615420, 0.7617031
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674763, 0.3674857
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9712201, 0.9712957
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755520, 0.2755659
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4962801, 0.4962648

Time for backsubstitution: 5.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3461

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0343075, upper bound: 0.0344204
time: 12.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344279, upper bound: 0.0342917
time: 364.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240577, 0.3240475
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3811168, 0.3811313
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2835212, 0.2835945
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6615499, 0.6616960
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4949263, 0.4950353
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7616638, 0.7618003
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674815, 0.3674879
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9712777, 0.9712962
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755523, 0.2755512
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4962608, 0.4962908

Time for backsubstitution: 5.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 3461

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0343108, upper bound: 0.0344066
time: 14.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344336, upper bound: 0.0342851
time: 13.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240475, 0.3240629
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3811352, 0.3811167
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2835189, 0.2835212
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6614854, 0.6615499
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4949518, 0.4949264
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7615813, 0.7616638
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674806, 0.3674814
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9712383, 0.9712777
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755512, 0.2755663
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4962794, 0.4962607

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3461

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0342776, upper bound: 0.0344359
time: 202.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0343985, upper bound: 0.0343150
time: 195.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240572, 0.3240481
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3811190, 0.3811291
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2835470, 0.2835687
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6616020, 0.6616438
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4950203, 0.4949414
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7617031, 0.7617610
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674858, 0.3674836
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9712957, 0.9712781
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755515, 0.2755520
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4962649, 0.4962867

Time for backsubstitution: 5.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3461

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0342910, upper bound: 0.0344340
time: 149.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0344131, upper bound: 0.0343107
time: 15.29 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 170.74 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 170.74
Output dim: 8, lower bound: -0.0343075, upper bound: 0.0344204
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 170.74
Output dim: 8, lower bound: -0.0344279, upper bound: 0.0342917
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 170.74
Output dim: 8, lower bound: -0.0343108, upper bound: 0.0344066
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 170.74
Output dim: 8, lower bound: -0.0344336, upper bound: 0.0342851
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 170.74
Output dim: 8, lower bound: -0.0342776, upper bound: 0.0344359
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 170.74
Output dim: 8, lower bound: -0.0343985, upper bound: 0.0343150
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 170.74
Output dim: 8, lower bound: -0.0342910, upper bound: 0.0344340
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 170.74
Output dim: 8, lower bound: -0.0344131, upper bound: 0.0343107

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240319, 0.3240483
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3809745, 0.3809544
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2834555, 0.2835065
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6605411, 0.6607072
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4947819, 0.4949443
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7603050, 0.7604633
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3671516, 0.3671697
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9688027, 0.9688174
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2744825, 0.2745436
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4961109, 0.4960878

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3459

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0341184, upper bound: 0.0344187
time: 107.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0343057, upper bound: 0.0342219
time: 163.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240323, 0.3240479
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3809693, 0.3809597
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2834569, 0.2835051
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6605386, 0.6607097
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4947819, 0.4949443
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7603021, 0.7604663
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3671603, 0.3671610
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9687418, 0.9688783
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2745298, 0.2744964
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4961030, 0.4960956

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 3459

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0342393, upper bound: 0.0342955
time: 25.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344310, upper bound: 0.0341041
time: 159.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240419, 0.3240313
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3809523, 0.3809720
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2834806, 0.2835526
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6606551, 0.6608037
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4948503, 0.4949594
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7604239, 0.7605635
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3671654, 0.3671632
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9687993, 0.9688786
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2745300, 0.2744817
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4960837, 0.4961216

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3459

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0342447, upper bound: 0.0342798
time: 256.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344330, upper bound: 0.0340934
time: 26.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240313, 0.3240471
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3809758, 0.3809523
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2834769, 0.2834806
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6605932, 0.6606551
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4948759, 0.4948503
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7603445, 0.7604239
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3671559, 0.3671654
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9688208, 0.9687992
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2744817, 0.2745440
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4961101, 0.4960837

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3459

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0340883, upper bound: 0.0344421
time: 11.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0342766, upper bound: 0.0342482
time: 187.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240409, 0.3240322
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3809597, 0.3809646
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2835051, 0.2835281
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6607097, 0.6607491
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4949444, 0.4948653
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7604663, 0.7605211
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3671610, 0.3671676
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9688782, 0.9687997
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2744820, 0.2745298
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4960957, 0.4961097

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3459

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0340995, upper bound: 0.0344329
time: 28.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0342882, upper bound: 0.0342438
time: 34.19 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 68.76 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 68.76
Output dim: 8, lower bound: -0.0341184, upper bound: 0.0344187
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 68.76
Output dim: 8, lower bound: -0.0343057, upper bound: 0.0342219
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 68.76
Output dim: 8, lower bound: -0.0342393, upper bound: 0.0342955
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 68.76
Output dim: 8, lower bound: -0.0344310, upper bound: 0.0341041
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 68.76
Output dim: 8, lower bound: -0.0342447, upper bound: 0.0342798
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 68.76
Output dim: 8, lower bound: -0.0344330, upper bound: 0.0340934
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 68.76
Output dim: 8, lower bound: -0.0340883, upper bound: 0.0344421
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 68.76
Output dim: 8, lower bound: -0.0342766, upper bound: 0.0342482
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 68.76
Output dim: 8, lower bound: -0.0340995, upper bound: 0.0344329
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 68.76
Output dim: 8, lower bound: -0.0342882, upper bound: 0.0342438

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3238665, 0.3239290
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3805716, 0.3804625
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2834026, 0.2834637
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6598607, 0.6600119
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4946398, 0.4948252
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7593770, 0.7595140
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3664009, 0.3665308
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9681969, 0.9681991
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2736810, 0.2738791
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4957654, 0.4956659

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3474

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0339932, upper bound: 0.0344138
time: 14.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0341103, upper bound: 0.0342837
time: 93.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3239130, 0.3238826
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3804774, 0.3805568
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2834142, 0.2834522
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6598433, 0.6600295
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4946629, 0.4948022
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7593529, 0.7595382
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3665213, 0.3664104
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9681236, 0.9682724
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2738652, 0.2736949
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4956812, 0.4957502

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3474

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0342971, upper bound: 0.0340957
time: 29.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344208, upper bound: 0.0339828
time: 27.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3239226, 0.3238660
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3804603, 0.3805692
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2834379, 0.2834997
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6599599, 0.6601234
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4947313, 0.4948173
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7594746, 0.7596353
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3665265, 0.3664125
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9681811, 0.9682728
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2738654, 0.2736802
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4956618, 0.4957761

Time for backsubstitution: 6.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3474

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0343047, upper bound: 0.0339678
time: 141.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344250, upper bound: 0.0339676
time: 21.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3238660, 0.3239278
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3805730, 0.3804604
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2834241, 0.2834379
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6599128, 0.6599599
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4947338, 0.4947313
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7594164, 0.7594745
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3664052, 0.3665265
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9682149, 0.9681811
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2736802, 0.2738794
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4957647, 0.4956618

Time for backsubstitution: 6.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3474

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0339698, upper bound: 0.0344289
time: 26.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340806, upper bound: 0.0343061
time: 24.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3238756, 0.3239130
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3805568, 0.3804727
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2834522, 0.2834855
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6600294, 0.6600539
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4948023, 0.4947463
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7595382, 0.7595717
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3664104, 0.3665286
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9682724, 0.9681815
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2736805, 0.2738652
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4957502, 0.4956878

Time for backsubstitution: 6.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3474

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0339806, upper bound: 0.0344263
time: 30.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340921, upper bound: 0.0343048
time: 17.51 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 54.42 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 54.42
Output dim: 8, lower bound: -0.0339932, upper bound: 0.0344138
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 54.42
Output dim: 8, lower bound: -0.0341103, upper bound: 0.0342837
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 54.42
Output dim: 8, lower bound: -0.0342971, upper bound: 0.0340957
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 54.42
Output dim: 8, lower bound: -0.0344208, upper bound: 0.0339828
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 54.42
Output dim: 8, lower bound: -0.0343047, upper bound: 0.0339678
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 54.42
Output dim: 8, lower bound: -0.0344250, upper bound: 0.0339676
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 54.42
Output dim: 8, lower bound: -0.0339698, upper bound: 0.0344289
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 54.42
Output dim: 8, lower bound: -0.0340806, upper bound: 0.0343061
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 54.42
Output dim: 8, lower bound: -0.0339806, upper bound: 0.0344263
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 54.42
Output dim: 8, lower bound: -0.0340921, upper bound: 0.0343048

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3236256, 0.3235638
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3794870, 0.3796288
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2835648, 0.2835944
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6579216, 0.6581193
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4948345, 0.4949594
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7569101, 0.7571153
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3643645, 0.3641793
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9667079, 0.9668612
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2719163, 0.2716615
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4946228, 0.4947454

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3476

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0343268, upper bound: 0.0339789
time: 73.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344194, upper bound: 0.0339011
time: 15.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3236352, 0.3235472
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3794700, 0.3796412
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2835886, 0.2836420
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6580381, 0.6582133
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4949029, 0.4949744
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7570319, 0.7572125
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3643696, 0.3641815
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9667655, 0.9668616
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2719165, 0.2716468
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4946034, 0.4947714

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3476

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0343305, upper bound: 0.0339723
time: 14.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344229, upper bound: 0.0338892
time: 10.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3235473, 0.3236404
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3796451, 0.3794700
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2835663, 0.2835886
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6580027, 0.6580381
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4948910, 0.4949029
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7569935, 0.7570319
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3641741, 0.3643696
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9668038, 0.9667654
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2716468, 0.2719305
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4947599, 0.4946034

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3476

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0338808, upper bound: 0.0344278
time: 42.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0339632, upper bound: 0.0342070
time: 34.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3235569, 0.3236255
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3796289, 0.3794823
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2835945, 0.2836361
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6581193, 0.6581320
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4949594, 0.4949179
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7571152, 0.7571290
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3641793, 0.3643718
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9668613, 0.9667658
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2716470, 0.2719163
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4947454, 0.4946294

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3529

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3476

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0338939, upper bound: 0.0344236
time: 165.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0339770, upper bound: 0.0343315
time: 30.61 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 202.92 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 202.92
Output dim: 8, lower bound: -0.0343268, upper bound: 0.0339789
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 202.92
Output dim: 8, lower bound: -0.0344194, upper bound: 0.0339011
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 202.92
Output dim: 8, lower bound: -0.0343305, upper bound: 0.0339723
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 202.92
Output dim: 8, lower bound: -0.0344229, upper bound: 0.0338892
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 202.92
Output dim: 8, lower bound: -0.0338808, upper bound: 0.0344278
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 202.92
Output dim: 8, lower bound: -0.0339632, upper bound: 0.0342070
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 202.92
Output dim: 8, lower bound: -0.0338939, upper bound: 0.0344236
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 202.92
Output dim: 8, lower bound: -0.0339770, upper bound: 0.0343315

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 53.90 + 3710.46 = 3764.37 seconds
