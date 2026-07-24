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
execution time: IAR + RelationalAnalysis = 7.44 + 44.94 = 52.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0344514, upper bound: 0.0344565

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 1066

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3016

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344446, upper bound: 0.0344501
time: 155.08 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344498, upper bound: 0.0344465
time: 340.46 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 495.56 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 495.56
Output dim: 8, lower bound: -0.0344446, upper bound: 0.0344501
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 495.56
Output dim: 8, lower bound: -0.0344498, upper bound: 0.0344465

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240820, 0.3240819
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3811133, 0.3811316
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2837700, 0.2837638
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6619630, 0.6619633
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4952167, 0.4951833
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7619988, 0.7619993
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674954, 0.3674973
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9713687, 0.9713688
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755503, 0.2755528
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4963224, 0.4963230

Time for backsubstitution: 6.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2989

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2266

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344411, upper bound: 0.0344386
time: 172.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344410, upper bound: 0.0344543
time: 18.56 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240818, 0.3240820
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3811317, 0.3811133
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2837638, 0.2837700
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6619633, 0.6619629
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4951832, 0.4952168
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7619994, 0.7619987
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674973, 0.3674953
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9713688, 0.9713687
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755528, 0.2755503
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4963230, 0.4963224

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3461

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1116

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344504, upper bound: 0.0344537
time: 29.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344504, upper bound: 0.0344491
time: 38.29 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 73.35 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 73.35
Output dim: 8, lower bound: -0.0344411, upper bound: 0.0344386
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 73.35
Output dim: 8, lower bound: -0.0344410, upper bound: 0.0344543
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 73.35
Output dim: 8, lower bound: -0.0344504, upper bound: 0.0344537
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 73.35
Output dim: 8, lower bound: -0.0344504, upper bound: 0.0344491

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240815, 0.3240811
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3809816, 0.3810807
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2837685, 0.2837613
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6619387, 0.6619527
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4952118, 0.4951529
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7619880, 0.7619952
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674440, 0.3674719
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9713451, 0.9713203
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755480, 0.2755510
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4963097, 0.4963172

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344392, upper bound: 0.0344300
time: 168.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344208, upper bound: 0.0344520
time: 15.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240812, 0.3240814
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3810623, 0.3810000
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2837675, 0.2837623
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6619524, 0.6619389
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4951863, 0.4951784
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7619945, 0.7619887
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674700, 0.3674459
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9713201, 0.9713453
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755485, 0.2755504
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4963167, 0.4963103

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3257

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344403, upper bound: 0.0344471
time: 127.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344400, upper bound: 0.0344485
time: 74.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240818, 0.3240820
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3811317, 0.3811133
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2837638, 0.2837700
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6619633, 0.6619629
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4951832, 0.4952168
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7619994, 0.7619987
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674973, 0.3674953
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9713688, 0.9713687
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755528, 0.2755503
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4963230, 0.4963224

Time for backsubstitution: 6.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 514

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1105

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344506, upper bound: 0.0344463
time: 221.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344498, upper bound: 0.0344459
time: 228.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240818, 0.3240820
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3811317, 0.3811133
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2837638, 0.2837700
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6619633, 0.6619629
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4951832, 0.4952168
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7619994, 0.7619987
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674973, 0.3674953
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9713688, 0.9713687
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755528, 0.2755503
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4963230, 0.4963224

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2090

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344434, upper bound: 0.0344413
time: 17.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344421, upper bound: 0.0344337
time: 52.74 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 75.92 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 75.92
Output dim: 8, lower bound: -0.0344392, upper bound: 0.0344300
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 75.92
Output dim: 8, lower bound: -0.0344208, upper bound: 0.0344520
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 75.92
Output dim: 8, lower bound: -0.0344403, upper bound: 0.0344471
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 75.92
Output dim: 8, lower bound: -0.0344400, upper bound: 0.0344485
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 75.92
Output dim: 8, lower bound: -0.0344506, upper bound: 0.0344463
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 75.92
Output dim: 8, lower bound: -0.0344498, upper bound: 0.0344459
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 75.92
Output dim: 8, lower bound: -0.0344434, upper bound: 0.0344413
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 75.92
Output dim: 8, lower bound: -0.0344421, upper bound: 0.0344337

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240624, 0.3240690
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3809747, 0.3810737
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2837652, 0.2837582
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6618957, 0.6619006
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4951979, 0.4951403
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7619856, 0.7619858
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674427, 0.3674706
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9713365, 0.9713126
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755246, 0.2755247
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4962916, 0.4962969

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3279

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2299

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344361, upper bound: 0.0344176
time: 215.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344382, upper bound: 0.0344275
time: 76.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240693, 0.3240621
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3809746, 0.3810738
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2837654, 0.2837580
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6618866, 0.6619098
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4951992, 0.4951390
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7619788, 0.7619926
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674427, 0.3674706
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9713376, 0.9713117
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755217, 0.2755276
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4962893, 0.4962991

Time for backsubstitution: 5.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 1122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 101

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344217, upper bound: 0.0344511
time: 16.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344203, upper bound: 0.0344480
time: 38.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240798, 0.3240799
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3802708, 0.3802565
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2837427, 0.2837372
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6618918, 0.6618769
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4951164, 0.4951057
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7616833, 0.7616749
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3660207, 0.3660418
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9712295, 0.9712616
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755474, 0.2755493
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4961969, 0.4961957

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2366

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2987

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344360, upper bound: 0.0344440
time: 21.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344408, upper bound: 0.0344418
time: 212.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240798, 0.3240799
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3803190, 0.3802084
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2837423, 0.2837375
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6618903, 0.6618783
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4951137, 0.4951084
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7616807, 0.7616775
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3660657, 0.3659967
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9712363, 0.9712547
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755474, 0.2755493
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4962020, 0.4961905

Time for backsubstitution: 5.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 1050

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3104

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344381, upper bound: 0.0344430
time: 155.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344381, upper bound: 0.0344465
time: 281.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240818, 0.3240820
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3811316, 0.3811132
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2837637, 0.2837699
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6619633, 0.6619629
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4951829, 0.4952164
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7619993, 0.7619987
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674973, 0.3674953
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9713687, 0.9713684
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755528, 0.2755503
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4963230, 0.4963224

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 1107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3150

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344495, upper bound: 0.0344459
time: 178.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344489, upper bound: 0.0344443
time: 158.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240818, 0.3240820
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3811316, 0.3811132
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2837637, 0.2837699
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6619633, 0.6619629
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4951830, 0.4952163
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7619993, 0.7619987
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3674973, 0.3674953
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9713687, 0.9713684
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755528, 0.2755503
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4963230, 0.4963224

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 536

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0343634, upper bound: 0.0344493
time: 49.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344469, upper bound: 0.0343628
time: 174.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240788, 0.3240761
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3811215, 0.3810983
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2837265, 0.2837424
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6616563, 0.6619667
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4951565, 0.4952078
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7613409, 0.7616761
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3672118, 0.3673629
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9711464, 0.9712505
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755522, 0.2755498
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4963218, 0.4963213

Time for backsubstitution: 5.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3462

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1099

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344409, upper bound: 0.0344381
time: 23.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344363, upper bound: 0.0344392
time: 162.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1613897, 0.2800541, -0.1613897, 0.2800541, -0.3240759, 0.3240789
1: -1.0758250, -0.2794920, -1.0758250, -0.2794920, -0.3811167, 0.3811030
2: -1.8380749, -1.0541350, -1.8380749, -1.0541350, -0.2837362, 0.2837328
3: -4.2607999, -2.6161935, -4.2607999, -2.6161935, -0.6619670, 0.6616560
4: -2.5294950, -1.2044777, -2.5294950, -1.2044777, -0.4951744, 0.4951900
5: -4.7281904, -2.8307877, -4.7281904, -2.8307877, -0.7616767, 0.7613403
6: -4.0879216, -2.5240469, -4.0879216, -2.5240469, -0.3673648, 0.3672098
7: -3.7078867, -1.7700897, -3.7078867, -1.7700897, -0.9712507, 0.9711462
8: 0.3456139, 0.7923894, 0.3456139, 0.7923894, -0.2755523, 0.2755498
9: -0.6296841, 0.1046304, -0.6296841, 0.1046304, -0.4963219, 0.4963212

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3170
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 1099
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3174
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 2281
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3225
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3015
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 1105
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 1066
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3160
type: DSZ, layer: 1, pos: 3161
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3197
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 353
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 3172
type: DSZ, layer: 1, pos: 1101
type: DSZ, layer: 1, pos: 1106
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3405
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3171
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 1100
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 3234
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 3448
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3071

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2077

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344369, upper bound: 0.0344424
time: 76.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344400, upper bound: 0.0344317
time: 121.50 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 203.56 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0344361, upper bound: 0.0344176
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0344382, upper bound: 0.0344275
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0344217, upper bound: 0.0344511
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0344203, upper bound: 0.0344480
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0344360, upper bound: 0.0344440
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0344408, upper bound: 0.0344418
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0344381, upper bound: 0.0344430
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0344381, upper bound: 0.0344465
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0344495, upper bound: 0.0344459
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0344489, upper bound: 0.0344443
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0343634, upper bound: 0.0344493
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0344469, upper bound: 0.0343628
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0344409, upper bound: 0.0344381
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0344363, upper bound: 0.0344392
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0344369, upper bound: 0.0344424
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 203.56
Output dim: 8, lower bound: -0.0344400, upper bound: 0.0344317

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 52.38 + 3706.75 = 3759.13 seconds
