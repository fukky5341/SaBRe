## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 11)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0165089745


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4758013, 0.4758013)
1: (-5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4215515, 0.4215515)
2: (-0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943305, 0.0943305)
3: (-1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1155567, 0.1155567)
4: (-0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1455552, 0.1455552)
5: (-0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1566667, 0.1566667)
6: (-1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1202665, 0.1202665)
7: (0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623855, 0.0623855)
8: (-5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3936854, 0.3936854)
9: (-4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4097505, 0.4097506)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.85 + 26.12 = 33.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0165178, upper bound: 0.0165269

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 512

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3363

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165180, upper bound: 0.0165260
time: 77.43 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165180, upper bound: 0.0165258
time: 111.69 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 189.13 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 189.13
Output dim: 7, lower bound: -0.0165180, upper bound: 0.0165260
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 189.13
Output dim: 7, lower bound: -0.0165180, upper bound: 0.0165258

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4758013, 0.4758013
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4215515, 0.4215515
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943305, 0.0943305
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1155567, 0.1155567
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1455552, 0.1455552
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1566667, 0.1566667
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1202665, 0.1202665
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623855, 0.0623855
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3936854, 0.3936854
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4097505, 0.4097506

Time for backsubstitution: 6.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2559

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3374

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165180, upper bound: 0.0165371
time: 5.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165180, upper bound: 0.0165260
time: 32.32 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4758013, 0.4758013
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4215515, 0.4215515
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943305, 0.0943305
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1155567, 0.1155567
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1455552, 0.1455552
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1566667, 0.1566667
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1202665, 0.1202665
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623855, 0.0623855
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3936854, 0.3936854
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4097505, 0.4097506

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2142

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 756

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165182, upper bound: 0.0165299
time: 27.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165181, upper bound: 0.0165315
time: 8.51 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 42.67 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 42.67
Output dim: 7, lower bound: -0.0165180, upper bound: 0.0165371
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 42.67
Output dim: 7, lower bound: -0.0165180, upper bound: 0.0165260
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 42.67
Output dim: 7, lower bound: -0.0165182, upper bound: 0.0165299
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 42.67
Output dim: 7, lower bound: -0.0165181, upper bound: 0.0165315

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4758013, 0.4758013
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4215515, 0.4215515
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943305, 0.0943305
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1155567, 0.1155567
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1455552, 0.1455552
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1566667, 0.1566667
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1202665, 0.1202665
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623855, 0.0623855
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3936854, 0.3936854
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4097505, 0.4097506

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 3025

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2514

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165085, upper bound: 0.0165168
time: 120.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165105, upper bound: 0.0165127
time: 55.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4758013, 0.4758013
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4215515, 0.4215515
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943305, 0.0943305
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1155567, 0.1155567
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1455552, 0.1455552
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1566667, 0.1566667
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1202665, 0.1202665
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623855, 0.0623855
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3936854, 0.3936854
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4097505, 0.4097506

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2498

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165164, upper bound: 0.0165223
time: 125.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165166, upper bound: 0.0165279
time: 63.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4754978, 0.4754139
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4209121, 0.4207014
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943294, 0.0943295
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1154328, 0.1154604
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1455469, 0.1455500
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1565512, 0.1565815
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1200143, 0.1200190
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623852, 0.0623851
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3936368, 0.3936017
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4096038, 0.4095452

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 3025

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 657

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165183, upper bound: 0.0165273
time: 58.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165181, upper bound: 0.0165280
time: 50.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4754140, 0.4754978
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4207014, 0.4209121
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943295, 0.0943294
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1154604, 0.1154328
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1455500, 0.1455469
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1565815, 0.1565512
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1200190, 0.1200143
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623851, 0.0623852
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3936016, 0.3936368
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4095451, 0.4096038

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2959

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165177, upper bound: 0.0165288
time: 46.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165163, upper bound: 0.0165303
time: 7.99 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 60.78 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 60.78
Output dim: 7, lower bound: -0.0165085, upper bound: 0.0165168
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 60.78
Output dim: 7, lower bound: -0.0165105, upper bound: 0.0165127
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 60.78
Output dim: 7, lower bound: -0.0165164, upper bound: 0.0165223
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 60.78
Output dim: 7, lower bound: -0.0165166, upper bound: 0.0165279
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 60.78
Output dim: 7, lower bound: -0.0165183, upper bound: 0.0165273
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 60.78
Output dim: 7, lower bound: -0.0165181, upper bound: 0.0165280
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 60.78
Output dim: 7, lower bound: -0.0165177, upper bound: 0.0165288
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 60.78
Output dim: 7, lower bound: -0.0165163, upper bound: 0.0165303

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4757705, 0.4757798
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4214512, 0.4215386
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943255, 0.0943225
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1155485, 0.1155423
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1455505, 0.1455508
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1566647, 0.1566584
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1202621, 0.1202442
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623747, 0.0623772
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3936707, 0.3936784
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4096687, 0.4096979

Time for backsubstitution: 6.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3013

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165084, upper bound: 0.0165183
time: 13.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165076, upper bound: 0.0165163
time: 27.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4757799, 0.4757705
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4215386, 0.4214512
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943225, 0.0943255
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1155423, 0.1155485
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1455508, 0.1455505
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1566584, 0.1566647
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1202442, 0.1202621
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623772, 0.0623747
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3936785, 0.3936708
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4096979, 0.4096687

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3543

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 660

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165096, upper bound: 0.0165190
time: 7.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165096, upper bound: 0.0165154
time: 77.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4757890, 0.4757867
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4210428, 0.4210397
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0942189, 0.0942185
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1155552, 0.1155553
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1452848, 0.1452808
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1566627, 0.1566626
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1201025, 0.1201015
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623362, 0.0623362
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3935432, 0.3935435
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4091195, 0.4091321

Time for backsubstitution: 6.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2648

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 793

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165138, upper bound: 0.0165207
time: 25.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165156, upper bound: 0.0165216
time: 95.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4757867, 0.4757891
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4210397, 0.4210428
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0942185, 0.0942189
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1155553, 0.1155552
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1452808, 0.1452848
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1566627, 0.1566626
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1201015, 0.1201025
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623362, 0.0623362
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3935435, 0.3935432
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4091321, 0.4091196

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3510

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2838

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165107, upper bound: 0.0165219
time: 25.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165124, upper bound: 0.0165242
time: 8.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4754978, 0.4754139
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4209113, 0.4207004
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943277, 0.0943275
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1154293, 0.1154564
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1455459, 0.1455492
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1565480, 0.1565778
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1200118, 0.1200160
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623839, 0.0623837
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3936344, 0.3935992
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4096009, 0.4095418

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3595

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165179, upper bound: 0.0165358
time: 4.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165179, upper bound: 0.0165257
time: 125.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4754977, 0.4754142
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4209112, 0.4207006
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943275, 0.0943277
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1154288, 0.1154568
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1455460, 0.1455491
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1565475, 0.1565784
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1200114, 0.1200165
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623837, 0.0623838
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3936343, 0.3935992
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4096006, 0.4095422

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2684

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 787

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165173, upper bound: 0.0165361
time: 4.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165180, upper bound: 0.0165256
time: 18.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4746482, 0.4745747
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4189531, 0.4187911
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943236, 0.0943235
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1151727, 0.1151913
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1455030, 0.1454951
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1562542, 0.1562752
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1193410, 0.1194380
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623826, 0.0623824
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3932352, 0.3932007
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4090573, 0.4090103

Time for backsubstitution: 6.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2338

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165091, upper bound: 0.0165186
time: 57.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165117, upper bound: 0.0165146
time: 10.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4744909, 0.4747320
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4185803, 0.4191638
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943235, 0.0943236
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1152190, 0.1151451
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1454983, 0.1454999
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1563055, 0.1562238
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1194426, 0.1193364
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623824, 0.0623826
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3931654, 0.3932703
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4089516, 0.4091159

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 566

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165154, upper bound: 0.0163864
time: 28.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163734, upper bound: 0.0165270
time: 65.56 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 100.79 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0165084, upper bound: 0.0165183
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0165076, upper bound: 0.0165163
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0165096, upper bound: 0.0165190
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0165096, upper bound: 0.0165154
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0165138, upper bound: 0.0165207
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0165156, upper bound: 0.0165216
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0165107, upper bound: 0.0165219
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0165124, upper bound: 0.0165242
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0165179, upper bound: 0.0165358
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0165179, upper bound: 0.0165257
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0165173, upper bound: 0.0165361
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0165180, upper bound: 0.0165256
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0165091, upper bound: 0.0165186
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0165117, upper bound: 0.0165146
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0165154, upper bound: 0.0163864
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 100.79
Output dim: 7, lower bound: -0.0163734, upper bound: 0.0165270

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4756302, 0.4756372
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4213321, 0.4214195
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943224, 0.0943220
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1155036, 0.1155121
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1455402, 0.1455336
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1566119, 0.1566231
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1202264, 0.1202204
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623741, 0.0623759
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3935690, 0.3935300
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4095978, 0.4096299

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2951

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0164992, upper bound: 0.0165150
time: 68.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165063, upper bound: 0.0165122
time: 92.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4756279, 0.4756396
1: -5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4213321, 0.4214195
2: -0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943250, 0.0943193
3: -1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1155183, 0.1154974
4: -0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1455333, 0.1455406
5: -0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1566294, 0.1566056
6: -1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1202382, 0.1202085
7: 0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623735, 0.0623766
8: -5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3935223, 0.3935767
9: -4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4096008, 0.4096268

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2873

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0164989, upper bound: 0.0165093
time: 124.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0165016, upper bound: 0.0165015
time: 23.45 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 154.11 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 154.11
Output dim: 7, lower bound: -0.0164992, upper bound: 0.0165150
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 154.11
Output dim: 7, lower bound: -0.0165063, upper bound: 0.0165122
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 154.11
Output dim: 7, lower bound: -0.0164989, upper bound: 0.0165093
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 154.11
Output dim: 7, lower bound: -0.0165016, upper bound: 0.0165015
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 154.11
Output dim: 7, lower bound: -0.0165096, upper bound: 0.0165190
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 154.11
Output dim: 7, lower bound: -0.0165096, upper bound: 0.0165154
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 154.11
Output dim: 7, lower bound: -0.0165138, upper bound: 0.0165207
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 154.11
Output dim: 7, lower bound: -0.0165156, upper bound: 0.0165216
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 154.11
Output dim: 7, lower bound: -0.0165107, upper bound: 0.0165219
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 154.11
Output dim: 7, lower bound: -0.0165124, upper bound: 0.0165242
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 154.11
Output dim: 7, lower bound: -0.0165179, upper bound: 0.0165358
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 154.11
Output dim: 7, lower bound: -0.0165179, upper bound: 0.0165257
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 154.11
Output dim: 7, lower bound: -0.0165173, upper bound: 0.0165361
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 154.11
Output dim: 7, lower bound: -0.0165180, upper bound: 0.0165256
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 154.11
Output dim: 7, lower bound: -0.0165091, upper bound: 0.0165186
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 154.11
Output dim: 7, lower bound: -0.0165117, upper bound: 0.0165146
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 154.11
Output dim: 7, lower bound: -0.0165154, upper bound: 0.0163864
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 154.11
Output dim: 7, lower bound: -0.0163734, upper bound: 0.0165270

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 33.97 + 1798.31 = 1832.28 seconds
