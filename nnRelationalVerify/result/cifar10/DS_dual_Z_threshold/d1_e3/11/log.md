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
execution time: IAR + RelationalAnalysis = 7.86 + 26.34 = 34.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0165178, upper bound: 0.0165269

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3527
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 225
type: DSZ, layer: 1, pos: 330
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 236
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 656
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2270
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2808
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2944
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2977
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3209
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 3272
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 3510
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 3526
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3592
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3595
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2283

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0165013, upper bound: 0.0164987
time: 6.04 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0164877, upper bound: 0.0165081
time: 28.74 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 34.85 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 34.85
Output dim: 7, lower bound: -0.0165013, upper bound: 0.0164987
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 34.85
Output dim: 7, lower bound: -0.0164877, upper bound: 0.0165081

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 34.20 + 34.85 = 69.05 seconds
