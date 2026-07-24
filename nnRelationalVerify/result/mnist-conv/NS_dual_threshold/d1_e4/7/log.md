## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.11948726150000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2823706, 0.2823703)
1: (-12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3489695, 0.3489695)
2: (-9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2716053, 0.2716053)
3: (-0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3234239, 0.3234239)
4: (-11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3780911, 0.3780909)
5: (7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2537158, 0.2537158)
6: (-6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2523494, 0.2523494)
7: (-15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4796824, 0.4796824)
8: (-3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2417318, 0.2417318)
9: (-3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3122460, 0.3122458)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.00 + 35.24 = 58.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.1200877, upper bound: 0.1200876

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6232
type: B, layer: 1, pos: 6232
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200863, upper bound: 0.1183522
time: 6.97 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200863, upper bound: 0.1200861
time: 4.22 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 11.43 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 11.43
Output dim: 5, lower bound: -0.1200863, upper bound: 0.1183522
NS_A2, status: Status.UNKNOWN, split count: 1, time: 11.43
Output dim: 5, lower bound: -0.1200863, upper bound: 0.1200861

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -10.9960213, -10.2271366, -10.9987946, -10.2271118, -0.2762604, 0.2790488
1: -12.4132214, -11.6261654, -12.4154472, -11.6261301, -0.3440664, 0.3462827
2: -9.6395922, -8.8947325, -9.6398697, -8.8946791, -0.2708919, 0.2711487
3: -0.2386980, 0.5735416, -0.2439842, 0.5736139, -0.3100485, 0.3151636
4: -11.7314434, -10.8032789, -11.7315254, -10.8003359, -0.3727696, 0.3701000
5: 7.6976633, 8.3776760, 7.6976194, 8.3797541, -0.2498031, 0.2477565
6: -6.3843861, -5.5822525, -6.3855133, -5.5822067, -0.2489082, 0.2499772
7: -15.9063110, -14.9500103, -15.9111080, -14.9499550, -0.4689763, 0.4737647
8: -3.8060317, -3.1032228, -3.8060818, -3.1007190, -0.2386796, 0.2362202
9: -3.6104774, -2.9848332, -3.6108236, -2.9779658, -0.3033961, 0.2969317

Time for backsubstitution: 21.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6232
type: B, layer: 1, pos: 6232
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6232

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1189510, upper bound: 0.1183403
time: 4.81 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200855, upper bound: 0.1183518
time: 5.15 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -11.0022717, -10.2151604, -11.0019884, -10.2270823, -0.2789814, 0.2871244
1: -12.4187469, -11.6171675, -12.4180088, -11.6260939, -0.3471639, 0.3538572
2: -9.6405535, -8.8934698, -9.6402025, -8.8946161, -0.2715416, 0.2727973
3: -0.2502286, 0.5955524, -0.2500761, 0.5736964, -0.3178705, 0.3246539
4: -11.7436361, -10.7965784, -11.7316208, -10.7969465, -0.3819356, 0.3757172
5: 7.6885214, 8.3822784, 7.6975675, 8.3821468, -0.2569425, 0.2522024
6: -6.3868170, -5.5767164, -6.3868093, -5.5821552, -0.2515869, 0.2557016
7: -15.9174738, -14.9309082, -15.9166470, -14.9498930, -0.4740248, 0.4852827
8: -3.8162079, -3.0977001, -3.8061390, -3.0978241, -0.2454976, 0.2385113
9: -3.6389813, -2.9697738, -3.6112208, -2.9700487, -0.3157101, 0.3031733

Time for backsubstitution: 21.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6232
type: B, layer: 1, pos: 6232
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6232

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1189510, upper bound: 0.1200739
time: 3.94 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200855, upper bound: 0.1200857
time: 4.66 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.11 seconds
NS_A1_A1, status: Status.VERIFIED, split count: 2, time: 30.11
Output dim: 5, lower bound: -0.1189510, upper bound: 0.1183403
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 30.11
Output dim: 5, lower bound: -0.1200855, upper bound: 0.1183518
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 30.11
Output dim: 5, lower bound: -0.1189510, upper bound: 0.1200739
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 30.11
Output dim: 5, lower bound: -0.1200855, upper bound: 0.1200857

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -10.9960213, -10.2271366, -10.9987946, -10.2271118, -0.2762599, 0.2728428
1: -12.4132195, -11.6261635, -12.4154472, -11.6261330, -0.3440659, 0.3398216
2: -9.6395931, -8.8947344, -9.6398716, -8.8946791, -0.2664175, 0.2711487
3: -0.2386980, 0.5735393, -0.2439842, 0.5736139, -0.3098211, 0.3095481
4: -11.7314415, -10.8032780, -11.7315254, -10.8003349, -0.3650570, 0.3700194
5: 7.6976638, 8.3776760, 7.6976194, 8.3797522, -0.2420900, 0.2477565
6: -6.3843861, -5.5822520, -6.3855133, -5.5822062, -0.2488822, 0.2447826
7: -15.9063101, -14.9500122, -15.9111099, -14.9499559, -0.4689271, 0.4702787
8: -3.8060308, -3.1032219, -3.8060827, -3.1007185, -0.2374730, 0.2360061
9: -3.6104784, -2.9848328, -3.6108227, -2.9779668, -0.3032262, 0.2882303

Time for backsubstitution: 21.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6232
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183517, upper bound: 0.1183521
time: 4.33 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183517, upper bound: 0.1183521
time: 3.72 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -11.0010338, -10.2214222, -11.0019627, -10.2300968, -0.2739818, 0.2808482
1: -12.4173508, -11.6242447, -12.4179783, -11.6295042, -0.3397088, 0.3467332
2: -9.6353617, -8.8949242, -9.6377020, -8.8949184, -0.2663212, 0.2648659
3: -0.2489653, 0.5888584, -0.2500761, 0.5704272, -0.3107791, 0.3135718
4: -11.7356453, -10.7981339, -11.7277641, -10.7969913, -0.3729855, 0.3700304
5: 7.6960611, 8.3807373, 7.7012019, 8.3821020, -0.2493676, 0.2459334
6: -6.3858333, -5.5819082, -6.3868093, -5.5846796, -0.2480152, 0.2500213
7: -15.9166508, -14.9345608, -15.9165754, -14.9516506, -0.4712491, 0.4814730
8: -3.8138185, -3.0981674, -3.8049459, -3.0978451, -0.2394416, 0.2361394
9: -3.6369405, -2.9783459, -3.6110191, -2.9741812, -0.3061868, 0.2943665

Time for backsubstitution: 21.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6232
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1172173, upper bound: 0.1200735
time: 3.48 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1172173, upper bound: 0.1200738
time: 5.02 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -11.0022707, -10.2151604, -11.0019884, -10.2270803, -0.2789810, 0.2809035
1: -12.4187460, -11.6171675, -12.4180088, -11.6260948, -0.3471642, 0.3473624
2: -9.6405544, -8.8934736, -9.6402016, -8.8946161, -0.2670672, 0.2727969
3: -0.2502286, 0.5955539, -0.2500761, 0.5736990, -0.3164700, 0.3188418
4: -11.7436352, -10.7965784, -11.7316217, -10.7969446, -0.3741996, 0.3756361
5: 7.6885214, 8.3822784, 7.6975694, 8.3821468, -0.2492192, 0.2522022
6: -6.3868170, -5.5767164, -6.3868093, -5.5821543, -0.2515609, 0.2504892
7: -15.9174757, -14.9309092, -15.9166460, -14.9498920, -0.4739754, 0.4817944
8: -3.8162069, -3.0977001, -3.8061399, -3.0978241, -0.2441150, 0.2382977
9: -3.6389809, -2.9697742, -3.6112220, -2.9700484, -0.3140465, 0.2944725

Time for backsubstitution: 21.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6232
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1183516, upper bound: 0.1200851
time: 4.30 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1183516, upper bound: 0.1200860
time: 3.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.11 seconds
NS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 29.11
Output dim: 5, lower bound: -0.1183517, upper bound: 0.1183521
NS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 29.11
Output dim: 5, lower bound: -0.1183517, upper bound: 0.1183521
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 5, lower bound: -0.1172173, upper bound: 0.1200735
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 5, lower bound: -0.1172173, upper bound: 0.1200738
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 5, lower bound: -0.1183516, upper bound: 0.1200851
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 5, lower bound: -0.1183516, upper bound: 0.1200860

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -11.0010338, -10.2214222, -10.9959936, -10.2301531, -0.2755775, 0.2747682
1: -12.4173508, -11.6242447, -12.4131927, -11.6295738, -0.3407643, 0.3418677
2: -9.6353617, -8.8949242, -9.6370916, -8.8950367, -0.2664318, 0.2642074
3: -0.2489653, 0.5888584, -0.2386980, 0.5702684, -0.3088050, 0.3010122
4: -11.7356453, -10.7981339, -11.7275829, -10.8033228, -0.3657620, 0.3683138
5: 7.6960611, 8.3807373, 7.7012982, 8.3776312, -0.2439449, 0.2445809
6: -6.3858333, -5.5819082, -6.3843865, -5.5847759, -0.2472723, 0.2469763
7: -15.9166508, -14.9345608, -15.9062366, -14.9517689, -0.4748377, 0.4708838
8: -3.8138185, -3.0981674, -3.8048396, -3.1032438, -0.2339993, 0.2391541
9: -3.6369405, -2.9783459, -3.6102755, -2.9889655, -0.2913488, 0.2962654

Time for backsubstitution: 20.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6232
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4657

## Relational analysis of NS_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1193642
time: 3.91 seconds

## Relational analysis of NS_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1200712
time: 6.07 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -11.0010338, -10.2214222, -11.0022449, -10.2181740, -0.2747133, 0.2734399
1: -12.4173508, -11.6242447, -12.4187164, -11.6205740, -0.3419828, 0.3433573
2: -9.6353617, -8.8949242, -9.6380491, -8.8937759, -0.2668846, 0.2641740
3: -0.2489653, 0.5888584, -0.2502286, 0.5922837, -0.3109421, 0.3070056
4: -11.7356453, -10.7981339, -11.7397757, -10.7966232, -0.3671651, 0.3704145
5: 7.6960611, 8.3807373, 7.6921558, 8.3822327, -0.2447081, 0.2460076
6: -6.3858333, -5.5819082, -6.3868175, -5.5792389, -0.2487540, 0.2466519
7: -15.9166508, -14.9345608, -15.9174004, -14.9326696, -0.4716527, 0.4706254
8: -3.8138185, -3.0981674, -3.8150148, -3.0977235, -0.2327136, 0.2363853
9: -3.6369405, -2.9783459, -3.6387806, -2.9739053, -0.2971749, 0.2948616

Time for backsubstitution: 20.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6232
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4657

## Relational analysis of NS_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1193650
time: 3.68 seconds

## Relational analysis of NS_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1200722
time: 3.57 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -11.0022707, -10.2151604, -10.9960194, -10.2271357, -0.2816736, 0.2748238
1: -12.4187460, -11.6171675, -12.4132214, -11.6261635, -0.3490074, 0.3424966
2: -9.6405544, -8.8934736, -9.6395912, -8.8947353, -0.2671785, 0.2721384
3: -0.2502286, 0.5955539, -0.2386980, 0.5735414, -0.3144956, 0.3062822
4: -11.7436352, -10.7965784, -11.7314425, -10.8032799, -0.3669760, 0.3752613
5: 7.6885214, 8.3822784, 7.6976628, 8.3776760, -0.2437965, 0.2516009
6: -6.3868170, -5.5767164, -6.3843861, -5.5822515, -0.2509687, 0.2474443
7: -15.9174757, -14.9309092, -15.9063101, -14.9500093, -0.4782546, 0.4712050
8: -3.8162069, -3.0977001, -3.8060327, -3.1032219, -0.2386736, 0.2412533
9: -3.6389809, -2.9697742, -3.6104789, -2.9848330, -0.2992085, 0.2963710

Time for backsubstitution: 21.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6232
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4657

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183498, upper bound: 0.1193760
time: 4.37 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1183498, upper bound: 0.1200832
time: 4.43 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -11.0022707, -10.2151604, -11.0022697, -10.2151585, -0.2797129, 0.2735069
1: -12.4187460, -11.6171675, -12.4187469, -11.6171665, -0.3504741, 0.3440135
2: -9.6405544, -8.8934736, -9.6405535, -8.8934708, -0.2676308, 0.2721052
3: -0.2502286, 0.5955539, -0.2502286, 0.5955565, -0.3166327, 0.3124212
4: -11.7436352, -10.7965784, -11.7436352, -10.7965765, -0.3683898, 0.3760211
5: 7.6885214, 8.3822784, 7.6885214, 8.3822765, -0.2445648, 0.2522774
6: -6.3868170, -5.5767164, -6.3868170, -5.5767155, -0.2523004, 0.2471324
7: -15.9174757, -14.9309092, -15.9174747, -14.9309101, -0.4743791, 0.4709420
8: -3.8162069, -3.0977001, -3.8162079, -3.0977006, -0.2375503, 0.2385432
9: -3.6389809, -2.9697742, -3.6389809, -2.9697747, -0.3036308, 0.2949646

Time for backsubstitution: 20.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6232
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4657

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183498, upper bound: 0.1193768
time: 3.73 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1183498, upper bound: 0.1200840
time: 3.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.52 seconds
NS_A2_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 28.52
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1193642
NS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 28.52
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1200712
NS_A2_A1_B2_A1, status: Status.VERIFIED, split count: 4, time: 28.52
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1193650
NS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 28.52
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1200722
NS_A2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 28.52
Output dim: 5, lower bound: -0.1183498, upper bound: 0.1193760
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 28.52
Output dim: 5, lower bound: -0.1183498, upper bound: 0.1200832
NS_A2_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 28.52
Output dim: 5, lower bound: -0.1183498, upper bound: 0.1193768
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 28.52
Output dim: 5, lower bound: -0.1183498, upper bound: 0.1200840

## BFS NS instance: NS_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -11.0015173, -10.2174273, -10.9959888, -10.2301550, -0.2760119, 0.2748655
1: -12.4191732, -11.6235256, -12.4131899, -11.6295738, -0.3402641, 0.3426864
2: -9.6355162, -8.8863087, -9.6370773, -8.8950386, -0.2661288, 0.2649449
3: -0.2489617, 0.5909069, -0.2386978, 0.5702665, -0.3087238, 0.3012021
4: -11.7375050, -10.7932510, -11.7275791, -10.8033247, -0.3686215, 0.3687454
5: 7.6918421, 8.3811483, 7.7012982, 8.3776245, -0.2436638, 0.2448632
6: -6.3942318, -5.5816393, -6.3843861, -5.5847850, -0.2471237, 0.2469386
7: -15.9182510, -14.9241943, -15.9062233, -14.9517698, -0.4776971, 0.4706807
8: -3.8241086, -3.0977125, -3.8048401, -3.1032505, -0.2340175, 0.2392710
9: -3.6382055, -2.9763370, -3.6102741, -2.9889655, -0.2926263, 0.2961662

Time for backsubstitution: 20.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6232
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6232

## Relational analysis of NS_A2_A1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1189482
time: 3.61 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1200714
time: 4.55 seconds

## BFS NS instance: NS_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -11.0015173, -10.2174273, -11.0022392, -10.2181759, -0.2767028, 0.2770327
1: -12.4191732, -11.6235256, -12.4187174, -11.6205769, -0.3414831, 0.3470950
2: -9.6355162, -8.8863087, -9.6380386, -8.8937740, -0.2665818, 0.2657502
3: -0.2489617, 0.5909069, -0.2502284, 0.5922832, -0.3108611, 0.3089454
4: -11.7375050, -10.7932510, -11.7397728, -10.7966223, -0.3699932, 0.3709054
5: 7.6918421, 8.3811483, 7.6921558, 8.3822241, -0.2486465, 0.2462897
6: -6.3942318, -5.5816393, -6.3868175, -5.5792484, -0.2488322, 0.2475120
7: -15.9182510, -14.9241943, -15.9173908, -14.9326706, -0.4777761, 0.4807770
8: -3.8241086, -3.0977125, -3.8150148, -3.0977278, -0.2369778, 0.2371804
9: -3.6382055, -2.9763370, -3.6387780, -2.9739053, -0.2987630, 0.2967696

Time for backsubstitution: 20.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6232
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6232

## Relational analysis of NS_A2_A1_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1189489
time: 3.58 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1200722
time: 3.49 seconds

## BFS NS instance: NS_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -11.0027542, -10.2111654, -10.9960155, -10.2271347, -0.2821081, 0.2749209
1: -12.4205751, -11.6164494, -12.4132223, -11.6261683, -0.3489838, 0.3433148
2: -9.6407080, -8.8848534, -9.6395798, -8.8947353, -0.2668757, 0.2737887
3: -0.2502260, 0.5976012, -0.2386978, 0.5735395, -0.3144145, 0.3064668
4: -11.7454958, -10.7916946, -11.7314377, -10.8032770, -0.3698232, 0.3756932
5: 7.6843019, 8.3826876, 7.6976628, 8.3776693, -0.2435155, 0.2518809
6: -6.3952160, -5.5764446, -6.3843861, -5.5822601, -0.2515109, 0.2474054
7: -15.9190655, -14.9205465, -15.9062967, -14.9500113, -0.4811016, 0.4710016
8: -3.8265004, -3.0972486, -3.8060341, -3.1032281, -0.2386899, 0.2413681
9: -3.6402416, -2.9677668, -3.6104774, -2.9848330, -0.3004857, 0.2962713

Time for backsubstitution: 20.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6232
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6232

## Relational analysis of NS_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183382, upper bound: 0.1189482
time: 7.35 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183386, upper bound: 0.1189482
time: 5.37 seconds

## BFS NS instance: NS_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -11.0027542, -10.2111654, -11.0022640, -10.2151594, -0.2823732, 0.2771009
1: -12.4205751, -11.6164494, -12.4187489, -11.6171684, -0.3502028, 0.3477528
2: -9.6407080, -8.8848534, -9.6405392, -8.8934717, -0.2673278, 0.2745939
3: -0.2502260, 0.5976012, -0.2502284, 0.5955558, -0.3165517, 0.3143555
4: -11.7454958, -10.7916946, -11.7436285, -10.7965755, -0.3712044, 0.3778532
5: 7.6843019, 8.3826876, 7.6885214, 8.3822718, -0.2485030, 0.2533072
6: -6.3952160, -5.5764446, -6.3868170, -5.5767255, -0.2532192, 0.2479922
7: -15.9190655, -14.9205465, -15.9174576, -14.9309082, -0.4804966, 0.4810927
8: -3.8265004, -3.0972486, -3.8162074, -3.0977049, -0.2418216, 0.2393373
9: -3.6402416, -2.9677668, -3.6389802, -2.9697747, -0.3064560, 0.2968726

Time for backsubstitution: 20.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6232
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6232

## Relational analysis of NS_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183382, upper bound: 0.1189490
time: 3.55 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183386, upper bound: 0.1189630
time: 3.49 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 28.18 seconds
NS_A2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 28.18
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1189482
NS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 28.18
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1200714
NS_A2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 28.18
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1189489
NS_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 28.18
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1200722
NS_A2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 28.18
Output dim: 5, lower bound: -0.1183382, upper bound: 0.1189482
NS_A2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 28.18
Output dim: 5, lower bound: -0.1183386, upper bound: 0.1189482
NS_A2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 28.18
Output dim: 5, lower bound: -0.1183382, upper bound: 0.1189490
NS_A2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 28.18
Output dim: 5, lower bound: -0.1183386, upper bound: 0.1189630

## BFS NS instance: NS_A2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -11.0015173, -10.2174273, -10.9960127, -10.2271376, -0.2760410, 0.2736654
1: -12.4191732, -11.6235256, -12.4132195, -11.6261673, -0.3402977, 0.3413545
2: -9.6355162, -8.8863087, -9.6395798, -8.8947468, -0.2661586, 0.2649448
3: -0.2489617, 0.5909069, -0.2386978, 0.5735366, -0.3087833, 0.2999364
4: -11.7375050, -10.7932510, -11.7314348, -10.8032770, -0.3671968, 0.3689640
5: 7.6918421, 8.3811483, 7.6976643, 8.3776684, -0.2422246, 0.2448632
6: -6.3942318, -5.5816393, -6.3843861, -5.5822630, -0.2471683, 0.2459476
7: -15.9182510, -14.9241943, -15.9062948, -14.9500113, -0.4777287, 0.4701302
8: -3.8241086, -3.0977125, -3.8060269, -3.1032281, -0.2336206, 0.2393231
9: -3.6382055, -2.9763370, -3.6104751, -2.9848328, -0.2926263, 0.2947371

Time for backsubstitution: 20.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4657

## Relational analysis of NS_A2_A1_B1_A2_B2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165079, upper bound: 0.1200713
time: 5.64 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165080, upper bound: 0.1200717
time: 3.69 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -11.0015173, -10.2174273, -11.0022621, -10.2151604, -0.2767317, 0.2770342
1: -12.4191732, -11.6235256, -12.4187450, -11.6171694, -0.3415165, 0.3463387
2: -9.6355162, -8.8863087, -9.6405411, -8.8934813, -0.2666111, 0.2657502
3: -0.2489617, 0.5909069, -0.2502284, 0.5955503, -0.3109206, 0.3082810
4: -11.7375050, -10.7932510, -11.7436256, -10.7965784, -0.3700304, 0.3711250
5: 7.6918421, 8.3811483, 7.6885228, 8.3822689, -0.2472533, 0.2462897
6: -6.3942318, -5.5816393, -6.3868175, -5.5767279, -0.2488768, 0.2475120
7: -15.9182510, -14.9241943, -15.9174595, -14.9309092, -0.4784734, 0.4808013
8: -3.8241086, -3.0977125, -3.8162060, -3.0977077, -0.2365830, 0.2388477
9: -3.6382055, -2.9763370, -3.6389794, -2.9697742, -0.2987630, 0.2962999

Time for backsubstitution: 20.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4657

## Relational analysis of NS_A2_A1_B2_A2_B2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165080, upper bound: 0.1200717
time: 4.36 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165079, upper bound: 0.1200722
time: 4.71 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 30.20 seconds
NS_A2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 30.20
Output dim: 5, lower bound: -0.1165079, upper bound: 0.1200713
NS_A2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 30.20
Output dim: 5, lower bound: -0.1165080, upper bound: 0.1200717
NS_A2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 30.20
Output dim: 5, lower bound: -0.1165080, upper bound: 0.1200717
NS_A2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 30.20
Output dim: 5, lower bound: -0.1165079, upper bound: 0.1200722

## BFS NS instance: NS_A2_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -11.0015173, -10.2174273, -10.9950638, -10.2271910, -0.2751667, 0.2724366
1: -12.4191732, -11.6235256, -12.4132128, -11.6265574, -0.3393505, 0.3403828
2: -9.6355162, -8.8863087, -9.6379213, -8.8947735, -0.2664821, 0.2632623
3: -0.2489617, 0.5909069, -0.2383361, 0.5734820, -0.3087443, 0.2995751
4: -11.7375050, -10.7932510, -11.7305536, -10.8034315, -0.3671794, 0.3683081
5: 7.6918421, 8.3811483, 7.6976643, 8.3766556, -0.2410443, 0.2444068
6: -6.3942318, -5.5816393, -6.3843865, -5.5839610, -0.2452877, 0.2456599
7: -15.9182510, -14.9241943, -15.9042397, -14.9500113, -0.4769906, 0.4675813
8: -3.8241086, -3.0977125, -3.8060265, -3.1052504, -0.2314806, 0.2391881
9: -3.6382055, -2.9763370, -3.6101818, -2.9848328, -0.2922927, 0.2940948

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6137

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1164780, upper bound: 0.1200707
time: 4.51 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165075, upper bound: 0.1200710
time: 5.37 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -11.0015173, -10.2174273, -10.9965029, -10.2231398, -0.2766336, 0.2749119
1: -12.4191732, -11.6235256, -12.4150486, -11.6254444, -0.3418579, 0.3413675
2: -9.6355162, -8.8863087, -9.6397457, -8.8861275, -0.2666667, 0.2649784
3: -0.2489617, 0.5909069, -0.2386956, 0.5755830, -0.3089681, 0.2999364
4: -11.7375050, -10.7932510, -11.7333088, -10.7983942, -0.3676283, 0.3719493
5: 7.6918421, 8.3811483, 7.6934447, 8.3780842, -0.2430828, 0.2448633
6: -6.3942318, -5.5816393, -6.3927851, -5.5819831, -0.2477571, 0.2459474
7: -15.9182510, -14.9241943, -15.9078836, -14.9396467, -0.4777606, 0.4738207
8: -3.8241086, -3.0977125, -3.8163199, -3.1027756, -0.2343267, 0.2393678
9: -3.6382055, -2.9763370, -3.6117463, -2.9828248, -0.2926263, 0.2962408

Time for backsubstitution: 20.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6137

## Relational analysis of NS_A2_A1_B1_A2_B2_B2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1164779, upper bound: 0.1200711
time: 4.46 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_B2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165075, upper bound: 0.1200710
time: 3.88 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -11.0015173, -10.2174273, -11.0013132, -10.2152119, -0.2758577, 0.2757281
1: -12.4191732, -11.6235256, -12.4187393, -11.6175613, -0.3405695, 0.3432915
2: -9.6355162, -8.8863087, -9.6388798, -8.8935118, -0.2669344, 0.2640675
3: -0.2489617, 0.5909069, -0.2498686, 0.5954981, -0.3108815, 0.3078709
4: -11.7375050, -10.7932510, -11.7427435, -10.7967319, -0.3701582, 0.3704703
5: 7.6918421, 8.3811483, 7.6885228, 8.3812551, -0.2459996, 0.2458332
6: -6.3942318, -5.5816393, -6.3868160, -5.5784245, -0.2469963, 0.2467649
7: -15.9182510, -14.9241943, -15.9154081, -14.9309120, -0.4759104, 0.4782271
8: -3.8241086, -3.0977125, -3.8162026, -3.0997286, -0.2344431, 0.2385424
9: -3.6382055, -2.9763370, -3.6386828, -2.9697742, -0.2983465, 0.2956580

Time for backsubstitution: 22.06 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.24 + 542.96 = 601.20 seconds
