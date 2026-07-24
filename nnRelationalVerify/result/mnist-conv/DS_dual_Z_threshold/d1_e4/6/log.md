## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.091838112


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.5056384, 0.5056384)
1: (3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2166868, 0.2166868)
2: (-4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2312573, 0.2312573)
3: (-12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2746441, 0.2746441)
4: (-2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2315657, 0.2315657)
5: (-9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1551014, 0.1551014)
6: (-6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3535559, 0.3535559)
7: (-3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2603815, 0.2603815)
8: (-2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2330493, 0.2330492)
9: (-12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3167467, 0.3167470)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.24 + 33.83 = 56.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0956647, upper bound: 0.0956647

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 456
type: DSZ, layer: 1, pos: 51

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 456

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0956636, upper bound: 0.0948427
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0948429, upper bound: 0.0956637
time: 3.36 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.59 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.59
Output dim: 1, lower bound: -0.0956636, upper bound: 0.0948427
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.59
Output dim: 1, lower bound: -0.0948429, upper bound: 0.0956637

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.5005915, 0.4998708
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2145908, 0.2135820
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2288620, 0.2285230
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2745030, 0.2750328
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2323772, 0.2322351
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1485742, 0.1476417
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3522475, 0.3516874
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2613072, 0.2625144
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2267742, 0.2275579
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3171992, 0.3172958

Time for backsubstitution: 21.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 51

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 51

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0948900, upper bound: 0.0948334
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0948358, upper bound: 0.0948382
time: 3.19 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4998705, 0.5005913
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2135820, 0.2145908
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2285230, 0.2288620
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2750328, 0.2745030
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2322352, 0.2323772
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1476418, 0.1485742
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3516874, 0.3522475
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2625144, 0.2613072
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2275579, 0.2267741
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3172958, 0.3171992

Time for backsubstitution: 21.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 51

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 51

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0948382, upper bound: 0.0948357
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0948334, upper bound: 0.0948901
time: 3.41 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.53 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.53
Output dim: 1, lower bound: -0.0948900, upper bound: 0.0948334
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.53
Output dim: 1, lower bound: -0.0948358, upper bound: 0.0948382
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.53
Output dim: 1, lower bound: -0.0948382, upper bound: 0.0948357
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.53
Output dim: 1, lower bound: -0.0948334, upper bound: 0.0948901

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.5005915, 0.4998722
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2145929, 0.2135821
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2288631, 0.2285225
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2745020, 0.2750328
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2323804, 0.2322352
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1485742, 0.1476429
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3522472, 0.3516872
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2613072, 0.2625146
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2267740, 0.2275581
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3171992, 0.3172972

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0919170, upper bound: 0.0929239
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0929806, upper bound: 0.0918604
time: 4.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.5005915, 0.4998713
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2145910, 0.2135820
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2288616, 0.2285230
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2745025, 0.2750328
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2323773, 0.2322351
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1485742, 0.1476417
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3522475, 0.3516874
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2613072, 0.2625141
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2267742, 0.2275579
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3171992, 0.3172958

Time for backsubstitution: 22.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.48 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0918627, upper bound: 0.0929288
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0929266, upper bound: 0.0918651
time: 3.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4998715, 0.5005903
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2135841, 0.2145910
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2285222, 0.2288616
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2750323, 0.2745025
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2322377, 0.2323773
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1476418, 0.1485724
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3516872, 0.3522460
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2625141, 0.2613072
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2275579, 0.2267743
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3172958, 0.3172009

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0918651, upper bound: 0.0929264
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0929290, upper bound: 0.0918628
time: 3.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4998705, 0.5005913
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2135820, 0.2145908
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2285225, 0.2288620
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2750328, 0.2745030
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2322352, 0.2323772
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1476418, 0.1485742
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3516874, 0.3522472
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2625144, 0.2613072
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2275579, 0.2267741
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3172958, 0.3171992

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0918604, upper bound: 0.0929807
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0929243, upper bound: 0.0919171
time: 3.36 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.62 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 1, lower bound: -0.0919170, upper bound: 0.0929239
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 1, lower bound: -0.0929806, upper bound: 0.0918604
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 1, lower bound: -0.0918627, upper bound: 0.0929288
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 1, lower bound: -0.0929266, upper bound: 0.0918651
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 1, lower bound: -0.0918651, upper bound: 0.0929264
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 1, lower bound: -0.0929290, upper bound: 0.0918628
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 1, lower bound: -0.0918604, upper bound: 0.0929807
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 1, lower bound: -0.0929243, upper bound: 0.0919171

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4030070, 0.4012477
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2115061, 0.2109334
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2151957, 0.2162975
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.1914196, 0.1862881
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.1965159, 0.2045437
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1513498, 0.1504093
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3701923, 0.3685327
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2508154, 0.2532628
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2003539, 0.1986964
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3308926, 0.3378263

Time for backsubstitution: 20.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 226

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0916238, upper bound: 0.0917697
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0916580, upper bound: 0.0925816
time: 3.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4019675, 0.4022875
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2119443, 0.2104952
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2166381, 0.2148551
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.1857575, 0.1919502
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2046888, 0.1963707
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1513407, 0.1504185
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3690927, 0.3696322
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2520554, 0.2520230
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.1979123, 0.2011380
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3377283, 0.3309906

Time for backsubstitution: 21.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 226

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0926382, upper bound: 0.0916014
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0918263, upper bound: 0.0915671
time: 3.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4030070, 0.4012470
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2115041, 0.2109334
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2151941, 0.2162980
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.1914202, 0.1862882
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.1965128, 0.2045437
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1513497, 0.1504081
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3701923, 0.3685327
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2508156, 0.2532625
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2003539, 0.1986962
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3308928, 0.3378248

Time for backsubstitution: 22.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 226

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0915695, upper bound: 0.0917745
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0916037, upper bound: 0.0925864
time: 3.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4019675, 0.4022865
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2119423, 0.2104952
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2166365, 0.2148556
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.1857581, 0.1919503
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2046857, 0.1963707
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1513407, 0.1504173
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3690925, 0.3696322
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2520554, 0.2520225
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.1979125, 0.2011377
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3377285, 0.3309891

Time for backsubstitution: 21.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 226

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0925839, upper bound: 0.0916062
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0917722, upper bound: 0.0915719
time: 3.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4022865, 0.4019661
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2104971, 0.2119423
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2148547, 0.2166365
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.1919495, 0.1857581
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.1963732, 0.2046858
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1504173, 0.1513389
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3696322, 0.3690915
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2520227, 0.2520556
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2011377, 0.1979126
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3309891, 0.3377299

Time for backsubstitution: 22.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 226

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0915718, upper bound: 0.0917720
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0916061, upper bound: 0.0925839
time: 3.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4012470, 0.4030056
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2109355, 0.2115041
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2162971, 0.2151941
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.1862874, 0.1914202
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2045461, 0.1965128
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1504082, 0.1513480
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3685327, 0.3701911
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2532625, 0.2508156
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.1986961, 0.2003542
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3378248, 0.3308940

Time for backsubstitution: 22.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 226

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0925865, upper bound: 0.0916038
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0917746, upper bound: 0.0915695
time: 3.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4022865, 0.4019673
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2104952, 0.2119423
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2148551, 0.2166370
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.1919502, 0.1857582
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.1963707, 0.2046858
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1504173, 0.1513406
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3696322, 0.3690927
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2520225, 0.2520554
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2011378, 0.1979124
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3309891, 0.3377283

Time for backsubstitution: 21.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 226

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0915671, upper bound: 0.0918264
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0916014, upper bound: 0.0926383
time: 3.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4012470, 0.4030070
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2109334, 0.2115041
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2162975, 0.2151946
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.1862881, 0.1914203
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2045436, 0.1965128
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1504081, 0.1513498
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3685327, 0.3701923
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2532628, 0.2508154
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.1986961, 0.2003539
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3378251, 0.3308926

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 226

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0925818, upper bound: 0.0916581
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0917698, upper bound: 0.0916238
time: 3.16 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0916238, upper bound: 0.0917697
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0916580, upper bound: 0.0925816
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0926382, upper bound: 0.0916014
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0918263, upper bound: 0.0915671
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0915695, upper bound: 0.0917745
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0916037, upper bound: 0.0925864
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0925839, upper bound: 0.0916062
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0917722, upper bound: 0.0915719
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0915718, upper bound: 0.0917720
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0916061, upper bound: 0.0925839
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0925865, upper bound: 0.0916038
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0917746, upper bound: 0.0915695
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0915671, upper bound: 0.0918264
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0916014, upper bound: 0.0926383
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0925818, upper bound: 0.0916581
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.0917698, upper bound: 0.0916238

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4970298, 0.4966257
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2149887, 0.2140164
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2269293, 0.2268569
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2765195, 0.2772353
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2412957, 0.2409669
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1204755, 0.1199729
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3547373, 0.3542576
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2543111, 0.2567420
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2129997, 0.2123279
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3053148, 0.3059320

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Candidate
type: DSZ, layer: 3, pos: 2578

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0889847, upper bound: 0.0907956
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0890546, upper bound: 0.0889241
time: 3.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4973454, 0.4963105
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2150273, 0.2139778
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2271974, 0.2265888
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2767048, 0.2770500
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2411121, 0.2411505
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1209042, 0.1195442
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3548176, 0.3541772
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2555346, 0.2555184
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2115438, 0.2137839
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3058341, 0.3054128

Time for backsubstitution: 21.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Candidate
type: DSZ, layer: 3, pos: 2578

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0889808, upper bound: 0.0889979
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0908522, upper bound: 0.0889281
time: 3.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4970298, 0.4966247
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2149867, 0.2140166
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2269279, 0.2268572
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2765200, 0.2772355
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2412926, 0.2409669
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1204755, 0.1199717
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3547373, 0.3542576
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2543111, 0.2567420
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2129998, 0.2123277
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3053148, 0.3059306

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Candidate
type: DSZ, layer: 3, pos: 2578

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0889307, upper bound: 0.0908003
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0890002, upper bound: 0.0889289
time: 3.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4973454, 0.4963095
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2150253, 0.2139779
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2271960, 0.2265891
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2767053, 0.2770503
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2411089, 0.2411506
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1209042, 0.1195430
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3548176, 0.3541772
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2555349, 0.2555180
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2115438, 0.2137837
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3058338, 0.3054113

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1856

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Candidate
type: DSZ, layer: 3, pos: 2578

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0889264, upper bound: 0.0890027
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0907979, upper bound: 0.0889328
time: 3.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4963098, 0.4973443
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2139798, 0.2150253
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2265884, 0.2271960
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2770495, 0.2767053
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2411530, 0.2411089
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1195430, 0.1209024
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3541772, 0.3548164
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2555180, 0.2555351
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2137835, 0.2115441
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3054113, 0.3058355

Time for backsubstitution: 21.80 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.07 + 544.55 = 600.61 seconds
