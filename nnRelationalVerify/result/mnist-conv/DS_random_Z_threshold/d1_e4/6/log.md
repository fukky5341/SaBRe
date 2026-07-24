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
execution time: IAR + RelationalAnalysis = 22.94 + 32.46 = 55.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0956647, upper bound: 0.0956647

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 456

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 51

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0948935, upper bound: 0.0948479
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0948479, upper bound: 0.0948934
time: 3.34 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.62 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.62
Output dim: 1, lower bound: -0.0948935, upper bound: 0.0948479
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.62
Output dim: 1, lower bound: -0.0948479, upper bound: 0.0948934

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.5056384, 0.5056396
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2166888, 0.2166867
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2312568, 0.2312572
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2746437, 0.2746441
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2315681, 0.2315657
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1551014, 0.1551026
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3535559, 0.3535557
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2603815, 0.2603817
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2330493, 0.2330495
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3167470, 0.3167484

Time for backsubstitution: 21.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 456

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 456

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0948900, upper bound: 0.0948334
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0948382, upper bound: 0.0948357
time: 3.34 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.5056384, 0.5056386
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2166867, 0.2166868
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2312572, 0.2312573
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2746441, 0.2746441
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2315657, 0.2315657
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1551014, 0.1551014
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3535559, 0.3535559
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2603815, 0.2603815
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2330493, 0.2330493
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3167467, 0.3167470

Time for backsubstitution: 21.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 456

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 456

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0948358, upper bound: 0.0948382
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0948335, upper bound: 0.0948901
time: 2.96 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.30 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.30
Output dim: 1, lower bound: -0.0948900, upper bound: 0.0948334
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.30
Output dim: 1, lower bound: -0.0948382, upper bound: 0.0948357
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.30
Output dim: 1, lower bound: -0.0948358, upper bound: 0.0948382
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.30
Output dim: 1, lower bound: -0.0948335, upper bound: 0.0948901

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

Time for backsubstitution: 21.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1856
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 961

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 600

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0948784, upper bound: 0.0931377
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0931971, upper bound: 0.0948226
time: 3.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2

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

Time for backsubstitution: 21.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1856
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 600

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1776

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0913673, upper bound: 0.0948017
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0948042, upper bound: 0.0913681
time: 3.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1

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

Time for backsubstitution: 21.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1856
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 2137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2559

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0868664, upper bound: 0.0901111
time: 4.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0901068, upper bound: 0.0868649
time: 3.17 seconds

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

Time for backsubstitution: 21.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1856
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2559

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 157

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0945015, upper bound: 0.0948724
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0948151, upper bound: 0.0945578
time: 3.10 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 27.81 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.81
Output dim: 1, lower bound: -0.0948784, upper bound: 0.0931377
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.81
Output dim: 1, lower bound: -0.0931971, upper bound: 0.0948226
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.81
Output dim: 1, lower bound: -0.0913673, upper bound: 0.0948017
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.81
Output dim: 1, lower bound: -0.0948042, upper bound: 0.0913681
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 27.81
Output dim: 1, lower bound: -0.0868664, upper bound: 0.0901111
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 27.81
Output dim: 1, lower bound: -0.0901068, upper bound: 0.0868649
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.81
Output dim: 1, lower bound: -0.0945015, upper bound: 0.0948724
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.81
Output dim: 1, lower bound: -0.0948151, upper bound: 0.0945578

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4996502, 0.4989874
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2145597, 0.2135431
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2254446, 0.2241906
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2733843, 0.2742674
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2321533, 0.2319684
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1391304, 0.1384701
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3520110, 0.3514731
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2614353, 0.2626419
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2246207, 0.2251918
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3166068, 0.3168576

Time for backsubstitution: 21.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1856
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 907

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 226

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0945087, upper bound: 0.0915204
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0932612, upper bound: 0.0927681
time: 2.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4997070, 0.4989307
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2145540, 0.2135488
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2245309, 0.2251040
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2737370, 0.2739148
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2321136, 0.2320082
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1394014, 0.1381990
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3520334, 0.3514509
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2614343, 0.2626429
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2244076, 0.2254047
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3167596, 0.3167045

Time for backsubstitution: 21.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1856
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1856

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0924831, upper bound: 0.0948214
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0931958, upper bound: 0.0941088
time: 3.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4971461, 0.4976315
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2123113, 0.2134007
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2286212, 0.2289732
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2745967, 0.2741189
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2317227, 0.2318959
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1470965, 0.1480485
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3526292, 0.3531725
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2636220, 0.2624137
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2259479, 0.2249952
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3211694, 0.3208022

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1856
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1487

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 600

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0913550, upper bound: 0.0931035
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0889042, upper bound: 0.0947911
time: 3.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4969125, 0.4978657
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2123938, 0.2133182
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2286338, 0.2289606
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2746482, 0.2740674
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2317563, 0.2318624
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1471177, 0.1480273
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3526137, 0.3531880
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2636206, 0.2624152
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2257787, 0.2251644
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3208973, 0.3210742

Time for backsubstitution: 21.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1856
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1151

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0948022, upper bound: 0.0821783
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0856134, upper bound: 0.0913662
time: 3.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4998181, 0.5005455
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2136770, 0.2147029
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2283995, 0.2287208
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2750812, 0.2745008
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2321031, 0.2322270
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1474524, 0.1484085
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3514614, 0.3519907
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2627213, 0.2614830
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2274359, 0.2266299
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3167348, 0.3167069

Time for backsubstitution: 21.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1856
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 600

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2559

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0864820, upper bound: 0.0901231
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0898455, upper bound: 0.0869272
time: 4.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4998243, 0.5005388
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2136941, 0.2146850
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2283816, 0.2287376
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2750278, 0.2745514
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2320848, 0.2322443
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1474749, 0.1483847
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3514309, 0.3520195
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2626879, 0.2615144
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2274138, 0.2266508
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3167996, 0.3166385

Time for backsubstitution: 21.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1856
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1494

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0923015, upper bound: 0.0924476
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0926903, upper bound: 0.0923582
time: 3.36 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.50
Output dim: 1, lower bound: -0.0945087, upper bound: 0.0915204
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.50
Output dim: 1, lower bound: -0.0932612, upper bound: 0.0927681
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.50
Output dim: 1, lower bound: -0.0924831, upper bound: 0.0948214
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.50
Output dim: 1, lower bound: -0.0931958, upper bound: 0.0941088
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.50
Output dim: 1, lower bound: -0.0913550, upper bound: 0.0931035
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.50
Output dim: 1, lower bound: -0.0889042, upper bound: 0.0947911
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.50
Output dim: 1, lower bound: -0.0948022, upper bound: 0.0821783
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.50
Output dim: 1, lower bound: -0.0856134, upper bound: 0.0913662
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.50
Output dim: 1, lower bound: -0.0864820, upper bound: 0.0901231
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.50
Output dim: 1, lower bound: -0.0898455, upper bound: 0.0869272
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.50
Output dim: 1, lower bound: -0.0923015, upper bound: 0.0924476
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.50
Output dim: 1, lower bound: -0.0926903, upper bound: 0.0923582

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 21.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1856
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2559

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1151

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0945067, upper bound: 0.0835203
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0853562, upper bound: 0.0915170
time: 3.40 seconds

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

Time for backsubstitution: 21.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1856
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0908512, upper bound: 0.0907288
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0912336, upper bound: 0.0905451
time: 6.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.5001547, 0.4994042
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2136666, 0.2129025
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2288605, 0.2286017
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2736917, 0.2742946
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2323569, 0.2322004
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1484947, 0.1476153
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3519616, 0.3514462
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2609980, 0.2622228
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2262231, 0.2271250
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3171225, 0.3171754

Time for backsubstitution: 21.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2559

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0844651, upper bound: 0.0900954
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0878511, upper bound: 0.0868494
time: 3.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.5001242, 0.4994347
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2139136, 0.2126555
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2289422, 0.2285199
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2737641, 0.2742224
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2323456, 0.2322116
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1485466, 0.1475634
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3520062, 0.3514016
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2610154, 0.2622054
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2263410, 0.2270072
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3170776, 0.3172204

Time for backsubstitution: 22.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1487

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2559

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0851779, upper bound: 0.0893827
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0885638, upper bound: 0.0861367
time: 3.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4989297, 0.4997056
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2135508, 0.2145519
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2251036, 0.2245297
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2739143, 0.2737374
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2320106, 0.2321105
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1381979, 0.1393996
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3514512, 0.3520322
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2626426, 0.2614346
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2254044, 0.2244080
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3167031, 0.3167610

Time for backsubstitution: 22.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1856
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1487

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0890513, upper bound: 0.0910653
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0893158, upper bound: 0.0907438
time: 3.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.4989865, 0.4996488
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2135450, 0.2145576
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2241900, 0.2254430
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2742670, 0.2733848
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2319709, 0.2321502
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1384689, 0.1391289
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3514733, 0.3520100
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2626417, 0.2614355
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2251915, 0.2246209
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3168561, 0.3166080

Time for backsubstitution: 22.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 1856
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1758

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0866009, upper bound: 0.0927528
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0868649, upper bound: 0.0924313
time: 3.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.5076332, 0.5093930
1: 3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2138205, 0.2147677
2: -4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2262185, 0.2265775
3: -12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2782924, 0.2790184
4: -2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2322284, 0.2323711
5: -9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1464671, 0.1478825
6: -6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3509533, 0.3517091
7: -3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2616327, 0.2602489
8: -2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2354721, 0.2367045
9: -12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3162634, 0.3163536

Time for backsubstitution: 23.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 886
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1856
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1758
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 961

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 886

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0931398, upper bound: 0.0821473
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0918261, upper bound: 0.0821332
time: 3.26 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 30.07 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.07
Output dim: 1, lower bound: -0.0945067, upper bound: 0.0835203
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.07
Output dim: 1, lower bound: -0.0853562, upper bound: 0.0915170
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.07
Output dim: 1, lower bound: -0.0908512, upper bound: 0.0907288
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.07
Output dim: 1, lower bound: -0.0912336, upper bound: 0.0905451
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.07
Output dim: 1, lower bound: -0.0844651, upper bound: 0.0900954
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.07
Output dim: 1, lower bound: -0.0878511, upper bound: 0.0868494
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.07
Output dim: 1, lower bound: -0.0851779, upper bound: 0.0893827
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.07
Output dim: 1, lower bound: -0.0885638, upper bound: 0.0861367
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.07
Output dim: 1, lower bound: -0.0890513, upper bound: 0.0910653
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.07
Output dim: 1, lower bound: -0.0893158, upper bound: 0.0907438
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.07
Output dim: 1, lower bound: -0.0866009, upper bound: 0.0927528
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.07
Output dim: 1, lower bound: -0.0868649, upper bound: 0.0924313
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.07
Output dim: 1, lower bound: -0.0931398, upper bound: 0.0821473
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.07
Output dim: 1, lower bound: -0.0918261, upper bound: 0.0821332
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.07
Output dim: 1, lower bound: -0.0923015, upper bound: 0.0924476
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.07
Output dim: 1, lower bound: -0.0926903, upper bound: 0.0923582

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 55.40 + 550.00 = 605.40 seconds
