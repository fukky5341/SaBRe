## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.6361788214999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7538667, 1.7538671)
1: (-16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5473294, 1.5473294)
2: (-7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5473089, 1.5473089)
3: (-12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.6015015, 2.6015015)
4: (-3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7983255, 1.7983255)
5: (-13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.2112970, 1.2112970)
6: (-15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7188849, 1.7188859)
7: (-7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1962242, 2.1962242)
8: (-6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5854363, 1.5854363)
9: (4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4497461, 1.4497461)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.15 + 33.91 = 58.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.6393757, upper bound: 0.6393755

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 4610
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 884

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6386561, upper bound: 0.6393753
time: 3.99 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393751, upper bound: 0.6386557
time: 4.04 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.05 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.05
Output dim: 9, lower bound: -0.6386561, upper bound: 0.6393753
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.05
Output dim: 9, lower bound: -0.6393751, upper bound: 0.6386557

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7532430, 1.7530336
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5473280, 1.5475101
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5468693, 1.5467215
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5987139, 2.5994177
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7939086, 1.7924194
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.2074332, 1.2084064
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7160068, 1.7167311
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1944962, 2.1949320
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5854363, 1.5854688
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4495697, 1.4496145

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 961
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 4610
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 906

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4608

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6380264, upper bound: 0.6393733
time: 3.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6386542, upper bound: 0.6387448
time: 4.08 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7530332, 1.7532430
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5475101, 1.5473280
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5467215, 1.5468693
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5994177, 2.5987139
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7924190, 1.7939081
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.2084064, 1.2074327
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7167315, 1.7160063
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1949320, 2.1944962
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5854688, 1.5854363
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4496145, 1.4495697

Time for backsubstitution: 22.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4610
type: DSZ, layer: 1, pos: 961
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 5875

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4610

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393739, upper bound: 0.6384042
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6391235, upper bound: 0.6386544
time: 3.98 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.06 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.06
Output dim: 9, lower bound: -0.6380264, upper bound: 0.6393733
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.06
Output dim: 9, lower bound: -0.6386542, upper bound: 0.6387448
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.06
Output dim: 9, lower bound: -0.6393739, upper bound: 0.6384042
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.06
Output dim: 9, lower bound: -0.6391235, upper bound: 0.6386544

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7521062, 1.7529502
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5471954, 1.5457001
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5467410, 1.5449929
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5984583, 2.5993977
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7936668, 1.7891150
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.2060189, 1.2082973
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7150493, 1.7166581
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1943092, 2.1949158
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5853596, 1.5843844
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4488230, 1.4495592

Time for backsubstitution: 22.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4610
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 961
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 5875

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4610

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6380252, upper bound: 0.6391214
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6377748, upper bound: 0.6393718
time: 4.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7531600, 1.7518969
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5455179, 1.5473776
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5451412, 1.5465932
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5986948, 2.5991621
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7906036, 1.7921758
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.2073236, 1.2069921
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7159343, 1.7157736
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1944799, 2.1947441
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5843520, 1.5853915
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4495149, 1.4488678

Time for backsubstitution: 21.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4610
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 961
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 5798

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4610

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6386529, upper bound: 0.6384932
time: 4.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6384025, upper bound: 0.6387435
time: 4.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7472811, 1.7451963
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5410185, 1.5382495
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5465379, 1.5466127
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5925426, 2.5938034
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7918262, 1.7930851
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.2025275, 1.1992126
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7160521, 1.7150564
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1941395, 2.1933851
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5840750, 1.5834842
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4489551, 1.4486475

Time for backsubstitution: 23.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 961
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 4557

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5875

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393730, upper bound: 0.6323019
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6332716, upper bound: 0.6384032
time: 4.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7449875, 1.7474909
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5384316, 1.5408363
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5464649, 1.5466857
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5945072, 2.5918398
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7915955, 1.7933154
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.2001858, 1.2015543
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7157812, 1.7153273
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1938200, 2.1937037
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5835161, 1.5840425
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4486923, 1.4489107

Time for backsubstitution: 23.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4557

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6391042, upper bound: 0.6316823
time: 4.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6321513, upper bound: 0.6386352
time: 3.98 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.61 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.61
Output dim: 9, lower bound: -0.6380252, upper bound: 0.6391214
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.61
Output dim: 9, lower bound: -0.6377748, upper bound: 0.6393718
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.61
Output dim: 9, lower bound: -0.6386529, upper bound: 0.6384932
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.61
Output dim: 9, lower bound: -0.6384025, upper bound: 0.6387435
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.61
Output dim: 9, lower bound: -0.6393730, upper bound: 0.6323019
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.61
Output dim: 9, lower bound: -0.6332716, upper bound: 0.6384032
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.61
Output dim: 9, lower bound: -0.6391042, upper bound: 0.6316823
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.61
Output dim: 9, lower bound: -0.6321513, upper bound: 0.6386352

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7463541, 1.7449036
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5407052, 1.5366235
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5465579, 1.5447369
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5915842, 2.5944881
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7930737, 1.7882905
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.2001400, 1.2000771
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7143698, 1.7157083
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1935167, 2.1938047
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5839643, 1.5824308
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4481640, 1.4486370

Time for backsubstitution: 23.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 961
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 5875

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 906

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6380249, upper bound: 0.6379062
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6368099, upper bound: 0.6391213
time: 3.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7440596, 1.7471981
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5381188, 1.5392098
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5464845, 1.5448098
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5935488, 2.5925236
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7928429, 1.7885208
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1977983, 1.2024183
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7140989, 1.7159791
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1931973, 2.1941233
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5834060, 1.5829892
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4479012, 1.4489002

Time for backsubstitution: 23.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 961
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 5875

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5798

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6373897, upper bound: 0.6393710
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6373914, upper bound: 0.6382685
time: 3.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7474079, 1.7438502
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5390277, 1.5383010
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5449576, 1.5463367
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5918207, 2.5942526
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7900105, 1.7913518
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.2014446, 1.1987715
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7152548, 1.7148237
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1936874, 2.1936331
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5829568, 1.5834384
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4488559, 1.4479456

Time for backsubstitution: 23.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 961
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 5875

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4557

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6386336, upper bound: 0.6315231
time: 4.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6316807, upper bound: 0.6384756
time: 4.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7451134, 1.7461448
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5364413, 1.5408874
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5448847, 1.5464096
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5937853, 2.5922880
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7897797, 1.7915821
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1991029, 1.2011132
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7149839, 1.7150946
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1933689, 2.1939526
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5823984, 1.5839968
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4485927, 1.4482088

Time for backsubstitution: 23.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 906

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6384022, upper bound: 0.6375283
time: 4.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6371873, upper bound: 0.6387435
time: 4.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.6814499, 1.6875992
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5148935, 1.5153894
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.4832683, 1.4912515
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5516071, 2.5579910
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7745905, 1.7731953
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1639180, 1.1654401
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.6700997, 1.6748519
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1726408, 2.1688156
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5336866, 1.5394020
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4142566, 1.4089894

Time for backsubstitution: 23.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 961
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 5798

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4608

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6387428, upper bound: 0.6323001
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393710, upper bound: 0.6316725
time: 3.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.6896839, 1.6793656
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5181580, 1.5121250
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.4911766, 1.4833431
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5567303, 2.5528679
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7719364, 1.7758489
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1687551, 1.1606030
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.6758466, 1.6691036
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1695700, 2.1718864
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5399928, 1.5330963
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4092970, 1.4139490

Time for backsubstitution: 23.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 961
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 4608

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4557

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6332524, upper bound: 0.6314313
time: 4.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6262997, upper bound: 0.6383840
time: 4.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7434568, 1.7464590
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5373058, 1.5398593
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5307837, 1.5329599
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5848475, 2.5833836
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7574854, 1.7658987
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1803761, 1.1789122
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7180476, 1.7179146
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1668510, 2.1570330
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5594454, 1.5565419
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4323163, 1.4301934

Time for backsubstitution: 23.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 961
type: DSZ, layer: 1, pos: 5875

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5798

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6380008, upper bound: 0.6312977
time: 4.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6391035, upper bound: 0.6312968
time: 4.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7439547, 1.7459602
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5374551, 1.5397100
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5327392, 1.5310049
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5860510, 2.5821800
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7641802, 1.7592049
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1775436, 1.1817446
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7183681, 1.7175941
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1571493, 2.1667337
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5560160, 1.5599713
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4299750, 1.4325347

Time for backsubstitution: 23.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 961
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 4608

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 961

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6320386, upper bound: 0.6337100
time: 3.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6272232, upper bound: 0.6385253
time: 3.91 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.07 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6380249, upper bound: 0.6379062
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6368099, upper bound: 0.6391213
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6373897, upper bound: 0.6393710
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6373914, upper bound: 0.6382685
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6386336, upper bound: 0.6315231
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6316807, upper bound: 0.6384756
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6384022, upper bound: 0.6375283
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6371873, upper bound: 0.6387435
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6387428, upper bound: 0.6323001
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6393710, upper bound: 0.6316725
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6332524, upper bound: 0.6314313
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6262997, upper bound: 0.6383840
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6380008, upper bound: 0.6312977
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6391035, upper bound: 0.6312968
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6320386, upper bound: 0.6337100
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 9, lower bound: -0.6272232, upper bound: 0.6385253

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7446389, 1.7429438
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5416756, 1.5379229
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5486007, 1.5473232
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5967665, 2.5989532
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7853460, 1.7794590
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1928401, 1.1917353
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7143660, 1.7159739
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1872101, 2.1865988
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5803080, 1.5800428
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4480581, 1.4485164

Time for backsubstitution: 23.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961
type: DSZ, layer: 1, pos: 5875

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4557

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6380056, upper bound: 0.6309203
time: 4.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6310531, upper bound: 0.6378886
time: 4.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7443948, 1.7431884
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5420051, 1.5375938
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5491443, 1.5467796
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5960503, 2.5996704
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7842407, 1.7805643
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1917987, 1.1927767
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7146349, 1.7157054
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1863108, 2.1874981
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5815763, 1.5787745
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4480433, 1.4485312

Time for backsubstitution: 23.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 5875

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5798

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6364248, upper bound: 0.6391201
time: 4.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6364265, upper bound: 0.6380179
time: 3.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7385001, 1.7408442
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5387263, 1.5399127
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5400548, 1.5374613
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5872345, 2.5882511
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7741742, 1.7653131
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1740856, 1.1816719
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7094560, 1.7136073
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1841183, 2.1861801
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5864205, 1.5864739
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4463649, 1.4475555

Time for backsubstitution: 23.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4557

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6373705, upper bound: 0.6323987
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6304178, upper bound: 0.6393524
time: 4.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7377057, 1.7416387
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5388212, 1.5398173
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5391364, 1.5383801
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5892763, 2.5862093
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7696347, 1.7698526
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1770515, 1.1787055
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7117257, 1.7113361
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1852541, 2.1850452
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5868907, 1.5860038
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4465566, 1.4473639

Time for backsubstitution: 23.19 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.06 + 561.54 = 619.60 seconds
