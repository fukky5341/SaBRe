## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0019356300000000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0069460, 0.0069460)
1: (-0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0019583, 0.0019583)
2: (-0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0144491, 0.0144491)
3: (0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0019121, 0.0019121)
4: (0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0107983, 0.0107983)
5: (0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0030001, 0.0030001)
6: (0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0027232, 0.0027232)
7: (-0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0101624, 0.0101624)
8: (-0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0079094, 0.0079094)
9: (-0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006824, 0.0006824)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.75 + 2.95 = 4.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0021507, upper bound: 0.0021506

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021094, upper bound: 0.0020113
time: 1.46 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021129, upper bound: 0.0021129
time: 1.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.55 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.55
Output dim: 5, lower bound: -0.0021094, upper bound: 0.0020113
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.55
Output dim: 5, lower bound: -0.0021129, upper bound: 0.0021129

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0132763, -0.0053062, -0.0136221, -0.0052798, -0.0064658, 0.0067521
1: -0.0066817, -0.0044347, -0.0067792, -0.0044272, -0.0018230, 0.0019037
2: -0.0107395, 0.0058399, -0.0114589, 0.0058948, -0.0134502, 0.0140457
3: 0.0002061, 0.0024001, 0.0001109, 0.0024074, -0.0017799, 0.0018587
4: 0.0017275, 0.0141179, 0.0016865, 0.0146555, -0.0104969, 0.0100518
5: 0.9959862, 0.9994286, 0.9959748, 0.9995780, -0.0029163, 0.0027927
6: 0.0042403, 0.0073650, 0.0042300, 0.0075006, -0.0026472, 0.0025349
7: -0.0075574, 0.0041034, -0.0075960, 0.0046093, -0.0098787, 0.0094599
8: -0.0123865, -0.0033110, -0.0127803, -0.0032809, -0.0073626, 0.0076886
9: -0.0037241, -0.0029411, -0.0037267, -0.0029071, -0.0006633, 0.0006352

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020113, upper bound: 0.0020113
time: 1.98 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020113, upper bound: 0.0020112
time: 2.11 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0137021, -0.0052661, -0.0137769, -0.0052372, -0.0065854, 0.0069237
1: -0.0068018, -0.0044234, -0.0068229, -0.0044152, -0.0018567, 0.0019520
2: -0.0116253, 0.0059232, -0.0117809, 0.0059833, -0.0136989, 0.0144027
3: 0.0000889, 0.0024111, 0.0000683, 0.0024191, -0.0018128, 0.0019060
4: 0.0016652, 0.0147799, 0.0016203, 0.0148962, -0.0107637, 0.0102377
5: 0.9959689, 0.9996125, 0.9959564, 0.9996448, -0.0029905, 0.0028443
6: 0.0042246, 0.0075319, 0.0042133, 0.0075613, -0.0027144, 0.0025818
7: -0.0076160, 0.0047264, -0.0076583, 0.0048358, -0.0101298, 0.0096348
8: -0.0128714, -0.0032653, -0.0129566, -0.0032324, -0.0074988, 0.0078840
9: -0.0037280, -0.0028993, -0.0037309, -0.0028919, -0.0006802, 0.0006470

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020113, upper bound: 0.0021094
time: 1.94 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020113, upper bound: 0.0021129
time: 2.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.70 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.70
Output dim: 5, lower bound: -0.0020113, upper bound: 0.0020113
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.70
Output dim: 5, lower bound: -0.0020113, upper bound: 0.0020112
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.70
Output dim: 5, lower bound: -0.0020113, upper bound: 0.0021094
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.70
Output dim: 5, lower bound: -0.0020113, upper bound: 0.0021129

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0132763, -0.0053062, -0.0132763, -0.0053062, -0.0064572, 0.0064572
1: -0.0066817, -0.0044347, -0.0066817, -0.0044347, -0.0018205, 0.0018205
2: -0.0107395, 0.0058399, -0.0107395, 0.0058399, -0.0134323, 0.0134323
3: 0.0002061, 0.0024001, 0.0002061, 0.0024001, -0.0017776, 0.0017776
4: 0.0017275, 0.0141179, 0.0017275, 0.0141179, -0.0100385, 0.0100385
5: 0.9959862, 0.9994286, 0.9959862, 0.9994286, -0.0027890, 0.0027890
6: 0.0042403, 0.0073650, 0.0042403, 0.0073650, -0.0025316, 0.0025316
7: -0.0075574, 0.0041034, -0.0075574, 0.0041034, -0.0094473, 0.0094473
8: -0.0123865, -0.0033110, -0.0123865, -0.0033110, -0.0073529, 0.0073529
9: -0.0037241, -0.0029411, -0.0037241, -0.0029411, -0.0006344, 0.0006344

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019377, upper bound: 0.0019260
time: 1.99 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019350, upper bound: 0.0019299
time: 1.95 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0132763, -0.0053062, -0.0137021, -0.0052661, -0.0064764, 0.0068319
1: -0.0066817, -0.0044347, -0.0068018, -0.0044234, -0.0018259, 0.0019262
2: -0.0107395, 0.0058399, -0.0116253, 0.0059232, -0.0134723, 0.0142117
3: 0.0002061, 0.0024001, 0.0000889, 0.0024111, -0.0017828, 0.0018807
4: 0.0017275, 0.0141179, 0.0016652, 0.0147799, -0.0106209, 0.0100683
5: 0.9959862, 0.9994286, 0.9959689, 0.9996125, -0.0029508, 0.0027973
6: 0.0042403, 0.0073650, 0.0042246, 0.0075319, -0.0026784, 0.0025391
7: -0.0075574, 0.0041034, -0.0076160, 0.0047264, -0.0099955, 0.0094754
8: -0.0123865, -0.0033110, -0.0128714, -0.0032653, -0.0073747, 0.0077795
9: -0.0037241, -0.0029411, -0.0037280, -0.0028993, -0.0006712, 0.0006363

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019377, upper bound: 0.0019261
time: 2.19 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019350, upper bound: 0.0019299
time: 2.29 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0137021, -0.0052661, -0.0132763, -0.0053062, -0.0068319, 0.0064764
1: -0.0068018, -0.0044234, -0.0066817, -0.0044347, -0.0019262, 0.0018259
2: -0.0116253, 0.0059232, -0.0107395, 0.0058399, -0.0142117, 0.0134723
3: 0.0000889, 0.0024111, 0.0002061, 0.0024001, -0.0018807, 0.0017828
4: 0.0016652, 0.0147799, 0.0017275, 0.0141179, -0.0100683, 0.0106209
5: 0.9959689, 0.9996125, 0.9959862, 0.9994286, -0.0027973, 0.0029508
6: 0.0042246, 0.0075319, 0.0042403, 0.0073650, -0.0025391, 0.0026784
7: -0.0076160, 0.0047264, -0.0075574, 0.0041034, -0.0094754, 0.0099954
8: -0.0128714, -0.0032653, -0.0123865, -0.0033110, -0.0077795, 0.0073747
9: -0.0037280, -0.0028993, -0.0037241, -0.0029411, -0.0006363, 0.0006712

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019305, upper bound: 0.0020301
time: 2.26 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019299, upper bound: 0.0020308
time: 1.78 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0137021, -0.0052661, -0.0137021, -0.0052661, -0.0065617, 0.0065617
1: -0.0068018, -0.0044234, -0.0068018, -0.0044234, -0.0018500, 0.0018500
2: -0.0116253, 0.0059232, -0.0116253, 0.0059232, -0.0136496, 0.0136496
3: 0.0000889, 0.0024111, 0.0000889, 0.0024111, -0.0018063, 0.0018063
4: 0.0016652, 0.0147799, 0.0016652, 0.0147799, -0.0102009, 0.0102009
5: 0.9959689, 0.9996125, 0.9959689, 0.9996125, -0.0028341, 0.0028341
6: 0.0042246, 0.0075319, 0.0042246, 0.0075319, -0.0025725, 0.0025725
7: -0.0076160, 0.0047264, -0.0076160, 0.0047264, -0.0096002, 0.0096002
8: -0.0128714, -0.0032653, -0.0128714, -0.0032653, -0.0074718, 0.0074718
9: -0.0037280, -0.0028993, -0.0037280, -0.0028993, -0.0006446, 0.0006446

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019305, upper bound: 0.0020359
time: 1.94 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019299, upper bound: 0.0020373
time: 2.19 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.04 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.04
Output dim: 5, lower bound: -0.0019377, upper bound: 0.0019260
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 6.04
Output dim: 5, lower bound: -0.0019350, upper bound: 0.0019299
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.04
Output dim: 5, lower bound: -0.0019377, upper bound: 0.0019261
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 6.04
Output dim: 5, lower bound: -0.0019350, upper bound: 0.0019299
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.04
Output dim: 5, lower bound: -0.0019305, upper bound: 0.0020301
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.04
Output dim: 5, lower bound: -0.0019299, upper bound: 0.0020308
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.04
Output dim: 5, lower bound: -0.0019305, upper bound: 0.0020359
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.04
Output dim: 5, lower bound: -0.0019299, upper bound: 0.0020373

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0131583, -0.0053365, -0.0132763, -0.0053062, -0.0063433, 0.0064315
1: -0.0066485, -0.0044432, -0.0066817, -0.0044347, -0.0017884, 0.0018133
2: -0.0104941, 0.0057769, -0.0107395, 0.0058399, -0.0131954, 0.0133789
3: 0.0002386, 0.0023918, 0.0002061, 0.0024001, -0.0017462, 0.0017705
4: 0.0017746, 0.0139345, 0.0017275, 0.0141179, -0.0099985, 0.0098614
5: 0.9959993, 0.9993777, 0.9959862, 0.9994286, -0.0027779, 0.0027398
6: 0.0042522, 0.0073187, 0.0042403, 0.0073650, -0.0025215, 0.0024869
7: -0.0075131, 0.0039308, -0.0075574, 0.0041034, -0.0094097, 0.0092807
8: -0.0122522, -0.0033454, -0.0123865, -0.0033110, -0.0072232, 0.0073236
9: -0.0037211, -0.0029527, -0.0037241, -0.0029411, -0.0006318, 0.0006232

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019310, upper bound: 0.0019309
time: 2.02 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019310, upper bound: 0.0019310
time: 2.19 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0131583, -0.0053365, -0.0137021, -0.0052661, -0.0063625, 0.0068062
1: -0.0066485, -0.0044432, -0.0068018, -0.0044234, -0.0017938, 0.0019189
2: -0.0104941, 0.0057769, -0.0116253, 0.0059232, -0.0132353, 0.0141582
3: 0.0002386, 0.0023918, 0.0000889, 0.0024111, -0.0017515, 0.0018736
4: 0.0017746, 0.0139345, 0.0016652, 0.0147799, -0.0105810, 0.0098912
5: 0.9959993, 0.9993777, 0.9959689, 0.9996125, -0.0029397, 0.0027481
6: 0.0042522, 0.0073187, 0.0042246, 0.0075319, -0.0026684, 0.0024944
7: -0.0075131, 0.0039308, -0.0076160, 0.0047264, -0.0099579, 0.0093088
8: -0.0122522, -0.0033454, -0.0128714, -0.0032653, -0.0072450, 0.0077502
9: -0.0037211, -0.0029527, -0.0037280, -0.0028993, -0.0006687, 0.0006251

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020277, upper bound: 0.0019239
time: 2.20 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020277, upper bound: 0.0019239
time: 2.18 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0135843, -0.0052947, -0.0132763, -0.0053062, -0.0067243, 0.0064511
1: -0.0067686, -0.0044314, -0.0066817, -0.0044347, -0.0018958, 0.0018188
2: -0.0113803, 0.0058639, -0.0107395, 0.0058399, -0.0139878, 0.0134196
3: 0.0001213, 0.0024033, 0.0002061, 0.0024001, -0.0018511, 0.0017759
4: 0.0017096, 0.0145968, 0.0017275, 0.0141179, -0.0100290, 0.0104536
5: 0.9959812, 0.9995617, 0.9959862, 0.9994286, -0.0027863, 0.0029043
6: 0.0042358, 0.0074858, 0.0042403, 0.0073650, -0.0025292, 0.0026363
7: -0.0075742, 0.0045541, -0.0075574, 0.0041034, -0.0094384, 0.0098380
8: -0.0127373, -0.0032978, -0.0123865, -0.0033110, -0.0076570, 0.0073459
9: -0.0037252, -0.0029108, -0.0037241, -0.0029411, -0.0006338, 0.0006606

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0020276
time: 2.12 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0020277
time: 2.16 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0133545, -0.0048472, -0.0131593, -0.0053352, -0.0066978, 0.0068488
1: -0.0067038, -0.0043053, -0.0066488, -0.0044429, -0.0018884, 0.0019309
2: -0.0109022, 0.0067946, -0.0104961, 0.0057795, -0.0139328, 0.0142469
3: 0.0001846, 0.0025265, 0.0002383, 0.0023921, -0.0018438, 0.0018853
4: 0.0010140, 0.0142395, 0.0017726, 0.0139360, -0.0106472, 0.0104125
5: 0.9957880, 0.9994624, 0.9959987, 0.9993781, -0.0029581, 0.0028929
6: 0.0040604, 0.0073957, 0.0042517, 0.0073191, -0.0026851, 0.0026259
7: -0.0082288, 0.0042178, -0.0075149, 0.0039322, -0.0100202, 0.0097993
8: -0.0124756, -0.0027883, -0.0122533, -0.0033440, -0.0076268, 0.0077988
9: -0.0037692, -0.0029334, -0.0037212, -0.0029526, -0.0006728, 0.0006580

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0020308
time: 1.96 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0020308
time: 2.28 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0135843, -0.0052947, -0.0137021, -0.0052661, -0.0064529, 0.0065366
1: -0.0067686, -0.0044314, -0.0068018, -0.0044234, -0.0018193, 0.0018429
2: -0.0113803, 0.0058639, -0.0116253, 0.0059232, -0.0134234, 0.0135974
3: 0.0001213, 0.0024033, 0.0000889, 0.0024111, -0.0017764, 0.0017994
4: 0.0017096, 0.0145968, 0.0016652, 0.0147799, -0.0101618, 0.0100318
5: 0.9959812, 0.9995617, 0.9959689, 0.9996125, -0.0028233, 0.0027871
6: 0.0042358, 0.0074858, 0.0042246, 0.0075319, -0.0025627, 0.0025299
7: -0.0075742, 0.0045541, -0.0076160, 0.0047264, -0.0095634, 0.0094410
8: -0.0127373, -0.0032978, -0.0128714, -0.0032653, -0.0073480, 0.0074432
9: -0.0037252, -0.0029108, -0.0037280, -0.0028993, -0.0006422, 0.0006339

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019421, upper bound: 0.0020338
time: 2.05 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019421, upper bound: 0.0020338
time: 2.35 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0133545, -0.0048472, -0.0135880, -0.0052939, -0.0064168, 0.0069561
1: -0.0067038, -0.0043053, -0.0067696, -0.0044312, -0.0018091, 0.0019612
2: -0.0109022, 0.0067946, -0.0113879, 0.0058654, -0.0133482, 0.0144700
3: 0.0001846, 0.0025265, 0.0001203, 0.0024035, -0.0017664, 0.0019149
4: 0.0010140, 0.0142395, 0.0017084, 0.0146025, -0.0108140, 0.0099756
5: 0.9957880, 0.9994624, 0.9959809, 0.9995632, -0.0030044, 0.0027715
6: 0.0040604, 0.0073957, 0.0042355, 0.0074872, -0.0027271, 0.0025157
7: -0.0082288, 0.0042178, -0.0075753, 0.0045594, -0.0101772, 0.0093882
8: -0.0124756, -0.0027883, -0.0127415, -0.0032970, -0.0073068, 0.0079209
9: -0.0037692, -0.0029334, -0.0037253, -0.0029105, -0.0006834, 0.0006304

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019114, upper bound: 0.0020097
time: 2.13 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019116, upper bound: 0.0019942
time: 2.24 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.24 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019310, upper bound: 0.0019309
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019310, upper bound: 0.0019310
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0020277, upper bound: 0.0019239
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0020277, upper bound: 0.0019239
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0020276
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0020277
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0020308
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0020308
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019421, upper bound: 0.0020338
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019421, upper bound: 0.0020338
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019114, upper bound: 0.0020097
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019116, upper bound: 0.0019942

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0131583, -0.0053365, -0.0135843, -0.0052947, -0.0063372, 0.0066986
1: -0.0066485, -0.0044432, -0.0067686, -0.0044314, -0.0017867, 0.0018886
2: -0.0104941, 0.0057769, -0.0113803, 0.0058639, -0.0131826, 0.0139344
3: 0.0002386, 0.0023918, 0.0001213, 0.0024033, -0.0017445, 0.0018440
4: 0.0017746, 0.0139345, 0.0017096, 0.0145968, -0.0104137, 0.0098519
5: 0.9959993, 0.9993777, 0.9959812, 0.9995617, -0.0028932, 0.0027371
6: 0.0042522, 0.0073187, 0.0042358, 0.0074858, -0.0026262, 0.0024845
7: -0.0075131, 0.0039308, -0.0075742, 0.0045541, -0.0098004, 0.0092717
8: -0.0122522, -0.0033454, -0.0127373, -0.0032978, -0.0072162, 0.0076277
9: -0.0037211, -0.0029527, -0.0037252, -0.0029108, -0.0006581, 0.0006226

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019998, upper bound: 0.0018796
time: 1.89 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019843, upper bound: 0.0018812
time: 2.06 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0131583, -0.0053365, -0.0133545, -0.0048472, -0.0068396, 0.0065159
1: -0.0066485, -0.0044432, -0.0067038, -0.0043053, -0.0019283, 0.0018371
2: -0.0104941, 0.0057769, -0.0109022, 0.0067946, -0.0142277, 0.0135544
3: 0.0002386, 0.0023918, 0.0001846, 0.0025265, -0.0018828, 0.0017937
4: 0.0017746, 0.0139345, 0.0010140, 0.0142395, -0.0101297, 0.0106329
5: 0.9959993, 0.9993777, 0.9957880, 0.9994624, -0.0028143, 0.0029541
6: 0.0042522, 0.0073187, 0.0040604, 0.0073957, -0.0025546, 0.0026815
7: -0.0075131, 0.0039308, -0.0082288, 0.0042178, -0.0095332, 0.0100067
8: -0.0122522, -0.0033454, -0.0124756, -0.0027883, -0.0077883, 0.0074197
9: -0.0037211, -0.0029527, -0.0037692, -0.0029334, -0.0006401, 0.0006719

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019998, upper bound: 0.0018796
time: 2.10 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019843, upper bound: 0.0018812
time: 2.32 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0135843, -0.0052947, -0.0131583, -0.0053365, -0.0066986, 0.0063372
1: -0.0067686, -0.0044314, -0.0066485, -0.0044432, -0.0018886, 0.0017867
2: -0.0113803, 0.0058639, -0.0104941, 0.0057769, -0.0139344, 0.0131826
3: 0.0001213, 0.0024033, 0.0002386, 0.0023918, -0.0018440, 0.0017445
4: 0.0017096, 0.0145968, 0.0017746, 0.0139345, -0.0098519, 0.0104137
5: 0.9959812, 0.9995617, 0.9959993, 0.9993777, -0.0027371, 0.0028932
6: 0.0042358, 0.0074858, 0.0042522, 0.0073187, -0.0024845, 0.0026262
7: -0.0075742, 0.0045541, -0.0075131, 0.0039308, -0.0092717, 0.0098004
8: -0.0127373, -0.0032978, -0.0122522, -0.0033454, -0.0076277, 0.0072162
9: -0.0037252, -0.0029108, -0.0037211, -0.0029527, -0.0006226, 0.0006581

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018937, upper bound: 0.0019795
time: 2.08 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018840, upper bound: 0.0019844
time: 1.55 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0135843, -0.0052947, -0.0129305, -0.0048721, -0.0072213, 0.0061882
1: -0.0067686, -0.0044314, -0.0065842, -0.0043123, -0.0020360, 0.0017447
2: -0.0113803, 0.0058639, -0.0100201, 0.0067428, -0.0150218, 0.0128727
3: 0.0001213, 0.0024033, 0.0003013, 0.0025196, -0.0019879, 0.0017035
4: 0.0017096, 0.0145968, 0.0010527, 0.0135803, -0.0096203, 0.0112264
5: 0.9959812, 0.9995617, 0.9957988, 0.9992793, -0.0026728, 0.0031190
6: 0.0042358, 0.0074858, 0.0040701, 0.0072294, -0.0024261, 0.0028311
7: -0.0075742, 0.0045541, -0.0081924, 0.0035974, -0.0090538, 0.0105653
8: -0.0127373, -0.0032978, -0.0119927, -0.0028167, -0.0082230, 0.0070466
9: -0.0037252, -0.0029108, -0.0037667, -0.0029751, -0.0006079, 0.0007094

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018937, upper bound: 0.0019795
time: 2.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018840, upper bound: 0.0019845
time: 2.18 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0133545, -0.0048472, -0.0131583, -0.0053365, -0.0065159, 0.0068396
1: -0.0067038, -0.0043053, -0.0066485, -0.0044432, -0.0018371, 0.0019283
2: -0.0109022, 0.0067946, -0.0104941, 0.0057769, -0.0135544, 0.0142277
3: 0.0001846, 0.0025265, 0.0002386, 0.0023918, -0.0017937, 0.0018828
4: 0.0010140, 0.0142395, 0.0017746, 0.0139345, -0.0106329, 0.0101297
5: 0.9957880, 0.9994624, 0.9959993, 0.9993777, -0.0029541, 0.0028143
6: 0.0040604, 0.0073957, 0.0042522, 0.0073187, -0.0026815, 0.0025546
7: -0.0082288, 0.0042178, -0.0075131, 0.0039308, -0.0100067, 0.0095332
8: -0.0124756, -0.0027883, -0.0122522, -0.0033454, -0.0074197, 0.0077883
9: -0.0037692, -0.0029334, -0.0037211, -0.0029527, -0.0006719, 0.0006401

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018894, upper bound: 0.0019827
time: 2.01 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018788, upper bound: 0.0019862
time: 2.16 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0133545, -0.0048472, -0.0129305, -0.0048721, -0.0068250, 0.0064678
1: -0.0067038, -0.0043053, -0.0065842, -0.0043123, -0.0019242, 0.0018235
2: -0.0109022, 0.0067946, -0.0100201, 0.0067428, -0.0141973, 0.0134544
3: 0.0001846, 0.0025265, 0.0003013, 0.0025196, -0.0018788, 0.0017805
4: 0.0010140, 0.0142395, 0.0010527, 0.0135803, -0.0100550, 0.0106102
5: 0.9957880, 0.9994624, 0.9957988, 0.9992793, -0.0027936, 0.0029478
6: 0.0040604, 0.0073957, 0.0040701, 0.0072294, -0.0025357, 0.0026757
7: -0.0082288, 0.0042178, -0.0081924, 0.0035974, -0.0094629, 0.0099854
8: -0.0124756, -0.0027883, -0.0119927, -0.0028167, -0.0077716, 0.0073650
9: -0.0037692, -0.0029334, -0.0037667, -0.0029751, -0.0006354, 0.0006705

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018894, upper bound: 0.0019827
time: 2.18 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018788, upper bound: 0.0019863
time: 1.57 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0135843, -0.0052947, -0.0135843, -0.0052947, -0.0064278, 0.0064278
1: -0.0067686, -0.0044314, -0.0067686, -0.0044314, -0.0018122, 0.0018122
2: -0.0113803, 0.0058639, -0.0113803, 0.0058639, -0.0133711, 0.0133711
3: 0.0001213, 0.0024033, 0.0001213, 0.0024033, -0.0017695, 0.0017695
4: 0.0017096, 0.0145968, 0.0017096, 0.0145968, -0.0099927, 0.0099927
5: 0.9959812, 0.9995617, 0.9959812, 0.9995617, -0.0027763, 0.0027763
6: 0.0042358, 0.0074858, 0.0042358, 0.0074858, -0.0025200, 0.0025200
7: -0.0075742, 0.0045541, -0.0075742, 0.0045541, -0.0094043, 0.0094043
8: -0.0127373, -0.0032978, -0.0127373, -0.0032978, -0.0073194, 0.0073194
9: -0.0037252, -0.0029108, -0.0037252, -0.0029108, -0.0006315, 0.0006315

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019150, upper bound: 0.0019872
time: 2.21 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019051, upper bound: 0.0019914
time: 2.18 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0135843, -0.0052947, -0.0133545, -0.0048472, -0.0069534, 0.0062731
1: -0.0067686, -0.0044314, -0.0067038, -0.0043053, -0.0019604, 0.0017686
2: -0.0113803, 0.0058639, -0.0109022, 0.0067946, -0.0144644, 0.0130494
3: 0.0001213, 0.0024033, 0.0001846, 0.0025265, -0.0019141, 0.0017269
4: 0.0017096, 0.0145968, 0.0010140, 0.0142395, -0.0097523, 0.0108098
5: 0.9959812, 0.9995617, 0.9957880, 0.9994624, -0.0027095, 0.0030033
6: 0.0042358, 0.0074858, 0.0040604, 0.0073957, -0.0024594, 0.0027261
7: -0.0075742, 0.0045541, -0.0082288, 0.0042178, -0.0091780, 0.0101732
8: -0.0127373, -0.0032978, -0.0124756, -0.0027883, -0.0079179, 0.0071433
9: -0.0037252, -0.0029108, -0.0037692, -0.0029334, -0.0006163, 0.0006831

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019150, upper bound: 0.0019872
time: 2.30 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019051, upper bound: 0.0019914
time: 2.29 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0133505, -0.0048943, -0.0136257, -0.0054806, -0.0062011, 0.0068465
1: -0.0067027, -0.0043186, -0.0067802, -0.0044839, -0.0017483, 0.0019303
2: -0.0108940, 0.0066966, -0.0114664, 0.0054770, -0.0128995, 0.0142421
3: 0.0001857, 0.0025135, 0.0001099, 0.0023521, -0.0017070, 0.0018847
4: 0.0010872, 0.0142333, 0.0019987, 0.0146611, -0.0106437, 0.0096402
5: 0.9958084, 0.9994607, 0.9960616, 0.9995796, -0.0029571, 0.0026783
6: 0.0040788, 0.0073941, 0.0043087, 0.0075020, -0.0026842, 0.0024311
7: -0.0081600, 0.0042120, -0.0073022, 0.0046146, -0.0100169, 0.0090725
8: -0.0124711, -0.0028420, -0.0127844, -0.0035096, -0.0070612, 0.0077962
9: -0.0037645, -0.0029338, -0.0037069, -0.0029068, -0.0006726, 0.0006092

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018849, upper bound: 0.0019371
time: 1.79 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018906, upper bound: 0.0019901
time: 2.18 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0133519, -0.0048961, -0.0135783, -0.0054785, -0.0061820, 0.0069015
1: -0.0067031, -0.0043191, -0.0067669, -0.0044832, -0.0017429, 0.0019458
2: -0.0108968, 0.0066929, -0.0113677, 0.0054815, -0.0128597, 0.0143564
3: 0.0001853, 0.0025130, 0.0001230, 0.0023527, -0.0017018, 0.0018998
4: 0.0010900, 0.0142355, 0.0019953, 0.0145874, -0.0107291, 0.0096106
5: 0.9958091, 0.9994613, 0.9960606, 0.9995591, -0.0029809, 0.0026701
6: 0.0040796, 0.0073947, 0.0043079, 0.0074834, -0.0027057, 0.0024236
7: -0.0081573, 0.0042140, -0.0073053, 0.0045452, -0.0100973, 0.0090446
8: -0.0124727, -0.0028440, -0.0127304, -0.0035071, -0.0070394, 0.0078587
9: -0.0037644, -0.0029337, -0.0037072, -0.0029114, -0.0006780, 0.0006073

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018859, upper bound: 0.0019252
time: 2.24 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018913, upper bound: 0.0019744
time: 2.30 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.26 seconds
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0019998, upper bound: 0.0018796
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0019843, upper bound: 0.0018812
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0019998, upper bound: 0.0018796
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0019843, upper bound: 0.0018812
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0018937, upper bound: 0.0019795
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0018840, upper bound: 0.0019844
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0018937, upper bound: 0.0019795
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0018840, upper bound: 0.0019845
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0018894, upper bound: 0.0019827
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0018788, upper bound: 0.0019862
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0018894, upper bound: 0.0019827
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0018788, upper bound: 0.0019863
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0019150, upper bound: 0.0019872
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0019051, upper bound: 0.0019914
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0019150, upper bound: 0.0019872
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0019051, upper bound: 0.0019914
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0018849, upper bound: 0.0019371
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0018906, upper bound: 0.0019901
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0018859, upper bound: 0.0019252
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 5, lower bound: -0.0018913, upper bound: 0.0019744

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0131819, -0.0055218, -0.0135801, -0.0053397, -0.0062173, 0.0064752
1: -0.0066551, -0.0044955, -0.0067674, -0.0044441, -0.0017529, 0.0018256
2: -0.0105432, 0.0053913, -0.0113716, 0.0057701, -0.0129333, 0.0134698
3: 0.0002321, 0.0023407, 0.0001225, 0.0023909, -0.0017115, 0.0017825
4: 0.0020628, 0.0139712, 0.0017797, 0.0145903, -0.0100665, 0.0096655
5: 0.9960793, 0.9993878, 0.9960007, 0.9995598, -0.0027968, 0.0026854
6: 0.0043249, 0.0073280, 0.0042535, 0.0074841, -0.0025386, 0.0024375
7: -0.0072419, 0.0039653, -0.0075083, 0.0045479, -0.0094737, 0.0090964
8: -0.0122791, -0.0035565, -0.0127325, -0.0033492, -0.0070797, 0.0073734
9: -0.0037029, -0.0029504, -0.0037208, -0.0029112, -0.0006361, 0.0006108

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019578, upper bound: 0.0018926
time: 2.49 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020089, upper bound: 0.0018943
time: 2.18 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0131488, -0.0055127, -0.0135816, -0.0053457, -0.0062803, 0.0064907
1: -0.0066458, -0.0044929, -0.0067678, -0.0044458, -0.0017707, 0.0018300
2: -0.0104743, 0.0054104, -0.0113746, 0.0057578, -0.0130643, 0.0135019
3: 0.0002412, 0.0023433, 0.0001221, 0.0023892, -0.0017289, 0.0017868
4: 0.0020485, 0.0139197, 0.0017888, 0.0145925, -0.0100905, 0.0097634
5: 0.9960754, 0.9993736, 0.9960032, 0.9995605, -0.0028034, 0.0027126
6: 0.0043213, 0.0073150, 0.0042558, 0.0074847, -0.0025447, 0.0024622
7: -0.0072553, 0.0039169, -0.0074996, 0.0045500, -0.0094963, 0.0091885
8: -0.0122414, -0.0035461, -0.0127342, -0.0033559, -0.0071514, 0.0073910
9: -0.0037038, -0.0029536, -0.0037202, -0.0029111, -0.0006377, 0.0006170

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019471, upper bound: 0.0018948
time: 2.47 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019928, upper bound: 0.0018962
time: 2.17 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0131819, -0.0055218, -0.0133505, -0.0048943, -0.0067186, 0.0062930
1: -0.0066551, -0.0044955, -0.0067027, -0.0043186, -0.0018942, 0.0017742
2: -0.0105432, 0.0053913, -0.0108940, 0.0066966, -0.0139761, 0.0130907
3: 0.0002321, 0.0023407, 0.0001857, 0.0025135, -0.0018495, 0.0017323
4: 0.0020628, 0.0139712, 0.0010872, 0.0142333, -0.0097832, 0.0104448
5: 0.9960793, 0.9993878, 0.9958084, 0.9994607, -0.0027181, 0.0029019
6: 0.0043249, 0.0073280, 0.0040788, 0.0073941, -0.0024672, 0.0026340
7: -0.0072419, 0.0039653, -0.0081600, 0.0042120, -0.0092070, 0.0098298
8: -0.0122791, -0.0035565, -0.0124711, -0.0028420, -0.0076505, 0.0071659
9: -0.0037029, -0.0029504, -0.0037645, -0.0029338, -0.0006182, 0.0006600

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019279, upper bound: 0.0018540
time: 2.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019819, upper bound: 0.0018592
time: 2.41 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0131488, -0.0055127, -0.0133519, -0.0048961, -0.0067825, 0.0063084
1: -0.0066458, -0.0044929, -0.0067031, -0.0043191, -0.0019122, 0.0017786
2: -0.0104743, 0.0054104, -0.0108968, 0.0066929, -0.0141090, 0.0131228
3: 0.0002412, 0.0023433, 0.0001853, 0.0025130, -0.0018671, 0.0017366
4: 0.0020485, 0.0139197, 0.0010900, 0.0142355, -0.0098072, 0.0105442
5: 0.9960754, 0.9993736, 0.9958091, 0.9994613, -0.0027247, 0.0029295
6: 0.0043213, 0.0073150, 0.0040796, 0.0073947, -0.0024732, 0.0026591
7: -0.0072553, 0.0039169, -0.0081573, 0.0042140, -0.0092296, 0.0099232
8: -0.0122414, -0.0035461, -0.0124727, -0.0028440, -0.0077233, 0.0071834
9: -0.0037038, -0.0029536, -0.0037644, -0.0029337, -0.0006198, 0.0006663

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019157, upper bound: 0.0018564
time: 2.30 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019659, upper bound: 0.0018614
time: 2.62 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0136314, -0.0054815, -0.0131544, -0.0053831, -0.0066162, 0.0061227
1: -0.0067819, -0.0044841, -0.0066474, -0.0044564, -0.0018654, 0.0017262
2: -0.0114782, 0.0054752, -0.0104861, 0.0056799, -0.0137631, 0.0127365
3: 0.0001083, 0.0023519, 0.0002396, 0.0023789, -0.0018213, 0.0016855
4: 0.0020000, 0.0146699, 0.0018470, 0.0139285, -0.0095185, 0.0102857
5: 0.9960619, 0.9995819, 0.9960194, 0.9993760, -0.0026445, 0.0028577
6: 0.0043090, 0.0075042, 0.0042705, 0.0073172, -0.0024004, 0.0025939
7: -0.0073009, 0.0046229, -0.0074449, 0.0039251, -0.0089579, 0.0096800
8: -0.0127909, -0.0035106, -0.0122478, -0.0033985, -0.0075339, 0.0069720
9: -0.0037069, -0.0029062, -0.0037165, -0.0029531, -0.0006015, 0.0006500

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018635, upper bound: 0.0019858
time: 1.53 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019060, upper bound: 0.0019866
time: 2.07 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0135744, -0.0054791, -0.0131557, -0.0053854, -0.0066411, 0.0061037
1: -0.0067658, -0.0044834, -0.0066477, -0.0044570, -0.0018724, 0.0017209
2: -0.0113598, 0.0054803, -0.0104886, 0.0056751, -0.0138149, 0.0126970
3: 0.0001240, 0.0023525, 0.0002393, 0.0023783, -0.0018282, 0.0016802
4: 0.0019963, 0.0145814, 0.0018506, 0.0139304, -0.0094890, 0.0103244
5: 0.9960608, 0.9995574, 0.9960204, 0.9993765, -0.0026363, 0.0028684
6: 0.0043081, 0.0074819, 0.0042714, 0.0073177, -0.0023930, 0.0026037
7: -0.0073044, 0.0045396, -0.0074415, 0.0039269, -0.0089302, 0.0097164
8: -0.0127261, -0.0035078, -0.0122492, -0.0034011, -0.0075623, 0.0069504
9: -0.0037071, -0.0029118, -0.0037163, -0.0029529, -0.0005996, 0.0006524

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018543, upper bound: 0.0019923
time: 2.06 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018962, upper bound: 0.0019928
time: 2.16 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0136314, -0.0054815, -0.0129268, -0.0049201, -0.0071377, 0.0059738
1: -0.0067819, -0.0044841, -0.0065832, -0.0043258, -0.0020124, 0.0016842
2: -0.0114782, 0.0054752, -0.0100126, 0.0066429, -0.0148479, 0.0124267
3: 0.0001083, 0.0023519, 0.0003023, 0.0025064, -0.0019649, 0.0016445
4: 0.0020000, 0.0146699, 0.0011274, 0.0135747, -0.0092869, 0.0110964
5: 0.9960619, 0.9995819, 0.9958194, 0.9992777, -0.0025802, 0.0030829
6: 0.0043090, 0.0075042, 0.0040890, 0.0072280, -0.0023420, 0.0027983
7: -0.0073009, 0.0046229, -0.0081222, 0.0035921, -0.0087400, 0.0104429
8: -0.0127909, -0.0035106, -0.0119886, -0.0028714, -0.0081278, 0.0068024
9: -0.0037069, -0.0029062, -0.0037620, -0.0029754, -0.0005869, 0.0007012

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018264, upper bound: 0.0019563
time: 1.57 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018764, upper bound: 0.0019588
time: 2.33 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0135744, -0.0054791, -0.0129281, -0.0049201, -0.0071637, 0.0059602
1: -0.0067658, -0.0044834, -0.0065836, -0.0043258, -0.0020197, 0.0016804
2: -0.0113598, 0.0054803, -0.0100151, 0.0066430, -0.0149019, 0.0123983
3: 0.0001240, 0.0023525, 0.0003020, 0.0025064, -0.0019720, 0.0016407
4: 0.0019963, 0.0145814, 0.0011273, 0.0135766, -0.0092657, 0.0111368
5: 0.9960608, 0.9995574, 0.9958195, 0.9992782, -0.0025743, 0.0030941
6: 0.0043081, 0.0074819, 0.0040890, 0.0072285, -0.0023367, 0.0028085
7: -0.0073044, 0.0045396, -0.0081222, 0.0035939, -0.0087201, 0.0104809
8: -0.0127261, -0.0035078, -0.0119900, -0.0028714, -0.0081573, 0.0067869
9: -0.0037071, -0.0029118, -0.0037620, -0.0029753, -0.0005855, 0.0007038

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018167, upper bound: 0.0019625
time: 2.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018654, upper bound: 0.0019649
time: 2.06 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0133904, -0.0050448, -0.0131544, -0.0053831, -0.0064327, 0.0066136
1: -0.0067139, -0.0043610, -0.0066474, -0.0044564, -0.0018136, 0.0018646
2: -0.0109770, 0.0063836, -0.0104861, 0.0056799, -0.0133813, 0.0137577
3: 0.0001747, 0.0024721, 0.0002396, 0.0023789, -0.0017708, 0.0018206
4: 0.0013212, 0.0142954, 0.0018470, 0.0139285, -0.0102816, 0.0100004
5: 0.9958733, 0.9994779, 0.9960194, 0.9993760, -0.0028565, 0.0027784
6: 0.0041379, 0.0074098, 0.0042705, 0.0073172, -0.0025929, 0.0025219
7: -0.0079398, 0.0042704, -0.0074449, 0.0039251, -0.0096761, 0.0094115
8: -0.0125165, -0.0030133, -0.0122478, -0.0033985, -0.0073250, 0.0075310
9: -0.0037498, -0.0029299, -0.0037165, -0.0029531, -0.0006497, 0.0006320

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018190, upper bound: 0.0019568
time: 1.57 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018719, upper bound: 0.0019616
time: 2.26 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0133452, -0.0050254, -0.0131557, -0.0053854, -0.0064592, 0.0066000
1: -0.0067012, -0.0043555, -0.0066477, -0.0044570, -0.0018211, 0.0018608
2: -0.0108829, 0.0064241, -0.0104886, 0.0056751, -0.0134365, 0.0137293
3: 0.0001871, 0.0024774, 0.0002393, 0.0023783, -0.0017781, 0.0018169
4: 0.0012909, 0.0142250, 0.0018506, 0.0139304, -0.0102604, 0.0100416
5: 0.9958649, 0.9994584, 0.9960204, 0.9993765, -0.0028507, 0.0027899
6: 0.0041302, 0.0073920, 0.0042714, 0.0073177, -0.0025875, 0.0025323
7: -0.0079683, 0.0042042, -0.0074415, 0.0039269, -0.0096562, 0.0094503
8: -0.0124650, -0.0029912, -0.0122492, -0.0034011, -0.0073552, 0.0075154
9: -0.0037517, -0.0029343, -0.0037163, -0.0029529, -0.0006484, 0.0006346

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018101, upper bound: 0.0019615
time: 1.98 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018615, upper bound: 0.0019659
time: 1.98 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0133904, -0.0050448, -0.0129268, -0.0049201, -0.0067398, 0.0062459
1: -0.0067139, -0.0043610, -0.0065832, -0.0043258, -0.0019002, 0.0017609
2: -0.0109770, 0.0063836, -0.0100126, 0.0066429, -0.0140201, 0.0129927
3: 0.0001747, 0.0024721, 0.0003023, 0.0025064, -0.0018553, 0.0017194
4: 0.0013212, 0.0142954, 0.0011274, 0.0135747, -0.0097099, 0.0104777
5: 0.9958733, 0.9994779, 0.9958194, 0.9992777, -0.0026977, 0.0029110
6: 0.0041379, 0.0074098, 0.0040890, 0.0072280, -0.0024487, 0.0026423
7: -0.0079398, 0.0042704, -0.0081222, 0.0035921, -0.0091381, 0.0098607
8: -0.0125165, -0.0030133, -0.0119886, -0.0028714, -0.0076746, 0.0071122
9: -0.0037498, -0.0029299, -0.0037620, -0.0029754, -0.0006136, 0.0006621

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018171, upper bound: 0.0019566
time: 2.00 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018694, upper bound: 0.0019616
time: 2.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0133452, -0.0050254, -0.0129281, -0.0049201, -0.0067687, 0.0062322
1: -0.0067012, -0.0043555, -0.0065836, -0.0043258, -0.0019083, 0.0017571
2: -0.0108829, 0.0064241, -0.0100151, 0.0066430, -0.0140803, 0.0129642
3: 0.0001871, 0.0024774, 0.0003020, 0.0025064, -0.0018633, 0.0017156
4: 0.0012909, 0.0142250, 0.0011273, 0.0135766, -0.0096886, 0.0105227
5: 0.9958649, 0.9994584, 0.9958195, 0.9992782, -0.0026918, 0.0029235
6: 0.0041302, 0.0073920, 0.0040890, 0.0072285, -0.0024433, 0.0026537
7: -0.0079683, 0.0042042, -0.0081222, 0.0035939, -0.0091181, 0.0099030
8: -0.0124650, -0.0029912, -0.0119900, -0.0028714, -0.0077075, 0.0070966
9: -0.0037517, -0.0029343, -0.0037620, -0.0029753, -0.0006123, 0.0006650

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018083, upper bound: 0.0019614
time: 1.65 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018589, upper bound: 0.0019659
time: 2.19 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0136314, -0.0054815, -0.0135801, -0.0053397, -0.0063151, 0.0062119
1: -0.0067819, -0.0044841, -0.0067674, -0.0044441, -0.0017805, 0.0017514
2: -0.0114782, 0.0054752, -0.0113716, 0.0057701, -0.0131368, 0.0129221
3: 0.0001083, 0.0023519, 0.0001225, 0.0023909, -0.0017384, 0.0017100
4: 0.0020000, 0.0146699, 0.0017797, 0.0145903, -0.0096572, 0.0098176
5: 0.9960619, 0.9995819, 0.9960007, 0.9995598, -0.0026830, 0.0027276
6: 0.0043090, 0.0075042, 0.0042535, 0.0074841, -0.0024354, 0.0024759
7: -0.0073009, 0.0046229, -0.0075083, 0.0045479, -0.0090885, 0.0092395
8: -0.0127909, -0.0035106, -0.0127325, -0.0033492, -0.0071911, 0.0070736
9: -0.0037069, -0.0029062, -0.0037208, -0.0029112, -0.0006103, 0.0006204

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018763, upper bound: 0.0019894
time: 2.30 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019160, upper bound: 0.0019902
time: 2.37 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0135744, -0.0054791, -0.0135816, -0.0053457, -0.0063723, 0.0061932
1: -0.0067658, -0.0044834, -0.0067678, -0.0044458, -0.0017966, 0.0017461
2: -0.0113598, 0.0054803, -0.0113746, 0.0057578, -0.0132557, 0.0128832
3: 0.0001240, 0.0023525, 0.0001221, 0.0023892, -0.0017542, 0.0017049
4: 0.0019963, 0.0145814, 0.0017888, 0.0145925, -0.0096281, 0.0099065
5: 0.9960608, 0.9995574, 0.9960032, 0.9995605, -0.0026750, 0.0027523
6: 0.0043081, 0.0074819, 0.0042558, 0.0074847, -0.0024281, 0.0024983
7: -0.0073044, 0.0045396, -0.0074996, 0.0045500, -0.0090611, 0.0093231
8: -0.0127261, -0.0035078, -0.0127342, -0.0033559, -0.0072562, 0.0070523
9: -0.0037071, -0.0029118, -0.0037202, -0.0029111, -0.0006084, 0.0006260

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018674, upper bound: 0.0019958
time: 2.14 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019070, upper bound: 0.0019963
time: 2.15 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0136314, -0.0054815, -0.0133505, -0.0048943, -0.0068384, 0.0060574
1: -0.0067819, -0.0044841, -0.0067027, -0.0043186, -0.0019280, 0.0017078
2: -0.0114782, 0.0054752, -0.0108940, 0.0066966, -0.0142253, 0.0126007
3: 0.0001083, 0.0023519, 0.0001857, 0.0025135, -0.0018825, 0.0016675
4: 0.0020000, 0.0146699, 0.0010872, 0.0142333, -0.0094170, 0.0106311
5: 0.9960619, 0.9995819, 0.9958084, 0.9994607, -0.0026163, 0.0029536
6: 0.0043090, 0.0075042, 0.0040788, 0.0073941, -0.0023748, 0.0026810
7: -0.0073009, 0.0046229, -0.0081600, 0.0042120, -0.0088624, 0.0100051
8: -0.0127909, -0.0035106, -0.0124711, -0.0028420, -0.0077870, 0.0068976
9: -0.0037069, -0.0029062, -0.0037645, -0.0029338, -0.0005951, 0.0006718

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018562, upper bound: 0.0019637
time: 1.87 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019018, upper bound: 0.0019668
time: 2.28 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0135744, -0.0054791, -0.0133519, -0.0048961, -0.0068984, 0.0060462
1: -0.0067658, -0.0044834, -0.0067031, -0.0043191, -0.0019449, 0.0017046
2: -0.0113598, 0.0054803, -0.0108968, 0.0066929, -0.0143501, 0.0125773
3: 0.0001240, 0.0023525, 0.0001853, 0.0025130, -0.0018990, 0.0016644
4: 0.0019963, 0.0145814, 0.0010900, 0.0142355, -0.0093995, 0.0107243
5: 0.9960608, 0.9995574, 0.9958091, 0.9994613, -0.0026114, 0.0029795
6: 0.0043081, 0.0074819, 0.0040796, 0.0073947, -0.0023704, 0.0027045
7: -0.0073044, 0.0045396, -0.0081573, 0.0042140, -0.0088459, 0.0100928
8: -0.0127261, -0.0035078, -0.0124727, -0.0028440, -0.0078552, 0.0068848
9: -0.0037071, -0.0029118, -0.0037644, -0.0029337, -0.0005940, 0.0006777

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018465, upper bound: 0.0019695
time: 2.52 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018913, upper bound: 0.0019722
time: 2.20 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0130818, -0.0049416, -0.0135269, -0.0055052, -0.0059056, 0.0067128
1: -0.0066269, -0.0043319, -0.0067524, -0.0044908, -0.0016650, 0.0018926
2: -0.0103350, 0.0065984, -0.0112608, 0.0054259, -0.0122848, 0.0139639
3: 0.0002596, 0.0025005, 0.0001371, 0.0023453, -0.0016257, 0.0018479
4: 0.0011606, 0.0138156, 0.0020368, 0.0145075, -0.0104358, 0.0091809
5: 0.9958287, 0.9993447, 0.9960722, 0.9995369, -0.0028994, 0.0025507
6: 0.0040974, 0.0072888, 0.0043183, 0.0074632, -0.0026317, 0.0023153
7: -0.0080908, 0.0038189, -0.0072662, 0.0044700, -0.0098212, 0.0086402
8: -0.0121651, -0.0028958, -0.0126719, -0.0035375, -0.0067247, 0.0076439
9: -0.0037599, -0.0029602, -0.0037045, -0.0029165, -0.0006595, 0.0005802

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018782, upper bound: 0.0019371
time: 2.28 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018782, upper bound: 0.0019371
time: 2.39 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0132850, -0.0049230, -0.0136202, -0.0054828, -0.0060215, 0.0068072
1: -0.0066842, -0.0043266, -0.0067787, -0.0044845, -0.0016977, 0.0019192
2: -0.0107577, 0.0066369, -0.0114550, 0.0054725, -0.0125260, 0.0141603
3: 0.0002037, 0.0025056, 0.0001114, 0.0023515, -0.0016576, 0.0018739
4: 0.0011318, 0.0141315, 0.0020021, 0.0146526, -0.0105825, 0.0093611
5: 0.9958207, 0.9994324, 0.9960625, 0.9995772, -0.0029401, 0.0026008
6: 0.0040901, 0.0073684, 0.0043096, 0.0074998, -0.0026688, 0.0023607
7: -0.0081180, 0.0041162, -0.0072990, 0.0046066, -0.0099593, 0.0088099
8: -0.0123965, -0.0028746, -0.0127782, -0.0035121, -0.0068567, 0.0077514
9: -0.0037617, -0.0029402, -0.0037067, -0.0029073, -0.0006688, 0.0005916

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018826, upper bound: 0.0019900
time: 2.56 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018826, upper bound: 0.0019900
time: 2.26 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0132864, -0.0049250, -0.0135724, -0.0054810, -0.0059792, 0.0068608
1: -0.0066846, -0.0043272, -0.0067652, -0.0044840, -0.0016858, 0.0019343
2: -0.0107605, 0.0066328, -0.0113556, 0.0054762, -0.0124379, 0.0142719
3: 0.0002033, 0.0025050, 0.0001246, 0.0023520, -0.0016460, 0.0018887
4: 0.0011349, 0.0141336, 0.0019993, 0.0145783, -0.0106659, 0.0092953
5: 0.9958215, 0.9994330, 0.9960617, 0.9995565, -0.0029633, 0.0025825
6: 0.0040909, 0.0073690, 0.0043089, 0.0074811, -0.0026898, 0.0023441
7: -0.0081151, 0.0041182, -0.0073016, 0.0045367, -0.0100378, 0.0087479
8: -0.0123981, -0.0028769, -0.0127238, -0.0035100, -0.0068085, 0.0078125
9: -0.0037615, -0.0029401, -0.0037069, -0.0029120, -0.0006740, 0.0005874

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018836, upper bound: 0.0019744
time: 2.31 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018836, upper bound: 0.0019745
time: 2.28 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.35 seconds
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0019578, upper bound: 0.0018926
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0020089, upper bound: 0.0018943
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0019471, upper bound: 0.0018948
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0019928, upper bound: 0.0018962
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0019279, upper bound: 0.0018540
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0019819, upper bound: 0.0018592
NS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0019157, upper bound: 0.0018564
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0019659, upper bound: 0.0018614
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018635, upper bound: 0.0019858
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0019060, upper bound: 0.0019866
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018543, upper bound: 0.0019923
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018962, upper bound: 0.0019928
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018264, upper bound: 0.0019563
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018764, upper bound: 0.0019588
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018167, upper bound: 0.0019625
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018654, upper bound: 0.0019649
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018190, upper bound: 0.0019568
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018719, upper bound: 0.0019616
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018101, upper bound: 0.0019615
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018615, upper bound: 0.0019659
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018171, upper bound: 0.0019566
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018694, upper bound: 0.0019616
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018083, upper bound: 0.0019614
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018589, upper bound: 0.0019659
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018763, upper bound: 0.0019894
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0019160, upper bound: 0.0019902
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018674, upper bound: 0.0019958
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0019070, upper bound: 0.0019963
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018562, upper bound: 0.0019637
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0019018, upper bound: 0.0019668
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018465, upper bound: 0.0019695
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018913, upper bound: 0.0019722
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018782, upper bound: 0.0019371
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018782, upper bound: 0.0019371
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018826, upper bound: 0.0019900
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018826, upper bound: 0.0019900
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018836, upper bound: 0.0019744
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.35
Output dim: 5, lower bound: -0.0018836, upper bound: 0.0019745

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0130810, -0.0055478, -0.0132955, -0.0053927, -0.0060727, 0.0061725
1: -0.0066267, -0.0045028, -0.0066871, -0.0044591, -0.0017121, 0.0017402
2: -0.0103332, 0.0053372, -0.0107794, 0.0056600, -0.0126325, 0.0128400
3: 0.0002599, 0.0023336, 0.0002008, 0.0023763, -0.0016717, 0.0016992
4: 0.0021031, 0.0138143, 0.0018619, 0.0141478, -0.0095958, 0.0094407
5: 0.9960906, 0.9993442, 0.9960235, 0.9994370, -0.0026660, 0.0026229
6: 0.0043351, 0.0072884, 0.0042742, 0.0073725, -0.0024199, 0.0023808
7: -0.0072038, 0.0038176, -0.0074308, 0.0041315, -0.0090307, 0.0088848
8: -0.0121641, -0.0035861, -0.0124084, -0.0034094, -0.0069150, 0.0070286
9: -0.0037003, -0.0029603, -0.0037156, -0.0029392, -0.0006064, 0.0005966

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019280, upper bound: 0.0018530
time: 2.17 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0018530
time: 2.47 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0131764, -0.0055241, -0.0135172, -0.0053679, -0.0061772, 0.0062991
1: -0.0066536, -0.0044961, -0.0067497, -0.0044521, -0.0017416, 0.0017760
2: -0.0105318, 0.0053865, -0.0112408, 0.0057116, -0.0128499, 0.0131035
3: 0.0002336, 0.0023401, 0.0001398, 0.0023831, -0.0017005, 0.0017340
4: 0.0020663, 0.0139627, 0.0018234, 0.0144925, -0.0097927, 0.0096032
5: 0.9960803, 0.9993855, 0.9960127, 0.9995326, -0.0027207, 0.0026681
6: 0.0043258, 0.0073259, 0.0042645, 0.0074595, -0.0024696, 0.0024218
7: -0.0072385, 0.0039573, -0.0074672, 0.0044559, -0.0092160, 0.0090377
8: -0.0122728, -0.0035591, -0.0126609, -0.0033812, -0.0070341, 0.0071729
9: -0.0037027, -0.0029509, -0.0037180, -0.0029174, -0.0006188, 0.0006069

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020085, upper bound: 0.0018540
time: 1.74 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020085, upper bound: 0.0018943
time: 2.40 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0130500, -0.0055389, -0.0132972, -0.0053956, -0.0061488, 0.0061753
1: -0.0066179, -0.0045003, -0.0066876, -0.0044599, -0.0017336, 0.0017410
2: -0.0102688, 0.0053559, -0.0107831, 0.0056538, -0.0127907, 0.0128458
3: 0.0002684, 0.0023361, 0.0002003, 0.0023755, -0.0016926, 0.0016999
4: 0.0020892, 0.0137662, 0.0018666, 0.0141505, -0.0096001, 0.0095590
5: 0.9960867, 0.9993309, 0.9960248, 0.9994376, -0.0026672, 0.0026558
6: 0.0043315, 0.0072763, 0.0042754, 0.0073732, -0.0024210, 0.0024106
7: -0.0072169, 0.0037723, -0.0074265, 0.0041340, -0.0090348, 0.0089960
8: -0.0121289, -0.0035759, -0.0124104, -0.0034128, -0.0070016, 0.0070318
9: -0.0037012, -0.0029633, -0.0037153, -0.0029390, -0.0006067, 0.0006041

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018863, upper bound: 0.0018201
time: 1.79 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017792, upper bound: 0.0018201
time: 2.47 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0131431, -0.0055150, -0.0135187, -0.0053739, -0.0062399, 0.0062933
1: -0.0066442, -0.0044935, -0.0067501, -0.0044537, -0.0017593, 0.0017743
2: -0.0104624, 0.0054055, -0.0112437, 0.0056991, -0.0129802, 0.0130913
3: 0.0002428, 0.0023426, 0.0001394, 0.0023815, -0.0017177, 0.0017324
4: 0.0020521, 0.0139108, 0.0018327, 0.0144947, -0.0097836, 0.0097006
5: 0.9960763, 0.9993711, 0.9960154, 0.9995333, -0.0027182, 0.0026951
6: 0.0043222, 0.0073128, 0.0042668, 0.0074600, -0.0024673, 0.0024464
7: -0.0072519, 0.0039085, -0.0074584, 0.0044580, -0.0092075, 0.0091294
8: -0.0122349, -0.0035487, -0.0126625, -0.0033880, -0.0071054, 0.0071662
9: -0.0037036, -0.0029542, -0.0037174, -0.0029173, -0.0006183, 0.0006130

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019924, upper bound: 0.0018543
time: 1.52 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019924, upper bound: 0.0018961
time: 2.22 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0131764, -0.0055241, -0.0132850, -0.0049230, -0.0066783, 0.0061371
1: -0.0066536, -0.0044961, -0.0066842, -0.0043266, -0.0018829, 0.0017303
2: -0.0105318, 0.0053865, -0.0107577, 0.0066369, -0.0138923, 0.0127664
3: 0.0002336, 0.0023401, 0.0002037, 0.0025056, -0.0018384, 0.0016894
4: 0.0020663, 0.0139627, 0.0011318, 0.0141315, -0.0095408, 0.0103822
5: 0.9960803, 0.9993855, 0.9958207, 0.9994324, -0.0026507, 0.0028845
6: 0.0043258, 0.0073259, 0.0040901, 0.0073684, -0.0024061, 0.0026182
7: -0.0072385, 0.0039573, -0.0081180, 0.0041162, -0.0089790, 0.0097708
8: -0.0122728, -0.0035591, -0.0123965, -0.0028746, -0.0076046, 0.0069884
9: -0.0037027, -0.0029509, -0.0037617, -0.0029402, -0.0006029, 0.0006561

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019790, upper bound: 0.0018101
time: 2.26 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019790, upper bound: 0.0018592
time: 2.38 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0131431, -0.0055150, -0.0132864, -0.0049250, -0.0067414, 0.0061293
1: -0.0066442, -0.0044935, -0.0066846, -0.0043272, -0.0019007, 0.0017281
2: -0.0104624, 0.0054055, -0.0107605, 0.0066328, -0.0140236, 0.0127502
3: 0.0002428, 0.0023426, 0.0002033, 0.0025050, -0.0018558, 0.0016873
4: 0.0020521, 0.0139108, 0.0011349, 0.0141336, -0.0095287, 0.0104804
5: 0.9960763, 0.9993711, 0.9958215, 0.9994330, -0.0026474, 0.0029118
6: 0.0043222, 0.0073128, 0.0040909, 0.0073690, -0.0024030, 0.0026430
7: -0.0072519, 0.0039085, -0.0081151, 0.0041182, -0.0089676, 0.0098632
8: -0.0122349, -0.0035487, -0.0123981, -0.0028769, -0.0076765, 0.0069795
9: -0.0037036, -0.0029542, -0.0037615, -0.0029401, -0.0006022, 0.0006623

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019615, upper bound: 0.0018101
time: 2.30 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019615, upper bound: 0.0018615
time: 2.05 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0135319, -0.0055061, -0.0128650, -0.0054345, -0.0064700, 0.0058253
1: -0.0067538, -0.0044910, -0.0065658, -0.0044709, -0.0018241, 0.0016424
2: -0.0112713, 0.0054240, -0.0098840, 0.0055729, -0.0134589, 0.0121179
3: 0.0001357, 0.0023451, 0.0003193, 0.0023648, -0.0017811, 0.0016036
4: 0.0020383, 0.0145153, 0.0019270, 0.0134785, -0.0090561, 0.0100584
5: 0.9960725, 0.9995390, 0.9960416, 0.9992509, -0.0025161, 0.0027945
6: 0.0043187, 0.0074652, 0.0042906, 0.0072038, -0.0022838, 0.0025366
7: -0.0072649, 0.0044774, -0.0073696, 0.0035017, -0.0085228, 0.0094660
8: -0.0126776, -0.0035386, -0.0119182, -0.0034571, -0.0073674, 0.0066333
9: -0.0037044, -0.0029160, -0.0037115, -0.0029815, -0.0005723, 0.0006356

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018196, upper bound: 0.0019495
time: 2.00 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018211, upper bound: 0.0019495
time: 1.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0136261, -0.0054837, -0.0130904, -0.0054089, -0.0065798, 0.0059576
1: -0.0067804, -0.0044847, -0.0066293, -0.0044636, -0.0018551, 0.0016797
2: -0.0114672, 0.0054707, -0.0103529, 0.0056262, -0.0136872, 0.0123931
3: 0.0001098, 0.0023513, 0.0002573, 0.0023718, -0.0018113, 0.0016400
4: 0.0020034, 0.0146617, 0.0018872, 0.0138290, -0.0092618, 0.0102290
5: 0.9960628, 0.9995797, 0.9960306, 0.9993483, -0.0025732, 0.0028419
6: 0.0043099, 0.0075021, 0.0042806, 0.0072921, -0.0023357, 0.0025796
7: -0.0072977, 0.0046152, -0.0074071, 0.0038315, -0.0087164, 0.0096266
8: -0.0127849, -0.0035130, -0.0121749, -0.0034279, -0.0074924, 0.0067840
9: -0.0037066, -0.0029067, -0.0037140, -0.0029593, -0.0005853, 0.0006464

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019051, upper bound: 0.0019457
time: 2.36 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019051, upper bound: 0.0019866
time: 2.18 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0134780, -0.0055065, -0.0128665, -0.0054356, -0.0065066, 0.0057819
1: -0.0067386, -0.0044912, -0.0065662, -0.0044712, -0.0018345, 0.0016301
2: -0.0111590, 0.0054231, -0.0098871, 0.0055706, -0.0135350, 0.0120276
3: 0.0001506, 0.0023450, 0.0003189, 0.0023645, -0.0017911, 0.0015917
4: 0.0020390, 0.0144314, 0.0019287, 0.0134809, -0.0089887, 0.0101152
5: 0.9960727, 0.9995157, 0.9960421, 0.9992516, -0.0024973, 0.0028103
6: 0.0043189, 0.0074441, 0.0042911, 0.0072043, -0.0022668, 0.0025509
7: -0.0072643, 0.0043985, -0.0073680, 0.0035039, -0.0084593, 0.0095196
8: -0.0126162, -0.0035391, -0.0119199, -0.0034583, -0.0074091, 0.0065839
9: -0.0037044, -0.0029213, -0.0037114, -0.0029813, -0.0005680, 0.0006392

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018114, upper bound: 0.0019570
time: 1.62 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018107, upper bound: 0.0019570
time: 2.34 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0135688, -0.0054816, -0.0130917, -0.0054114, -0.0066038, 0.0059111
1: -0.0067642, -0.0044841, -0.0066297, -0.0044643, -0.0018619, 0.0016666
2: -0.0113480, 0.0054749, -0.0103555, 0.0056211, -0.0137372, 0.0122963
3: 0.0001256, 0.0023518, 0.0002569, 0.0023712, -0.0018179, 0.0016272
4: 0.0020003, 0.0145727, 0.0018910, 0.0138309, -0.0091895, 0.0102664
5: 0.9960620, 0.9995549, 0.9960316, 0.9993489, -0.0025531, 0.0028523
6: 0.0043091, 0.0074797, 0.0042816, 0.0072926, -0.0023174, 0.0025890
7: -0.0073007, 0.0045314, -0.0074035, 0.0038333, -0.0086483, 0.0096618
8: -0.0127196, -0.0035107, -0.0121763, -0.0034307, -0.0075198, 0.0067310
9: -0.0037068, -0.0029123, -0.0037137, -0.0029592, -0.0005807, 0.0006488

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018948, upper bound: 0.0019471
time: 2.38 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018948, upper bound: 0.0019928
time: 2.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0135319, -0.0055061, -0.0126555, -0.0049600, -0.0070181, 0.0056806
1: -0.0067538, -0.0044910, -0.0065067, -0.0043371, -0.0019787, 0.0016016
2: -0.0112713, 0.0054240, -0.0094482, 0.0065600, -0.0145992, 0.0118167
3: 0.0001357, 0.0023451, 0.0003770, 0.0024954, -0.0019320, 0.0015638
4: 0.0020383, 0.0145153, 0.0011894, 0.0131529, -0.0088311, 0.0109105
5: 0.9960725, 0.9995390, 0.9958367, 0.9991604, -0.0024535, 0.0030313
6: 0.0043187, 0.0074652, 0.0041046, 0.0071216, -0.0022271, 0.0027515
7: -0.0072649, 0.0044774, -0.0080638, 0.0031952, -0.0083110, 0.0102680
8: -0.0126776, -0.0035386, -0.0116797, -0.0029168, -0.0079916, 0.0064685
9: -0.0037044, -0.0029160, -0.0037581, -0.0030021, -0.0005581, 0.0006895

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017811, upper bound: 0.0019185
time: 1.59 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017795, upper bound: 0.0019185
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0136261, -0.0054837, -0.0128604, -0.0049462, -0.0071015, 0.0058146
1: -0.0067804, -0.0044847, -0.0065645, -0.0043332, -0.0020022, 0.0016393
2: -0.0114672, 0.0054707, -0.0098744, 0.0065887, -0.0147725, 0.0120955
3: 0.0001098, 0.0023513, 0.0003206, 0.0024992, -0.0019549, 0.0016006
4: 0.0020034, 0.0146617, 0.0011679, 0.0134713, -0.0090394, 0.0110401
5: 0.9960628, 0.9995797, 0.9958307, 0.9992489, -0.0025114, 0.0030673
6: 0.0043099, 0.0075021, 0.0040992, 0.0072019, -0.0022796, 0.0027841
7: -0.0072977, 0.0046152, -0.0080841, 0.0034949, -0.0085071, 0.0103899
8: -0.0127849, -0.0035130, -0.0119129, -0.0029010, -0.0080865, 0.0066211
9: -0.0037066, -0.0029067, -0.0037594, -0.0029819, -0.0005712, 0.0006977

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018702, upper bound: 0.0019146
time: 2.39 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018719, upper bound: 0.0019588
time: 2.40 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0134780, -0.0055065, -0.0126569, -0.0049589, -0.0070510, 0.0056564
1: -0.0067386, -0.0044912, -0.0065071, -0.0043368, -0.0019879, 0.0015948
2: -0.0111590, 0.0054231, -0.0094512, 0.0065623, -0.0146675, 0.0117665
3: 0.0001506, 0.0023450, 0.0003766, 0.0024957, -0.0019410, 0.0015571
4: 0.0020390, 0.0144314, 0.0011876, 0.0131551, -0.0087936, 0.0109616
5: 0.9960727, 0.9995157, 0.9958362, 0.9991611, -0.0024431, 0.0030455
6: 0.0043189, 0.0074441, 0.0041042, 0.0071222, -0.0022176, 0.0027644
7: -0.0072643, 0.0043985, -0.0080655, 0.0031972, -0.0082757, 0.0103161
8: -0.0126162, -0.0035391, -0.0116813, -0.0029155, -0.0080290, 0.0064410
9: -0.0037044, -0.0029213, -0.0037582, -0.0030019, -0.0005557, 0.0006927

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017723, upper bound: 0.0019252
time: 2.05 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017696, upper bound: 0.0019252
time: 2.23 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0135688, -0.0054816, -0.0128616, -0.0049463, -0.0071265, 0.0057869
1: -0.0067642, -0.0044841, -0.0065648, -0.0043332, -0.0020092, 0.0016315
2: -0.0113480, 0.0054749, -0.0098769, 0.0065886, -0.0148246, 0.0120380
3: 0.0001256, 0.0023518, 0.0003202, 0.0024992, -0.0019618, 0.0015930
4: 0.0020003, 0.0145727, 0.0011680, 0.0134732, -0.0089964, 0.0110790
5: 0.9960620, 0.9995549, 0.9958308, 0.9992495, -0.0024995, 0.0030781
6: 0.0043091, 0.0074797, 0.0040992, 0.0072024, -0.0022688, 0.0027940
7: -0.0073007, 0.0045314, -0.0080839, 0.0034967, -0.0084666, 0.0104266
8: -0.0127196, -0.0035107, -0.0119143, -0.0029011, -0.0081150, 0.0065896
9: -0.0037068, -0.0029123, -0.0037594, -0.0029818, -0.0005685, 0.0007001

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018584, upper bound: 0.0019152
time: 2.29 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018584, upper bound: 0.0019649
time: 2.27 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0132943, -0.0050718, -0.0128650, -0.0054345, -0.0062891, 0.0063127
1: -0.0066868, -0.0043686, -0.0065658, -0.0044709, -0.0017731, 0.0017798
2: -0.0107770, 0.0063276, -0.0098840, 0.0055729, -0.0130826, 0.0131317
3: 0.0002011, 0.0024646, 0.0003193, 0.0023648, -0.0017313, 0.0017378
4: 0.0013630, 0.0141459, 0.0019270, 0.0134785, -0.0098138, 0.0097771
5: 0.9958850, 0.9994364, 0.9960416, 0.9992509, -0.0027266, 0.0027164
6: 0.0041484, 0.0073721, 0.0042906, 0.0072038, -0.0024749, 0.0024657
7: -0.0079004, 0.0041298, -0.0073696, 0.0035017, -0.0092359, 0.0092014
8: -0.0124071, -0.0030440, -0.0119182, -0.0034571, -0.0071614, 0.0071883
9: -0.0037471, -0.0029393, -0.0037115, -0.0029815, -0.0006202, 0.0006179

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017276, upper bound: 0.0018479
time: 1.63 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016852, upper bound: 0.0018393
time: 1.83 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0133849, -0.0050473, -0.0130904, -0.0054089, -0.0063953, 0.0064614
1: -0.0067123, -0.0043617, -0.0066293, -0.0044636, -0.0018031, 0.0018217
2: -0.0109654, 0.0063783, -0.0103529, 0.0056262, -0.0133035, 0.0134410
3: 0.0001762, 0.0024714, 0.0002573, 0.0023718, -0.0017605, 0.0017787
4: 0.0013251, 0.0142867, 0.0018872, 0.0138290, -0.0100449, 0.0099422
5: 0.9958743, 0.9994755, 0.9960306, 0.9993483, -0.0027908, 0.0027622
6: 0.0041388, 0.0074076, 0.0042806, 0.0072921, -0.0025332, 0.0025073
7: -0.0079361, 0.0042622, -0.0074071, 0.0038315, -0.0094534, 0.0093567
8: -0.0125102, -0.0030162, -0.0121749, -0.0034279, -0.0072823, 0.0073576
9: -0.0037495, -0.0029304, -0.0037140, -0.0029593, -0.0006348, 0.0006283

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018689, upper bound: 0.0019151
time: 2.01 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018689, upper bound: 0.0019616
time: 2.05 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0132519, -0.0050532, -0.0128665, -0.0054356, -0.0063280, 0.0062781
1: -0.0066749, -0.0043634, -0.0065662, -0.0044712, -0.0017841, 0.0017700
2: -0.0106887, 0.0063661, -0.0098871, 0.0055706, -0.0131635, 0.0130598
3: 0.0002128, 0.0024697, 0.0003189, 0.0023645, -0.0017420, 0.0017283
4: 0.0013343, 0.0140800, 0.0019287, 0.0134809, -0.0097601, 0.0098375
5: 0.9958770, 0.9994181, 0.9960421, 0.9992516, -0.0027116, 0.0027332
6: 0.0041411, 0.0073554, 0.0042911, 0.0072043, -0.0024613, 0.0024809
7: -0.0079275, 0.0040677, -0.0073680, 0.0035039, -0.0091853, 0.0092582
8: -0.0123587, -0.0030229, -0.0119199, -0.0034583, -0.0072057, 0.0071489
9: -0.0037489, -0.0029435, -0.0037114, -0.0029813, -0.0006168, 0.0006217

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017173, upper bound: 0.0018514
time: 2.05 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016690, upper bound: 0.0018416
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0133392, -0.0050280, -0.0130917, -0.0054114, -0.0064209, 0.0064213
1: -0.0066995, -0.0043562, -0.0066297, -0.0044643, -0.0018103, 0.0018104
2: -0.0108705, 0.0064187, -0.0103555, 0.0056211, -0.0133569, 0.0133577
3: 0.0001888, 0.0024767, 0.0002569, 0.0023712, -0.0017676, 0.0017677
4: 0.0012949, 0.0142158, 0.0018910, 0.0138309, -0.0099827, 0.0099821
5: 0.9958660, 0.9994557, 0.9960316, 0.9993489, -0.0027735, 0.0027733
6: 0.0041312, 0.0073897, 0.0042816, 0.0072926, -0.0025175, 0.0025173
7: -0.0079645, 0.0041955, -0.0074035, 0.0038333, -0.0093948, 0.0093943
8: -0.0124582, -0.0029941, -0.0121763, -0.0034307, -0.0073116, 0.0073120
9: -0.0037514, -0.0029349, -0.0037137, -0.0029592, -0.0006308, 0.0006308

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018565, upper bound: 0.0019156
time: 2.00 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018565, upper bound: 0.0019659
time: 2.09 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0132943, -0.0050718, -0.0126555, -0.0049600, -0.0065987, 0.0059510
1: -0.0066868, -0.0043686, -0.0065067, -0.0043371, -0.0018604, 0.0016778
2: -0.0107770, 0.0063276, -0.0094482, 0.0065600, -0.0137266, 0.0123794
3: 0.0002011, 0.0024646, 0.0003770, 0.0024954, -0.0018165, 0.0016382
4: 0.0013630, 0.0141459, 0.0011894, 0.0131529, -0.0092516, 0.0102584
5: 0.9958850, 0.9994364, 0.9958367, 0.9991604, -0.0025704, 0.0028501
6: 0.0041484, 0.0073721, 0.0041046, 0.0071216, -0.0023331, 0.0025870
7: -0.0079004, 0.0041298, -0.0080638, 0.0031952, -0.0087068, 0.0096543
8: -0.0124071, -0.0030440, -0.0116797, -0.0029168, -0.0075140, 0.0067765
9: -0.0037471, -0.0029393, -0.0037581, -0.0030021, -0.0005846, 0.0006483

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017137, upper bound: 0.0018315
time: 1.90 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016663, upper bound: 0.0018170
time: 1.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0133849, -0.0050473, -0.0128604, -0.0049462, -0.0067035, 0.0060741
1: -0.0067123, -0.0043617, -0.0065645, -0.0043332, -0.0018900, 0.0017125
2: -0.0109654, 0.0063783, -0.0098744, 0.0065887, -0.0139445, 0.0126353
3: 0.0001762, 0.0024714, 0.0003206, 0.0024992, -0.0018453, 0.0016721
4: 0.0013251, 0.0142867, 0.0011679, 0.0134713, -0.0094428, 0.0104213
5: 0.9958743, 0.9994755, 0.9958307, 0.9992489, -0.0026235, 0.0028953
6: 0.0041388, 0.0074076, 0.0040992, 0.0072019, -0.0023813, 0.0026281
7: -0.0079361, 0.0042622, -0.0080841, 0.0034949, -0.0088868, 0.0098076
8: -0.0125102, -0.0030162, -0.0119129, -0.0029010, -0.0076333, 0.0069166
9: -0.0037495, -0.0029304, -0.0037594, -0.0029819, -0.0005967, 0.0006586

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018655, upper bound: 0.0019145
time: 2.04 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018655, upper bound: 0.0019616
time: 2.02 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0132519, -0.0050532, -0.0126569, -0.0049589, -0.0066399, 0.0059177
1: -0.0066749, -0.0043634, -0.0065071, -0.0043368, -0.0018720, 0.0016684
2: -0.0106887, 0.0063661, -0.0094512, 0.0065623, -0.0138123, 0.0123099
3: 0.0002128, 0.0024697, 0.0003766, 0.0024957, -0.0018278, 0.0016290
4: 0.0013343, 0.0140800, 0.0011876, 0.0131551, -0.0091997, 0.0103225
5: 0.9958770, 0.9994181, 0.9958362, 0.9991611, -0.0025559, 0.0028679
6: 0.0041411, 0.0073554, 0.0041042, 0.0071222, -0.0023200, 0.0026032
7: -0.0079275, 0.0040677, -0.0080655, 0.0031972, -0.0086579, 0.0097146
8: -0.0123587, -0.0030229, -0.0116813, -0.0029155, -0.0075609, 0.0067385
9: -0.0037489, -0.0029435, -0.0037582, -0.0030019, -0.0005814, 0.0006523

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017040, upper bound: 0.0018355
time: 1.93 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016499, upper bound: 0.0018197
time: 1.94 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0133392, -0.0050280, -0.0128616, -0.0049463, -0.0067314, 0.0060372
1: -0.0066995, -0.0043562, -0.0065648, -0.0043332, -0.0018978, 0.0017021
2: -0.0108705, 0.0064187, -0.0098769, 0.0065886, -0.0140026, 0.0125586
3: 0.0001888, 0.0024767, 0.0003202, 0.0024992, -0.0018530, 0.0016619
4: 0.0012949, 0.0142158, 0.0011680, 0.0134732, -0.0093855, 0.0104647
5: 0.9958660, 0.9994557, 0.9958308, 0.9992495, -0.0026076, 0.0029074
6: 0.0041312, 0.0073897, 0.0040992, 0.0072024, -0.0023669, 0.0026390
7: -0.0079645, 0.0041955, -0.0080839, 0.0034967, -0.0088328, 0.0098485
8: -0.0124582, -0.0029941, -0.0119143, -0.0029011, -0.0076651, 0.0068746
9: -0.0037514, -0.0029349, -0.0037594, -0.0029818, -0.0005931, 0.0006613

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018534, upper bound: 0.0019150
time: 1.99 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018534, upper bound: 0.0019659
time: 1.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0135319, -0.0055061, -0.0132955, -0.0053927, -0.0061709, 0.0059090
1: -0.0067538, -0.0044910, -0.0066871, -0.0044591, -0.0017398, 0.0016660
2: -0.0112713, 0.0054240, -0.0107794, 0.0056600, -0.0128368, 0.0122919
3: 0.0001357, 0.0023451, 0.0002008, 0.0023763, -0.0016987, 0.0016266
4: 0.0020383, 0.0145153, 0.0018619, 0.0141478, -0.0091862, 0.0095934
5: 0.9960725, 0.9995390, 0.9960235, 0.9994370, -0.0025522, 0.0026653
6: 0.0043187, 0.0074652, 0.0042742, 0.0073725, -0.0023166, 0.0024193
7: -0.0072649, 0.0044774, -0.0074308, 0.0041315, -0.0086452, 0.0090285
8: -0.0126776, -0.0035386, -0.0124084, -0.0034094, -0.0070269, 0.0067286
9: -0.0037044, -0.0029160, -0.0037156, -0.0029392, -0.0005805, 0.0006062

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018428, upper bound: 0.0019563
time: 1.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018461, upper bound: 0.0019563
time: 1.98 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0136261, -0.0054837, -0.0135172, -0.0053679, -0.0062763, 0.0060339
1: -0.0067804, -0.0044847, -0.0067497, -0.0044521, -0.0017695, 0.0017012
2: -0.0114672, 0.0054707, -0.0112408, 0.0057116, -0.0130560, 0.0125518
3: 0.0001098, 0.0023513, 0.0001398, 0.0023831, -0.0017278, 0.0016610
4: 0.0020034, 0.0146617, 0.0018234, 0.0144925, -0.0093804, 0.0097572
5: 0.9960628, 0.9995797, 0.9960127, 0.9995326, -0.0026062, 0.0027109
6: 0.0043099, 0.0075021, 0.0042645, 0.0074595, -0.0023656, 0.0024606
7: -0.0072977, 0.0046152, -0.0074672, 0.0044559, -0.0088280, 0.0091827
8: -0.0127849, -0.0035130, -0.0126609, -0.0033812, -0.0071469, 0.0068709
9: -0.0037066, -0.0029067, -0.0037180, -0.0029174, -0.0005928, 0.0006166

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019153, upper bound: 0.0019491
time: 2.27 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019153, upper bound: 0.0019902
time: 2.42 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0134780, -0.0055065, -0.0132972, -0.0053956, -0.0062381, 0.0058726
1: -0.0067386, -0.0044912, -0.0066876, -0.0044599, -0.0017588, 0.0016557
2: -0.0111590, 0.0054231, -0.0107831, 0.0056538, -0.0129766, 0.0122162
3: 0.0001506, 0.0023450, 0.0002003, 0.0023755, -0.0017172, 0.0016166
4: 0.0020390, 0.0144314, 0.0018666, 0.0141505, -0.0091296, 0.0096979
5: 0.9960727, 0.9995157, 0.9960248, 0.9994376, -0.0025365, 0.0026944
6: 0.0043189, 0.0074441, 0.0042754, 0.0073732, -0.0023024, 0.0024457
7: -0.0072643, 0.0043985, -0.0074265, 0.0041340, -0.0085920, 0.0091268
8: -0.0126162, -0.0035391, -0.0124104, -0.0034128, -0.0071034, 0.0066872
9: -0.0037044, -0.0029213, -0.0037153, -0.0029390, -0.0005769, 0.0006128

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018352, upper bound: 0.0019628
time: 2.28 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018365, upper bound: 0.0019628
time: 1.82 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0135688, -0.0054816, -0.0135187, -0.0053739, -0.0063331, 0.0059944
1: -0.0067642, -0.0044841, -0.0067501, -0.0044537, -0.0017855, 0.0016900
2: -0.0113480, 0.0054749, -0.0112437, 0.0056991, -0.0131742, 0.0124695
3: 0.0001256, 0.0023518, 0.0001394, 0.0023815, -0.0017434, 0.0016501
4: 0.0020003, 0.0145727, 0.0018327, 0.0144947, -0.0093189, 0.0098456
5: 0.9960620, 0.9995549, 0.9960154, 0.9995333, -0.0025891, 0.0027354
6: 0.0043091, 0.0074797, 0.0042668, 0.0074600, -0.0023501, 0.0024829
7: -0.0073007, 0.0045314, -0.0074584, 0.0044580, -0.0087702, 0.0092658
8: -0.0127196, -0.0035107, -0.0126625, -0.0033880, -0.0072116, 0.0068258
9: -0.0037068, -0.0029123, -0.0037174, -0.0029173, -0.0005889, 0.0006222

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019059, upper bound: 0.0019501
time: 2.11 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019059, upper bound: 0.0019963
time: 2.30 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0135319, -0.0055061, -0.0130818, -0.0049416, -0.0067040, 0.0057676
1: -0.0067538, -0.0044910, -0.0066269, -0.0043319, -0.0018901, 0.0016261
2: -0.0112713, 0.0054240, -0.0103350, 0.0065984, -0.0139456, 0.0119979
3: 0.0001357, 0.0023451, 0.0002596, 0.0025005, -0.0018455, 0.0015877
4: 0.0020383, 0.0145153, 0.0011606, 0.0138156, -0.0089665, 0.0104221
5: 0.9960725, 0.9995390, 0.9958287, 0.9993447, -0.0024911, 0.0028956
6: 0.0043187, 0.0074652, 0.0040974, 0.0072888, -0.0022612, 0.0026283
7: -0.0072649, 0.0044774, -0.0080908, 0.0038189, -0.0084384, 0.0098083
8: -0.0126776, -0.0035386, -0.0121651, -0.0028958, -0.0076338, 0.0065677
9: -0.0037044, -0.0029160, -0.0037599, -0.0029602, -0.0005666, 0.0006586

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018234, upper bound: 0.0019294
time: 2.11 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018238, upper bound: 0.0019293
time: 2.22 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0136261, -0.0054837, -0.0132850, -0.0049230, -0.0067991, 0.0058946
1: -0.0067804, -0.0044847, -0.0066842, -0.0043266, -0.0019169, 0.0016619
2: -0.0114672, 0.0054707, -0.0107577, 0.0066369, -0.0141436, 0.0122620
3: 0.0001098, 0.0023513, 0.0002037, 0.0025056, -0.0018717, 0.0016227
4: 0.0020034, 0.0146617, 0.0011318, 0.0141315, -0.0091639, 0.0105700
5: 0.9960628, 0.9995797, 0.9958207, 0.9994324, -0.0025460, 0.0029367
6: 0.0043099, 0.0075021, 0.0040901, 0.0073684, -0.0023110, 0.0026656
7: -0.0072977, 0.0046152, -0.0081180, 0.0041162, -0.0086242, 0.0099476
8: -0.0127849, -0.0035130, -0.0123965, -0.0028746, -0.0077422, 0.0067122
9: -0.0037066, -0.0029067, -0.0037617, -0.0029402, -0.0005791, 0.0006680

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018970, upper bound: 0.0019229
time: 2.51 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018970, upper bound: 0.0019669
time: 2.58 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0134780, -0.0055065, -0.0130834, -0.0049413, -0.0067736, 0.0057450
1: -0.0067386, -0.0044912, -0.0066274, -0.0043318, -0.0019097, 0.0016197
2: -0.0111590, 0.0054231, -0.0103383, 0.0065990, -0.0140905, 0.0119508
3: 0.0001506, 0.0023450, 0.0002592, 0.0025006, -0.0018647, 0.0015815
4: 0.0020390, 0.0144314, 0.0011602, 0.0138180, -0.0089313, 0.0105304
5: 0.9960727, 0.9995157, 0.9958286, 0.9993453, -0.0024814, 0.0029256
6: 0.0043189, 0.0074441, 0.0040973, 0.0072894, -0.0022523, 0.0026556
7: -0.0072643, 0.0043985, -0.0080913, 0.0038212, -0.0084053, 0.0099103
8: -0.0126162, -0.0035391, -0.0121669, -0.0028954, -0.0077132, 0.0065419
9: -0.0037044, -0.0029213, -0.0037599, -0.0029600, -0.0005644, 0.0006655

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018144, upper bound: 0.0019355
time: 2.37 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018133, upper bound: 0.0019355
time: 2.36 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0135688, -0.0054816, -0.0132864, -0.0049250, -0.0068583, 0.0058682
1: -0.0067642, -0.0044841, -0.0066846, -0.0043272, -0.0019336, 0.0016545
2: -0.0113480, 0.0054749, -0.0107605, 0.0066328, -0.0142667, 0.0122071
3: 0.0001256, 0.0023518, 0.0002033, 0.0025050, -0.0018880, 0.0016154
4: 0.0020003, 0.0145727, 0.0011349, 0.0141336, -0.0091228, 0.0106620
5: 0.9960620, 0.9995549, 0.9958215, 0.9994330, -0.0025346, 0.0029622
6: 0.0043091, 0.0074797, 0.0040909, 0.0073690, -0.0023006, 0.0026888
7: -0.0073007, 0.0045314, -0.0081151, 0.0041182, -0.0085856, 0.0100341
8: -0.0127196, -0.0035107, -0.0123981, -0.0028769, -0.0078096, 0.0066822
9: -0.0037068, -0.0029123, -0.0037615, -0.0029401, -0.0005765, 0.0006738

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018859, upper bound: 0.0019238
time: 2.22 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018859, upper bound: 0.0019721
time: 2.44 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0130818, -0.0049416, -0.0135319, -0.0055061, -0.0057676, 0.0067040
1: -0.0066269, -0.0043319, -0.0067538, -0.0044910, -0.0016261, 0.0018901
2: -0.0103350, 0.0065984, -0.0112713, 0.0054240, -0.0119979, 0.0139456
3: 0.0002596, 0.0025005, 0.0001357, 0.0023451, -0.0015877, 0.0018455
4: 0.0011606, 0.0138156, 0.0020383, 0.0145153, -0.0104221, 0.0089665
5: 0.9958287, 0.9993447, 0.9960725, 0.9995390, -0.0028956, 0.0024911
6: 0.0040974, 0.0072888, 0.0043187, 0.0074652, -0.0026283, 0.0022612
7: -0.0080908, 0.0038189, -0.0072649, 0.0044774, -0.0098083, 0.0084384
8: -0.0121651, -0.0028958, -0.0126776, -0.0035386, -0.0065677, 0.0076338
9: -0.0037599, -0.0029602, -0.0037044, -0.0029160, -0.0006586, 0.0005666

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017962, upper bound: 0.0018160
time: 2.31 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017648, upper bound: 0.0018068
time: 2.21 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0130818, -0.0049416, -0.0132943, -0.0050718, -0.0060206, 0.0062845
1: -0.0066269, -0.0043319, -0.0066868, -0.0043686, -0.0016974, 0.0017718
2: -0.0103350, 0.0065984, -0.0107770, 0.0063276, -0.0125241, 0.0130731
3: 0.0002596, 0.0025005, 0.0002011, 0.0024646, -0.0016574, 0.0017300
4: 0.0011606, 0.0138156, 0.0013630, 0.0141459, -0.0097700, 0.0093598
5: 0.9958287, 0.9993447, 0.9958850, 0.9994364, -0.0027144, 0.0026004
6: 0.0040974, 0.0072888, 0.0041484, 0.0073721, -0.0024639, 0.0023604
7: -0.0080908, 0.0038189, -0.0079004, 0.0041298, -0.0091947, 0.0088086
8: -0.0121651, -0.0028958, -0.0124071, -0.0030440, -0.0068557, 0.0071562
9: -0.0037599, -0.0029602, -0.0037471, -0.0029393, -0.0006174, 0.0005915

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017962, upper bound: 0.0018160
time: 2.37 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017648, upper bound: 0.0018068
time: 2.33 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0132850, -0.0049230, -0.0136261, -0.0054837, -0.0058946, 0.0067991
1: -0.0066842, -0.0043266, -0.0067804, -0.0044847, -0.0016619, 0.0019169
2: -0.0107577, 0.0066369, -0.0114672, 0.0054707, -0.0122620, 0.0141436
3: 0.0002037, 0.0025056, 0.0001098, 0.0023513, -0.0016227, 0.0018717
4: 0.0011318, 0.0141315, 0.0020034, 0.0146617, -0.0105700, 0.0091639
5: 0.9958207, 0.9994324, 0.9960628, 0.9995797, -0.0029367, 0.0025460
6: 0.0040901, 0.0073684, 0.0043099, 0.0075021, -0.0026656, 0.0023110
7: -0.0081180, 0.0041162, -0.0072977, 0.0046152, -0.0099476, 0.0086242
8: -0.0123965, -0.0028746, -0.0127849, -0.0035130, -0.0067122, 0.0077422
9: -0.0037617, -0.0029402, -0.0037066, -0.0029067, -0.0006680, 0.0005791

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018041, upper bound: 0.0018802
time: 2.25 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017731, upper bound: 0.0018695
time: 2.01 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0132850, -0.0049230, -0.0133849, -0.0050491, -0.0061394, 0.0063947
1: -0.0066842, -0.0043266, -0.0067123, -0.0043622, -0.0017309, 0.0018029
2: -0.0107577, 0.0066369, -0.0109654, 0.0063746, -0.0127713, 0.0133022
3: 0.0002037, 0.0025056, 0.0001762, 0.0024709, -0.0016901, 0.0017603
4: 0.0011318, 0.0141315, 0.0013279, 0.0142867, -0.0099413, 0.0095445
5: 0.9958207, 0.9994324, 0.9958752, 0.9994755, -0.0027620, 0.0026517
6: 0.0040901, 0.0073684, 0.0041395, 0.0074076, -0.0025070, 0.0024070
7: -0.0081180, 0.0041162, -0.0079335, 0.0042622, -0.0093558, 0.0089824
8: -0.0123965, -0.0028746, -0.0125102, -0.0030182, -0.0069910, 0.0072817
9: -0.0037617, -0.0029402, -0.0037493, -0.0029304, -0.0006282, 0.0006032

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018041, upper bound: 0.0018802
time: 1.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017731, upper bound: 0.0018695
time: 2.31 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0132864, -0.0049250, -0.0135688, -0.0054816, -0.0058682, 0.0068583
1: -0.0066846, -0.0043272, -0.0067642, -0.0044841, -0.0016545, 0.0019336
2: -0.0107605, 0.0066328, -0.0113480, 0.0054749, -0.0122071, 0.0142667
3: 0.0002033, 0.0025050, 0.0001256, 0.0023518, -0.0016154, 0.0018880
4: 0.0011349, 0.0141336, 0.0020003, 0.0145727, -0.0106620, 0.0091228
5: 0.9958215, 0.9994330, 0.9960620, 0.9995549, -0.0029622, 0.0025346
6: 0.0040909, 0.0073690, 0.0043091, 0.0074797, -0.0026888, 0.0023006
7: -0.0081151, 0.0041182, -0.0073007, 0.0045314, -0.0100341, 0.0085856
8: -0.0123981, -0.0028769, -0.0127196, -0.0035107, -0.0066822, 0.0078096
9: -0.0037615, -0.0029401, -0.0037068, -0.0029123, -0.0006738, 0.0005765

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018043, upper bound: 0.0018608
time: 2.13 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017736, upper bound: 0.0018501
time: 1.96 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0132864, -0.0049250, -0.0133392, -0.0050280, -0.0061065, 0.0064536
1: -0.0066846, -0.0043272, -0.0066995, -0.0043562, -0.0017216, 0.0018195
2: -0.0107605, 0.0066328, -0.0108705, 0.0064187, -0.0127027, 0.0134248
3: 0.0002033, 0.0025050, 0.0001888, 0.0024767, -0.0016810, 0.0017766
4: 0.0011349, 0.0141336, 0.0012949, 0.0142158, -0.0100328, 0.0094932
5: 0.9958215, 0.9994330, 0.9958660, 0.9994557, -0.0027874, 0.0026375
6: 0.0040909, 0.0073690, 0.0041312, 0.0073897, -0.0025301, 0.0023940
7: -0.0081151, 0.0041182, -0.0079645, 0.0041955, -0.0094420, 0.0089341
8: -0.0123981, -0.0028769, -0.0124582, -0.0029941, -0.0069535, 0.0073487
9: -0.0037615, -0.0029401, -0.0037514, -0.0029349, -0.0006340, 0.0005999

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018043, upper bound: 0.0018609
time: 2.48 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017736, upper bound: 0.0018500
time: 2.26 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 6.83 seconds
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019280, upper bound: 0.0018530
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0018530
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0020085, upper bound: 0.0018540
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0020085, upper bound: 0.0018943
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018863, upper bound: 0.0018201
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017792, upper bound: 0.0018201
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019924, upper bound: 0.0018543
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019924, upper bound: 0.0018961
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019790, upper bound: 0.0018101
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019790, upper bound: 0.0018592
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019615, upper bound: 0.0018101
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019615, upper bound: 0.0018615
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018196, upper bound: 0.0019495
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018211, upper bound: 0.0019495
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019051, upper bound: 0.0019457
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019051, upper bound: 0.0019866
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018114, upper bound: 0.0019570
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018107, upper bound: 0.0019570
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018948, upper bound: 0.0019471
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018948, upper bound: 0.0019928
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017811, upper bound: 0.0019185
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017795, upper bound: 0.0019185
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018702, upper bound: 0.0019146
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018719, upper bound: 0.0019588
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017723, upper bound: 0.0019252
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017696, upper bound: 0.0019252
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018584, upper bound: 0.0019152
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018584, upper bound: 0.0019649
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017276, upper bound: 0.0018479
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0016852, upper bound: 0.0018393
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018689, upper bound: 0.0019151
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018689, upper bound: 0.0019616
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017173, upper bound: 0.0018514
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0016690, upper bound: 0.0018416
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018565, upper bound: 0.0019156
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018565, upper bound: 0.0019659
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017137, upper bound: 0.0018315
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0016663, upper bound: 0.0018170
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018655, upper bound: 0.0019145
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018655, upper bound: 0.0019616
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017040, upper bound: 0.0018355
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0016499, upper bound: 0.0018197
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018534, upper bound: 0.0019150
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018534, upper bound: 0.0019659
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018428, upper bound: 0.0019563
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018461, upper bound: 0.0019563
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019153, upper bound: 0.0019491
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019153, upper bound: 0.0019902
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018352, upper bound: 0.0019628
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018365, upper bound: 0.0019628
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019059, upper bound: 0.0019501
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019059, upper bound: 0.0019963
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018234, upper bound: 0.0019294
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018238, upper bound: 0.0019293
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018970, upper bound: 0.0019229
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018970, upper bound: 0.0019669
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018144, upper bound: 0.0019355
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018133, upper bound: 0.0019355
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018859, upper bound: 0.0019238
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018859, upper bound: 0.0019721
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017962, upper bound: 0.0018160
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017648, upper bound: 0.0018068
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017962, upper bound: 0.0018160
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017648, upper bound: 0.0018068
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018041, upper bound: 0.0018802
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017731, upper bound: 0.0018695
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018041, upper bound: 0.0018802
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017731, upper bound: 0.0018695
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018043, upper bound: 0.0018608
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017736, upper bound: 0.0018501
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018043, upper bound: 0.0018609
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017736, upper bound: 0.0018500

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0128854, -0.0055724, -0.0135172, -0.0053679, -0.0058896, 0.0063747
1: -0.0065715, -0.0045097, -0.0067497, -0.0044521, -0.0016605, 0.0017973
2: -0.0099264, 0.0052860, -0.0112408, 0.0057116, -0.0122517, 0.0132606
3: 0.0003137, 0.0023268, 0.0001398, 0.0023831, -0.0016213, 0.0017548
4: 0.0021414, 0.0135103, 0.0018234, 0.0144925, -0.0099102, 0.0091561
5: 0.9961011, 0.9992598, 0.9960127, 0.9995326, -0.0027533, 0.0025438
6: 0.0043447, 0.0072118, 0.0042645, 0.0074595, -0.0024992, 0.0023090
7: -0.0071678, 0.0035315, -0.0074672, 0.0044559, -0.0093266, 0.0086169
8: -0.0119414, -0.0036141, -0.0126609, -0.0033812, -0.0067066, 0.0072589
9: -0.0036979, -0.0029795, -0.0037180, -0.0029174, -0.0006263, 0.0005786

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0018111
time: 2.18 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0018119
time: 2.53 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0131215, -0.0055474, -0.0135172, -0.0053679, -0.0059975, 0.0062702
1: -0.0066381, -0.0045027, -0.0067497, -0.0044521, -0.0016909, 0.0017678
2: -0.0104176, 0.0053382, -0.0112408, 0.0057116, -0.0124761, 0.0130432
3: 0.0002487, 0.0023337, 0.0001398, 0.0023831, -0.0016510, 0.0017261
4: 0.0021024, 0.0138773, 0.0018234, 0.0144925, -0.0097477, 0.0093239
5: 0.9960904, 0.9993617, 0.9960127, 0.9995326, -0.0027082, 0.0025904
6: 0.0043349, 0.0073043, 0.0042645, 0.0074595, -0.0024582, 0.0023513
7: -0.0072045, 0.0038769, -0.0074672, 0.0044559, -0.0091736, 0.0087748
8: -0.0122103, -0.0035856, -0.0126609, -0.0033812, -0.0068294, 0.0071399
9: -0.0037004, -0.0029563, -0.0037180, -0.0029174, -0.0006160, 0.0005892

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0018609
time: 2.23 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0018598
time: 2.48 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0128599, -0.0055630, -0.0135187, -0.0053739, -0.0059777, 0.0063875
1: -0.0065644, -0.0045071, -0.0067501, -0.0044537, -0.0016853, 0.0018009
2: -0.0098734, 0.0053056, -0.0112437, 0.0056991, -0.0124349, 0.0132873
3: 0.0003207, 0.0023294, 0.0001394, 0.0023815, -0.0016456, 0.0017584
4: 0.0021268, 0.0134706, 0.0018327, 0.0144947, -0.0099301, 0.0092931
5: 0.9960971, 0.9992487, 0.9960154, 0.9995333, -0.0027589, 0.0025819
6: 0.0043410, 0.0072018, 0.0042668, 0.0074600, -0.0025042, 0.0023436
7: -0.0071816, 0.0034942, -0.0074584, 0.0044580, -0.0093453, 0.0087458
8: -0.0119124, -0.0036034, -0.0126625, -0.0033880, -0.0068069, 0.0072735
9: -0.0036989, -0.0029820, -0.0037174, -0.0029173, -0.0006275, 0.0005873

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019131, upper bound: 0.0018114
time: 2.19 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019131, upper bound: 0.0018124
time: 2.40 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0130848, -0.0055392, -0.0135187, -0.0053739, -0.0060877, 0.0062631
1: -0.0066277, -0.0045004, -0.0067501, -0.0044537, -0.0017164, 0.0017658
2: -0.0103412, 0.0053552, -0.0112437, 0.0056991, -0.0126637, 0.0130285
3: 0.0002588, 0.0023360, 0.0001394, 0.0023815, -0.0016758, 0.0017241
4: 0.0020897, 0.0138202, 0.0018327, 0.0144947, -0.0097367, 0.0094640
5: 0.9960868, 0.9993459, 0.9960154, 0.9995333, -0.0027051, 0.0026294
6: 0.0043317, 0.0072899, 0.0042668, 0.0074600, -0.0024555, 0.0023867
7: -0.0072165, 0.0038232, -0.0074584, 0.0044580, -0.0091633, 0.0089067
8: -0.0121685, -0.0035762, -0.0126625, -0.0033880, -0.0069321, 0.0071318
9: -0.0037012, -0.0029599, -0.0037174, -0.0029173, -0.0006153, 0.0005981

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019131, upper bound: 0.0018626
time: 2.30 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019131, upper bound: 0.0018622
time: 2.35 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0128854, -0.0055724, -0.0132850, -0.0049230, -0.0063907, 0.0061848
1: -0.0065715, -0.0045097, -0.0066842, -0.0043266, -0.0018018, 0.0017437
2: -0.0099264, 0.0052860, -0.0107577, 0.0066369, -0.0132940, 0.0128656
3: 0.0003137, 0.0023268, 0.0002037, 0.0025056, -0.0017592, 0.0017026
4: 0.0021414, 0.0135103, 0.0011318, 0.0141315, -0.0096150, 0.0099351
5: 0.9961011, 0.9992598, 0.9958207, 0.9994324, -0.0026713, 0.0027603
6: 0.0043447, 0.0072118, 0.0040901, 0.0073684, -0.0024248, 0.0025055
7: -0.0071678, 0.0035315, -0.0081180, 0.0041162, -0.0090487, 0.0093500
8: -0.0119414, -0.0036141, -0.0123965, -0.0028746, -0.0072772, 0.0070427
9: -0.0036979, -0.0029795, -0.0037617, -0.0029402, -0.0006076, 0.0006278

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018201, upper bound: 0.0017174
time: 2.15 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018121, upper bound: 0.0016684
time: 2.31 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0131215, -0.0055474, -0.0132850, -0.0049230, -0.0065128, 0.0061081
1: -0.0066381, -0.0045027, -0.0066842, -0.0043266, -0.0018362, 0.0017221
2: -0.0104176, 0.0053382, -0.0107577, 0.0066369, -0.0135480, 0.0127061
3: 0.0002487, 0.0023337, 0.0002037, 0.0025056, -0.0017929, 0.0016815
4: 0.0021024, 0.0138773, 0.0011318, 0.0141315, -0.0094958, 0.0101249
5: 0.9960904, 0.9993617, 0.9958207, 0.9994324, -0.0026382, 0.0028130
6: 0.0043349, 0.0073043, 0.0040901, 0.0073684, -0.0023947, 0.0025534
7: -0.0072045, 0.0038769, -0.0081180, 0.0041162, -0.0089366, 0.0095287
8: -0.0122103, -0.0035856, -0.0123965, -0.0028746, -0.0074162, 0.0069554
9: -0.0037004, -0.0029563, -0.0037617, -0.0029402, -0.0006001, 0.0006398

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018201, upper bound: 0.0017794
time: 2.27 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018121, upper bound: 0.0017394
time: 2.35 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0128599, -0.0055630, -0.0132864, -0.0049250, -0.0064793, 0.0061977
1: -0.0065644, -0.0045071, -0.0066846, -0.0043272, -0.0018268, 0.0017474
2: -0.0098734, 0.0053056, -0.0107605, 0.0066328, -0.0134783, 0.0128924
3: 0.0003207, 0.0023294, 0.0002033, 0.0025050, -0.0017836, 0.0017061
4: 0.0021268, 0.0134706, 0.0011349, 0.0141336, -0.0096350, 0.0100728
5: 0.9960971, 0.9992487, 0.9958215, 0.9994330, -0.0026769, 0.0027985
6: 0.0043410, 0.0072018, 0.0040909, 0.0073690, -0.0024298, 0.0025402
7: -0.0071816, 0.0034942, -0.0081151, 0.0041182, -0.0090676, 0.0094797
8: -0.0119124, -0.0036034, -0.0123981, -0.0028769, -0.0073780, 0.0070573
9: -0.0036989, -0.0029820, -0.0037615, -0.0029401, -0.0006089, 0.0006365

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016949, upper bound: 0.0017174
time: 2.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017967, upper bound: 0.0016690
time: 2.32 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0130848, -0.0055392, -0.0132864, -0.0049250, -0.0066053, 0.0060991
1: -0.0066277, -0.0045004, -0.0066846, -0.0043272, -0.0018623, 0.0017196
2: -0.0103412, 0.0053552, -0.0107605, 0.0066328, -0.0137404, 0.0126875
3: 0.0002588, 0.0023360, 0.0002033, 0.0025050, -0.0018183, 0.0016790
4: 0.0020897, 0.0138202, 0.0011349, 0.0141336, -0.0094818, 0.0102687
5: 0.9960868, 0.9993459, 0.9958215, 0.9994330, -0.0026343, 0.0028530
6: 0.0043317, 0.0072899, 0.0040909, 0.0073690, -0.0023912, 0.0025896
7: -0.0072165, 0.0038232, -0.0081151, 0.0041182, -0.0089234, 0.0096640
8: -0.0121685, -0.0035762, -0.0123981, -0.0028769, -0.0075215, 0.0069451
9: -0.0037012, -0.0029599, -0.0037615, -0.0029401, -0.0005992, 0.0006489

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016949, upper bound: 0.0017807
time: 2.28 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017967, upper bound: 0.0017413
time: 2.37 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0135218, -0.0056284, -0.0128637, -0.0054513, -0.0063889, 0.0056561
1: -0.0067509, -0.0045255, -0.0065654, -0.0044756, -0.0018013, 0.0015947
2: -0.0112501, 0.0051697, -0.0098813, 0.0055381, -0.0132901, 0.0117658
3: 0.0001385, 0.0023114, 0.0003197, 0.0023602, -0.0017587, 0.0015570
4: 0.0022284, 0.0144995, 0.0019530, 0.0134766, -0.0087930, 0.0099322
5: 0.9961253, 0.9995347, 0.9960488, 0.9992504, -0.0024430, 0.0027595
6: 0.0043666, 0.0074612, 0.0042972, 0.0072033, -0.0022175, 0.0025048
7: -0.0070860, 0.0044625, -0.0073451, 0.0034998, -0.0082752, 0.0093473
8: -0.0126661, -0.0036778, -0.0119168, -0.0034762, -0.0072750, 0.0064406
9: -0.0036924, -0.0029170, -0.0037098, -0.0029816, -0.0005557, 0.0006277

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017297, upper bound: 0.0018799
time: 1.97 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017294, upper bound: 0.0018730
time: 2.25 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0135655, -0.0054608, -0.0128615, -0.0055024, -0.0064077, 0.0058651
1: -0.0067633, -0.0044783, -0.0065648, -0.0044900, -0.0018066, 0.0016536
2: -0.0113411, 0.0055184, -0.0098766, 0.0054318, -0.0133294, 0.0122007
3: 0.0001265, 0.0023576, 0.0003203, 0.0023461, -0.0017639, 0.0016146
4: 0.0019678, 0.0145675, 0.0020325, 0.0134730, -0.0091180, 0.0099616
5: 0.9960529, 0.9995535, 0.9960710, 0.9992495, -0.0025333, 0.0027676
6: 0.0043009, 0.0074784, 0.0043172, 0.0072024, -0.0022994, 0.0025122
7: -0.0073312, 0.0045265, -0.0072703, 0.0034965, -0.0085811, 0.0093749
8: -0.0127159, -0.0034870, -0.0119142, -0.0035344, -0.0072965, 0.0066787
9: -0.0037089, -0.0029127, -0.0037048, -0.0029818, -0.0005762, 0.0006295

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017257, upper bound: 0.0018799
time: 2.19 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017248, upper bound: 0.0018730
time: 1.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0133378, -0.0055343, -0.0130904, -0.0054089, -0.0062974, 0.0060271
1: -0.0066991, -0.0044990, -0.0066293, -0.0044636, -0.0017755, 0.0016993
2: -0.0108675, 0.0053653, -0.0103529, 0.0056262, -0.0130998, 0.0125377
3: 0.0001892, 0.0023373, 0.0002573, 0.0023718, -0.0017336, 0.0016592
4: 0.0020822, 0.0142136, 0.0018872, 0.0138290, -0.0093699, 0.0097900
5: 0.9960847, 0.9994552, 0.9960306, 0.9993483, -0.0026032, 0.0027200
6: 0.0043298, 0.0073891, 0.0042806, 0.0072921, -0.0023629, 0.0024689
7: -0.0072236, 0.0041934, -0.0074071, 0.0038315, -0.0088181, 0.0092135
8: -0.0124566, -0.0035707, -0.0121749, -0.0034279, -0.0071709, 0.0068631
9: -0.0037017, -0.0029350, -0.0037140, -0.0029593, -0.0005921, 0.0006187

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017872, upper bound: 0.0018851
time: 2.55 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017872, upper bound: 0.0018763
time: 2.52 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0135725, -0.0055061, -0.0130904, -0.0054089, -0.0063970, 0.0059310
1: -0.0067653, -0.0044910, -0.0066293, -0.0044636, -0.0018036, 0.0016722
2: -0.0113557, 0.0054241, -0.0103529, 0.0056262, -0.0133072, 0.0123377
3: 0.0001245, 0.0023451, 0.0002573, 0.0023718, -0.0017610, 0.0016327
4: 0.0020383, 0.0145784, 0.0018872, 0.0138290, -0.0092204, 0.0099449
5: 0.9960725, 0.9995565, 0.9960306, 0.9993483, -0.0025617, 0.0027630
6: 0.0043187, 0.0074811, 0.0042806, 0.0072921, -0.0023253, 0.0025080
7: -0.0072649, 0.0045368, -0.0074071, 0.0038315, -0.0086774, 0.0093593
8: -0.0127239, -0.0035386, -0.0121749, -0.0034279, -0.0072844, 0.0067537
9: -0.0037044, -0.0029120, -0.0037140, -0.0029593, -0.0005827, 0.0006285

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017872, upper bound: 0.0019260
time: 2.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017872, upper bound: 0.0019220
time: 2.31 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0134679, -0.0056269, -0.0128652, -0.0054523, -0.0064256, 0.0056129
1: -0.0067358, -0.0045251, -0.0065659, -0.0044759, -0.0018116, 0.0015825
2: -0.0111382, 0.0051728, -0.0098845, 0.0055360, -0.0133665, 0.0116759
3: 0.0001533, 0.0023118, 0.0003192, 0.0023599, -0.0017688, 0.0015451
4: 0.0022261, 0.0144158, 0.0019546, 0.0134789, -0.0087259, 0.0099893
5: 0.9961247, 0.9995114, 0.9960493, 0.9992511, -0.0024243, 0.0027753
6: 0.0043660, 0.0074401, 0.0042976, 0.0072039, -0.0022005, 0.0025192
7: -0.0070882, 0.0043838, -0.0073436, 0.0035020, -0.0082120, 0.0094010
8: -0.0126048, -0.0036761, -0.0119185, -0.0034773, -0.0073168, 0.0063914
9: -0.0036926, -0.0029223, -0.0037097, -0.0029815, -0.0005514, 0.0006313

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017183, upper bound: 0.0018856
time: 2.02 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017178, upper bound: 0.0018794
time: 2.28 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0135133, -0.0054607, -0.0128630, -0.0055037, -0.0064464, 0.0058239
1: -0.0067486, -0.0044782, -0.0065652, -0.0044904, -0.0018175, 0.0016420
2: -0.0112326, 0.0055184, -0.0098797, 0.0054290, -0.0134098, 0.0121148
3: 0.0001408, 0.0023576, 0.0003199, 0.0023457, -0.0017746, 0.0016032
4: 0.0019678, 0.0144864, 0.0020346, 0.0134754, -0.0090538, 0.0100216
5: 0.9960529, 0.9995310, 0.9960715, 0.9992501, -0.0025154, 0.0027843
6: 0.0043009, 0.0074579, 0.0043178, 0.0072030, -0.0022832, 0.0025273
7: -0.0073313, 0.0044502, -0.0072684, 0.0034987, -0.0085207, 0.0094315
8: -0.0126564, -0.0034869, -0.0119159, -0.0035359, -0.0073405, 0.0066317
9: -0.0037089, -0.0029178, -0.0037047, -0.0029817, -0.0005721, 0.0006333

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017128, upper bound: 0.0018856
time: 2.15 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017117, upper bound: 0.0018794
time: 2.09 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0132905, -0.0055254, -0.0130917, -0.0054114, -0.0063389, 0.0060062
1: -0.0066857, -0.0044965, -0.0066297, -0.0044643, -0.0017872, 0.0016934
2: -0.0107691, 0.0053838, -0.0103555, 0.0056211, -0.0131861, 0.0124941
3: 0.0002022, 0.0023398, 0.0002569, 0.0023712, -0.0017450, 0.0016534
4: 0.0020683, 0.0141400, 0.0018910, 0.0138309, -0.0093373, 0.0098545
5: 0.9960809, 0.9994348, 0.9960316, 0.9993489, -0.0025942, 0.0027379
6: 0.0043263, 0.0073706, 0.0042816, 0.0072926, -0.0023547, 0.0024852
7: -0.0072366, 0.0041242, -0.0074035, 0.0038333, -0.0087874, 0.0092742
8: -0.0124027, -0.0035606, -0.0121763, -0.0034307, -0.0072181, 0.0068393
9: -0.0037025, -0.0029397, -0.0037137, -0.0029592, -0.0005901, 0.0006227

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017766, upper bound: 0.0018863
time: 2.36 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017766, upper bound: 0.0018780
time: 2.47 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0135116, -0.0055080, -0.0130917, -0.0054114, -0.0064436, 0.0058781
1: -0.0067481, -0.0044916, -0.0066297, -0.0044643, -0.0018167, 0.0016573
2: -0.0112291, 0.0054201, -0.0103555, 0.0056211, -0.0134039, 0.0122277
3: 0.0001413, 0.0023446, 0.0002569, 0.0023712, -0.0017738, 0.0016181
4: 0.0020413, 0.0144838, 0.0018910, 0.0138309, -0.0091382, 0.0100173
5: 0.9960733, 0.9995303, 0.9960316, 0.9993489, -0.0025389, 0.0027831
6: 0.0043194, 0.0074573, 0.0042816, 0.0072926, -0.0023045, 0.0025262
7: -0.0072621, 0.0044477, -0.0074035, 0.0038333, -0.0086001, 0.0094274
8: -0.0126545, -0.0035408, -0.0121763, -0.0034307, -0.0073373, 0.0066935
9: -0.0037043, -0.0029180, -0.0037137, -0.0029592, -0.0005775, 0.0006330

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017766, upper bound: 0.0019314
time: 2.35 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017766, upper bound: 0.0019282
time: 2.24 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0135725, -0.0055061, -0.0128604, -0.0049462, -0.0069454, 0.0057879
1: -0.0067653, -0.0044910, -0.0065645, -0.0043332, -0.0019582, 0.0016318
2: -0.0113557, 0.0054241, -0.0098744, 0.0065887, -0.0144479, 0.0120401
3: 0.0001245, 0.0023451, 0.0003206, 0.0024992, -0.0019119, 0.0015933
4: 0.0020383, 0.0145784, 0.0011679, 0.0134713, -0.0089980, 0.0107975
5: 0.9960725, 0.9995565, 0.9958307, 0.9992489, -0.0024999, 0.0029999
6: 0.0043187, 0.0074811, 0.0040992, 0.0072019, -0.0022692, 0.0027230
7: -0.0072649, 0.0045368, -0.0080841, 0.0034949, -0.0084681, 0.0101616
8: -0.0127239, -0.0035386, -0.0119129, -0.0029010, -0.0079088, 0.0065908
9: -0.0037044, -0.0029120, -0.0037594, -0.0029819, -0.0005686, 0.0006823

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017063, upper bound: 0.0018779
time: 2.26 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016992, upper bound: 0.0018489
time: 2.01 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0135116, -0.0055080, -0.0128616, -0.0049463, -0.0069881, 0.0057540
1: -0.0067481, -0.0044916, -0.0065648, -0.0043332, -0.0019702, 0.0016223
2: -0.0112291, 0.0054201, -0.0098769, 0.0065886, -0.0145367, 0.0119694
3: 0.0001413, 0.0023446, 0.0003202, 0.0024992, -0.0019237, 0.0015840
4: 0.0020413, 0.0144838, 0.0011680, 0.0134732, -0.0089452, 0.0108638
5: 0.9960733, 0.9995303, 0.9958308, 0.9992495, -0.0024852, 0.0030183
6: 0.0043194, 0.0074573, 0.0040992, 0.0072024, -0.0022558, 0.0027397
7: -0.0072621, 0.0044477, -0.0080839, 0.0034967, -0.0084184, 0.0102241
8: -0.0126545, -0.0035408, -0.0119143, -0.0029011, -0.0079574, 0.0065521
9: -0.0037043, -0.0029180, -0.0037594, -0.0029818, -0.0005653, 0.0006865

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016949, upper bound: 0.0018820
time: 2.14 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016887, upper bound: 0.0018535
time: 2.05 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0133281, -0.0050721, -0.0130904, -0.0054089, -0.0062308, 0.0064311
1: -0.0066963, -0.0043687, -0.0066293, -0.0044636, -0.0017567, 0.0018132
2: -0.0108473, 0.0063268, -0.0103529, 0.0056262, -0.0129612, 0.0133780
3: 0.0001918, 0.0024645, 0.0002573, 0.0023718, -0.0017152, 0.0017704
4: 0.0013636, 0.0141985, 0.0018872, 0.0138290, -0.0099978, 0.0096864
5: 0.9958851, 0.9994510, 0.9960306, 0.9993483, -0.0027777, 0.0026912
6: 0.0041486, 0.0073853, 0.0042806, 0.0072921, -0.0025213, 0.0024428
7: -0.0078998, 0.0041792, -0.0074071, 0.0038315, -0.0094091, 0.0091160
8: -0.0124455, -0.0030444, -0.0121749, -0.0034279, -0.0070950, 0.0073231
9: -0.0037471, -0.0029360, -0.0037140, -0.0029593, -0.0006318, 0.0006121

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016909, upper bound: 0.0018752
time: 2.24 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016852, upper bound: 0.0018511
time: 2.45 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.70 + 597.42 = 602.12 seconds
