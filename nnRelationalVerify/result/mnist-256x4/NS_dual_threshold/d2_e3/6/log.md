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
execution time: IAR + RelationalAnalysis = 1.83 + 2.99 = 4.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0021507, upper bound: 0.0021506

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021094, upper bound: 0.0020113
time: 1.69 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021129, upper bound: 0.0021129
time: 2.00 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.88 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.88
Output dim: 5, lower bound: -0.0021094, upper bound: 0.0020113
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.88
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

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020308, upper bound: 0.0019261
time: 2.10 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020308, upper bound: 0.0019299
time: 2.03 seconds

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

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020113, upper bound: 0.0021094
time: 1.97 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020113, upper bound: 0.0021129
time: 2.07 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.87 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 5.87
Output dim: 5, lower bound: -0.0020308, upper bound: 0.0019261
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 5.87
Output dim: 5, lower bound: -0.0020308, upper bound: 0.0019299
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.87
Output dim: 5, lower bound: -0.0020113, upper bound: 0.0021094
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.87
Output dim: 5, lower bound: -0.0020113, upper bound: 0.0021129

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -0.0131583, -0.0053365, -0.0136221, -0.0052798, -0.0063519, 0.0067264
1: -0.0066485, -0.0044432, -0.0067792, -0.0044272, -0.0017908, 0.0018964
2: -0.0104941, 0.0057769, -0.0114589, 0.0058948, -0.0132132, 0.0139922
3: 0.0002386, 0.0023918, 0.0001109, 0.0024074, -0.0017486, 0.0018516
4: 0.0017746, 0.0139345, 0.0016865, 0.0146555, -0.0104569, 0.0098747
5: 0.9959993, 0.9993777, 0.9959748, 0.9995780, -0.0029052, 0.0027435
6: 0.0042522, 0.0073187, 0.0042300, 0.0075006, -0.0026371, 0.0024903
7: -0.0075131, 0.0039308, -0.0075960, 0.0046093, -0.0098411, 0.0092932
8: -0.0122522, -0.0033454, -0.0127803, -0.0032809, -0.0072329, 0.0076594
9: -0.0037211, -0.0029527, -0.0037267, -0.0029071, -0.0006608, 0.0006240

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019377, upper bound: 0.0019261
time: 2.05 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019377, upper bound: 0.0019261
time: 2.25 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -0.0129305, -0.0048721, -0.0135057, -0.0053075, -0.0063255, 0.0071403
1: -0.0065842, -0.0043123, -0.0067464, -0.0044351, -0.0017834, 0.0020131
2: -0.0100201, 0.0067428, -0.0112167, 0.0058371, -0.0131584, 0.0148532
3: 0.0003013, 0.0025196, 0.0001429, 0.0023997, -0.0017413, 0.0019656
4: 0.0010527, 0.0135803, 0.0017296, 0.0144745, -0.0111004, 0.0098338
5: 0.9957988, 0.9992793, 0.9959867, 0.9995277, -0.0030840, 0.0027321
6: 0.0040701, 0.0072294, 0.0042408, 0.0074549, -0.0027994, 0.0024799
7: -0.0081924, 0.0035974, -0.0075554, 0.0044390, -0.0104467, 0.0092547
8: -0.0119927, -0.0028167, -0.0126478, -0.0033125, -0.0072029, 0.0081307
9: -0.0037667, -0.0029751, -0.0037240, -0.0029185, -0.0007015, 0.0006214

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019350, upper bound: 0.0019299
time: 2.00 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019350, upper bound: 0.0019299
time: 2.41 seconds

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

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019261, upper bound: 0.0020308
time: 2.29 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019299, upper bound: 0.0020308
time: 2.37 seconds

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

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019305, upper bound: 0.0020359
time: 2.04 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019299, upper bound: 0.0020373
time: 2.25 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.31 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 6.31
Output dim: 5, lower bound: -0.0019377, upper bound: 0.0019261
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 6.31
Output dim: 5, lower bound: -0.0019377, upper bound: 0.0019261
NS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 6.31
Output dim: 5, lower bound: -0.0019350, upper bound: 0.0019299
NS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 6.31
Output dim: 5, lower bound: -0.0019350, upper bound: 0.0019299
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 6.31
Output dim: 5, lower bound: -0.0019261, upper bound: 0.0020308
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 6.31
Output dim: 5, lower bound: -0.0019299, upper bound: 0.0020308
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.31
Output dim: 5, lower bound: -0.0019305, upper bound: 0.0020359
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.31
Output dim: 5, lower bound: -0.0019299, upper bound: 0.0020373

## BFS NS instance: NS_A1_A1_B1

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

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019369, upper bound: 0.0019261
time: 1.71 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019369, upper bound: 0.0019261
time: 2.30 seconds

## BFS NS instance: NS_A1_A1_B2

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

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019369, upper bound: 0.0019261
time: 2.26 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019369, upper bound: 0.0019261
time: 2.27 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0137021, -0.0052661, -0.0131583, -0.0053365, -0.0068062, 0.0063625
1: -0.0068018, -0.0044234, -0.0066485, -0.0044432, -0.0019189, 0.0017938
2: -0.0116253, 0.0059232, -0.0104941, 0.0057769, -0.0141582, 0.0132353
3: 0.0000889, 0.0024111, 0.0002386, 0.0023918, -0.0018736, 0.0017515
4: 0.0016652, 0.0147799, 0.0017746, 0.0139345, -0.0098912, 0.0105809
5: 0.9959689, 0.9996125, 0.9959993, 0.9993777, -0.0027481, 0.0029397
6: 0.0042246, 0.0075319, 0.0042522, 0.0073187, -0.0024944, 0.0026684
7: -0.0076160, 0.0047264, -0.0075131, 0.0039308, -0.0093088, 0.0099579
8: -0.0128714, -0.0032653, -0.0122522, -0.0033454, -0.0077502, 0.0072450
9: -0.0037280, -0.0028993, -0.0037211, -0.0029527, -0.0006251, 0.0006687

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0020276
time: 2.15 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0020308
time: 2.08 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0135880, -0.0052939, -0.0129305, -0.0048721, -0.0072162, 0.0063361
1: -0.0067696, -0.0044312, -0.0065842, -0.0043123, -0.0020345, 0.0017864
2: -0.0113879, 0.0058654, -0.0100201, 0.0067428, -0.0150111, 0.0131804
3: 0.0001203, 0.0024035, 0.0003013, 0.0025196, -0.0019865, 0.0017442
4: 0.0017084, 0.0146025, 0.0010527, 0.0135803, -0.0098502, 0.0112184
5: 0.9959809, 0.9995632, 0.9957988, 0.9992793, -0.0027367, 0.0031168
6: 0.0042355, 0.0074872, 0.0040701, 0.0072294, -0.0024841, 0.0028291
7: -0.0075753, 0.0045594, -0.0081924, 0.0035974, -0.0092702, 0.0105577
8: -0.0127415, -0.0032970, -0.0119927, -0.0028167, -0.0082171, 0.0072150
9: -0.0037253, -0.0029105, -0.0037667, -0.0029751, -0.0006225, 0.0007089

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018961, upper bound: 0.0019827
time: 2.24 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018849, upper bound: 0.0019862
time: 2.34 seconds

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

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019421, upper bound: 0.0020338
time: 2.10 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019421, upper bound: 0.0020338
time: 2.37 seconds

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

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019114, upper bound: 0.0020097
time: 2.15 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019116, upper bound: 0.0019942
time: 2.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.24 seconds
NS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019369, upper bound: 0.0019261
NS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019369, upper bound: 0.0019261
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019369, upper bound: 0.0019261
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019369, upper bound: 0.0019261
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0020276
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0020308
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0018961, upper bound: 0.0019827
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0018849, upper bound: 0.0019862
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019421, upper bound: 0.0020338
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019421, upper bound: 0.0020338
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019114, upper bound: 0.0020097
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 5, lower bound: -0.0019116, upper bound: 0.0019942

## BFS NS instance: NS_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0131583, -0.0053365, -0.0131583, -0.0053365, -0.0063176, 0.0063176
1: -0.0066485, -0.0044432, -0.0066485, -0.0044432, -0.0017812, 0.0017812
2: -0.0104941, 0.0057769, -0.0104941, 0.0057769, -0.0131419, 0.0131419
3: 0.0002386, 0.0023918, 0.0002386, 0.0023918, -0.0017391, 0.0017391
4: 0.0017746, 0.0139345, 0.0017746, 0.0139345, -0.0098214, 0.0098214
5: 0.9959993, 0.9993777, 0.9959993, 0.9993777, -0.0027287, 0.0027287
6: 0.0042522, 0.0073187, 0.0042522, 0.0073187, -0.0024768, 0.0024768
7: -0.0075131, 0.0039308, -0.0075131, 0.0039308, -0.0092431, 0.0092431
8: -0.0122522, -0.0033454, -0.0122522, -0.0033454, -0.0071939, 0.0071939
9: -0.0037211, -0.0029527, -0.0037211, -0.0029527, -0.0006207, 0.0006207

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_A1_B1_B1_A1

### Relational analysis result of NS_A1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019029, upper bound: 0.0018863
time: 2.31 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2

### Relational analysis result of NS_A1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018907, upper bound: 0.0018878
time: 2.04 seconds

## BFS NS instance: NS_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0131583, -0.0053365, -0.0129305, -0.0048721, -0.0068404, 0.0061686
1: -0.0066485, -0.0044432, -0.0065842, -0.0043123, -0.0019286, 0.0017392
2: -0.0104941, 0.0057769, -0.0100201, 0.0067428, -0.0142293, 0.0128320
3: 0.0002386, 0.0023918, 0.0003013, 0.0025196, -0.0018830, 0.0016981
4: 0.0017746, 0.0139345, 0.0010527, 0.0135803, -0.0095898, 0.0106341
5: 0.9959993, 0.9993777, 0.9957988, 0.9992793, -0.0026643, 0.0029545
6: 0.0042522, 0.0073187, 0.0040701, 0.0072294, -0.0024184, 0.0026818
7: -0.0075131, 0.0039308, -0.0081924, 0.0035974, -0.0090251, 0.0100079
8: -0.0122522, -0.0033454, -0.0119927, -0.0028167, -0.0077892, 0.0070243
9: -0.0037211, -0.0029527, -0.0037667, -0.0029751, -0.0006060, 0.0006720

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_A1_B1_B2_A1

### Relational analysis result of NS_A1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019029, upper bound: 0.0018863
time: 2.39 seconds

## Relational analysis of NS_A1_A1_B1_B2_A2

### Relational analysis result of NS_A1_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018907, upper bound: 0.0018878
time: 2.41 seconds

## BFS NS instance: NS_A1_A1_B2_B1

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

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019800, upper bound: 0.0018914
time: 2.18 seconds

## Relational analysis of NS_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019843, upper bound: 0.0018812
time: 2.09 seconds

## BFS NS instance: NS_A1_A1_B2_B2

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

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019800, upper bound: 0.0018914
time: 2.46 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019843, upper bound: 0.0018812
time: 2.25 seconds

## BFS NS instance: NS_A2_B1_B1_A1

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

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018914, upper bound: 0.0019800
time: 1.94 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018812, upper bound: 0.0019843
time: 1.59 seconds

## BFS NS instance: NS_A2_B1_B1_A2

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

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018914, upper bound: 0.0019827
time: 2.19 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018812, upper bound: 0.0019863
time: 2.16 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0136257, -0.0054806, -0.0129268, -0.0049201, -0.0071307, 0.0061218
1: -0.0067802, -0.0044839, -0.0065832, -0.0043258, -0.0020104, 0.0017260
2: -0.0114664, 0.0054770, -0.0100126, 0.0066429, -0.0148333, 0.0127346
3: 0.0001099, 0.0023521, 0.0003023, 0.0025064, -0.0019629, 0.0016852
4: 0.0019987, 0.0146611, 0.0011274, 0.0135747, -0.0095170, 0.0110854
5: 0.9960616, 0.9995796, 0.9958194, 0.9992777, -0.0026441, 0.0030799
6: 0.0043087, 0.0075020, 0.0040890, 0.0072280, -0.0024001, 0.0027956
7: -0.0073022, 0.0046146, -0.0081222, 0.0035921, -0.0089566, 0.0104326
8: -0.0127844, -0.0035096, -0.0119886, -0.0028714, -0.0081197, 0.0069709
9: -0.0037069, -0.0029068, -0.0037620, -0.0029754, -0.0006014, 0.0007005

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018706, upper bound: 0.0019145
time: 2.08 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018694, upper bound: 0.0019616
time: 2.44 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0135783, -0.0054785, -0.0129281, -0.0049201, -0.0071590, 0.0060995
1: -0.0067669, -0.0044832, -0.0065836, -0.0043258, -0.0020184, 0.0017197
2: -0.0113677, 0.0054815, -0.0100151, 0.0066430, -0.0148923, 0.0126882
3: 0.0001230, 0.0023527, 0.0003020, 0.0025064, -0.0019708, 0.0016791
4: 0.0019953, 0.0145874, 0.0011273, 0.0135766, -0.0094823, 0.0111296
5: 0.9960606, 0.9995591, 0.9958195, 0.9992782, -0.0026345, 0.0030921
6: 0.0043079, 0.0074834, 0.0040890, 0.0072285, -0.0023913, 0.0028067
7: -0.0073053, 0.0045452, -0.0081222, 0.0035939, -0.0089239, 0.0104742
8: -0.0127304, -0.0035071, -0.0119900, -0.0028714, -0.0081521, 0.0069455
9: -0.0037072, -0.0029114, -0.0037620, -0.0029753, -0.0005992, 0.0007033

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018575, upper bound: 0.0019150
time: 2.18 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018645, upper bound: 0.0019659
time: 2.36 seconds

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

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019150, upper bound: 0.0019872
time: 2.30 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019051, upper bound: 0.0019914
time: 2.21 seconds

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

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019150, upper bound: 0.0019872
time: 2.33 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019051, upper bound: 0.0019914
time: 2.37 seconds

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

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018464, upper bound: 0.0019868
time: 2.43 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018906, upper bound: 0.0019900
time: 2.17 seconds

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

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018126, upper bound: 0.0019696
time: 2.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018913, upper bound: 0.0019745
time: 1.74 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.29 seconds
NS_A1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0019029, upper bound: 0.0018863
NS_A1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0018907, upper bound: 0.0018878
NS_A1_A1_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0019029, upper bound: 0.0018863
NS_A1_A1_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0018907, upper bound: 0.0018878
NS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0019800, upper bound: 0.0018914
NS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0019843, upper bound: 0.0018812
NS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0019800, upper bound: 0.0018914
NS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0019843, upper bound: 0.0018812
NS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0018914, upper bound: 0.0019800
NS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0018812, upper bound: 0.0019843
NS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0018914, upper bound: 0.0019827
NS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0018812, upper bound: 0.0019863
NS_A2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0018706, upper bound: 0.0019145
NS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0018694, upper bound: 0.0019616
NS_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0018575, upper bound: 0.0019150
NS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0018645, upper bound: 0.0019659
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0019150, upper bound: 0.0019872
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0019051, upper bound: 0.0019914
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0019150, upper bound: 0.0019872
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0019051, upper bound: 0.0019914
NS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0018464, upper bound: 0.0019868
NS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0018906, upper bound: 0.0019900
NS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0018126, upper bound: 0.0019696
NS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 5, lower bound: -0.0018913, upper bound: 0.0019745

## BFS NS instance: NS_A1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0131544, -0.0053831, -0.0136314, -0.0054815, -0.0061227, 0.0066162
1: -0.0066474, -0.0044564, -0.0067819, -0.0044841, -0.0017262, 0.0018654
2: -0.0104861, 0.0056799, -0.0114782, 0.0054752, -0.0127365, 0.0137631
3: 0.0002396, 0.0023789, 0.0001083, 0.0023519, -0.0016855, 0.0018213
4: 0.0018470, 0.0139285, 0.0020000, 0.0146699, -0.0102857, 0.0095185
5: 0.9960194, 0.9993760, 0.9960619, 0.9995819, -0.0028577, 0.0026445
6: 0.0042705, 0.0073172, 0.0043090, 0.0075042, -0.0025939, 0.0024004
7: -0.0074449, 0.0039251, -0.0073009, 0.0046229, -0.0096800, 0.0089579
8: -0.0122478, -0.0033985, -0.0127909, -0.0035106, -0.0069720, 0.0075339
9: -0.0037165, -0.0029531, -0.0037069, -0.0029062, -0.0006500, 0.0006015

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A1_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019457, upper bound: 0.0019050
time: 2.41 seconds

## Relational analysis of NS_A1_A1_B2_B1_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019866, upper bound: 0.0019060
time: 2.21 seconds

## BFS NS instance: NS_A1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0131557, -0.0053854, -0.0135744, -0.0054791, -0.0061037, 0.0066411
1: -0.0066477, -0.0044570, -0.0067658, -0.0044834, -0.0017209, 0.0018724
2: -0.0104886, 0.0056751, -0.0113598, 0.0054803, -0.0126970, 0.0138149
3: 0.0002393, 0.0023783, 0.0001240, 0.0023525, -0.0016802, 0.0018282
4: 0.0018506, 0.0139304, 0.0019963, 0.0145814, -0.0103244, 0.0094890
5: 0.9960204, 0.9993765, 0.9960608, 0.9995574, -0.0028684, 0.0026363
6: 0.0042714, 0.0073177, 0.0043081, 0.0074819, -0.0026037, 0.0023930
7: -0.0074415, 0.0039269, -0.0073044, 0.0045396, -0.0097164, 0.0089302
8: -0.0122492, -0.0034011, -0.0127261, -0.0035078, -0.0069504, 0.0075623
9: -0.0037163, -0.0029529, -0.0037071, -0.0029118, -0.0006524, 0.0005996

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A1_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019924, upper bound: 0.0018543
time: 1.70 seconds

## Relational analysis of NS_A1_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019928, upper bound: 0.0018962
time: 1.71 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0131544, -0.0053831, -0.0133904, -0.0050448, -0.0066136, 0.0064327
1: -0.0066474, -0.0044564, -0.0067139, -0.0043610, -0.0018646, 0.0018136
2: -0.0104861, 0.0056799, -0.0109770, 0.0063836, -0.0137576, 0.0133813
3: 0.0002396, 0.0023789, 0.0001747, 0.0024721, -0.0018206, 0.0017708
4: 0.0018470, 0.0139285, 0.0013212, 0.0142954, -0.0100004, 0.0102816
5: 0.9960194, 0.9993760, 0.9958733, 0.9994779, -0.0027784, 0.0028565
6: 0.0042705, 0.0073172, 0.0041379, 0.0074098, -0.0025219, 0.0025929
7: -0.0074449, 0.0039251, -0.0079398, 0.0042704, -0.0094115, 0.0096761
8: -0.0122478, -0.0033985, -0.0125165, -0.0030133, -0.0075310, 0.0073250
9: -0.0037165, -0.0029531, -0.0037498, -0.0029299, -0.0006320, 0.0006497

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A1_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019568, upper bound: 0.0018190
time: 2.27 seconds

## Relational analysis of NS_A1_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019616, upper bound: 0.0018718
time: 2.36 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0131557, -0.0053854, -0.0133452, -0.0050254, -0.0066000, 0.0064592
1: -0.0066477, -0.0044570, -0.0067012, -0.0043555, -0.0018608, 0.0018211
2: -0.0104886, 0.0056751, -0.0108829, 0.0064241, -0.0137293, 0.0134365
3: 0.0002393, 0.0023783, 0.0001871, 0.0024774, -0.0018169, 0.0017781
4: 0.0018506, 0.0139304, 0.0012909, 0.0142250, -0.0100416, 0.0102604
5: 0.9960204, 0.9993765, 0.9958649, 0.9994584, -0.0027899, 0.0028507
6: 0.0042714, 0.0073177, 0.0041302, 0.0073920, -0.0025323, 0.0025875
7: -0.0074415, 0.0039269, -0.0079683, 0.0042042, -0.0094503, 0.0096562
8: -0.0122492, -0.0034011, -0.0124650, -0.0029912, -0.0075154, 0.0073552
9: -0.0037163, -0.0029529, -0.0037517, -0.0029343, -0.0006346, 0.0006484

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019615, upper bound: 0.0018101
time: 2.19 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019659, upper bound: 0.0018614
time: 2.43 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1

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

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_B1_A1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019051, upper bound: 0.0019457
time: 1.70 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019060, upper bound: 0.0019866
time: 2.02 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2

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

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018543, upper bound: 0.0019923
time: 2.16 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018962, upper bound: 0.0019928
time: 2.21 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1

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

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018190, upper bound: 0.0019568
time: 1.60 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018719, upper bound: 0.0019616
time: 2.23 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2

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

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of NS_A2_B1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018101, upper bound: 0.0019615
time: 2.05 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018615, upper bound: 0.0019659
time: 1.98 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0135646, -0.0055053, -0.0129207, -0.0049225, -0.0069673, 0.0060872
1: -0.0067630, -0.0044908, -0.0065815, -0.0043265, -0.0019643, 0.0017162
2: -0.0113393, 0.0054257, -0.0099999, 0.0066381, -0.0144934, 0.0126626
3: 0.0001267, 0.0023453, 0.0003040, 0.0025057, -0.0019180, 0.0016757
4: 0.0020371, 0.0145661, 0.0011310, 0.0135652, -0.0094633, 0.0108314
5: 0.9960722, 0.9995531, 0.9958204, 0.9992751, -0.0026292, 0.0030093
6: 0.0043184, 0.0074780, 0.0040899, 0.0072256, -0.0023865, 0.0027315
7: -0.0072660, 0.0045252, -0.0081188, 0.0035832, -0.0089060, 0.0101936
8: -0.0127148, -0.0035377, -0.0119817, -0.0028740, -0.0079337, 0.0069315
9: -0.0037045, -0.0029128, -0.0037618, -0.0029760, -0.0005980, 0.0006845

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B2_A1_A2_A1

### Relational analysis result of NS_A2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018694, upper bound: 0.0019570
time: 2.29 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018694, upper bound: 0.0019616
time: 1.58 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0135143, -0.0055075, -0.0129220, -0.0049225, -0.0070197, 0.0060592
1: -0.0067488, -0.0044914, -0.0065818, -0.0043265, -0.0019791, 0.0017083
2: -0.0112346, 0.0054212, -0.0100025, 0.0066381, -0.0146023, 0.0126043
3: 0.0001406, 0.0023447, 0.0003036, 0.0025057, -0.0019324, 0.0016680
4: 0.0020404, 0.0144879, 0.0011310, 0.0135671, -0.0094197, 0.0109129
5: 0.9960731, 0.9995314, 0.9958205, 0.9992756, -0.0026171, 0.0030319
6: 0.0043192, 0.0074583, 0.0040899, 0.0072261, -0.0023755, 0.0027521
7: -0.0072629, 0.0044516, -0.0081188, 0.0035850, -0.0088649, 0.0102702
8: -0.0126575, -0.0035401, -0.0119831, -0.0028740, -0.0079933, 0.0068996
9: -0.0037043, -0.0029177, -0.0037618, -0.0029759, -0.0005953, 0.0006896

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B2_A2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018589, upper bound: 0.0019626
time: 2.16 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018589, upper bound: 0.0019659
time: 2.24 seconds

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
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A1_B1_A1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019153, upper bound: 0.0019490
time: 1.69 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019161, upper bound: 0.0019902
time: 2.32 seconds

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

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A1_B1_A2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019059, upper bound: 0.0019501
time: 2.18 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019070, upper bound: 0.0019963
time: 2.25 seconds

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

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A1_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018970, upper bound: 0.0019229
time: 2.47 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019018, upper bound: 0.0019668
time: 2.44 seconds

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

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of NS_A2_B2_A1_B2_A2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018858, upper bound: 0.0019238
time: 2.30 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018913, upper bound: 0.0019721
time: 2.15 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0132573, -0.0049215, -0.0133351, -0.0055339, -0.0060638, 0.0065249
1: -0.0066764, -0.0043262, -0.0066983, -0.0044989, -0.0017096, 0.0018396
2: -0.0106999, 0.0066402, -0.0108619, 0.0053662, -0.0126140, 0.0135730
3: 0.0002113, 0.0025060, 0.0001899, 0.0023374, -0.0016693, 0.0017962
4: 0.0011294, 0.0140883, 0.0020815, 0.0142094, -0.0101436, 0.0094269
5: 0.9958200, 0.9994205, 0.9960846, 0.9994540, -0.0028182, 0.0026191
6: 0.0040895, 0.0073575, 0.0043296, 0.0073881, -0.0025581, 0.0023773
7: -0.0081203, 0.0040756, -0.0072242, 0.0041895, -0.0095463, 0.0088718
8: -0.0123649, -0.0028729, -0.0124536, -0.0035702, -0.0069049, 0.0074299
9: -0.0037619, -0.0029430, -0.0037017, -0.0029353, -0.0006410, 0.0005957

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A2_B1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018124, upper bound: 0.0019595
time: 2.10 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018124, upper bound: 0.0019501
time: 1.92 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0133446, -0.0048969, -0.0135646, -0.0055053, -0.0061679, 0.0066772
1: -0.0067010, -0.0043193, -0.0067630, -0.0044908, -0.0017390, 0.0018825
2: -0.0108815, 0.0066914, -0.0113393, 0.0054257, -0.0128306, 0.0138899
3: 0.0001873, 0.0025128, 0.0001267, 0.0023453, -0.0016979, 0.0018381
4: 0.0010911, 0.0142241, 0.0020371, 0.0145661, -0.0103804, 0.0095888
5: 0.9958094, 0.9994581, 0.9960722, 0.9995531, -0.0028840, 0.0026640
6: 0.0040798, 0.0073918, 0.0043184, 0.0074780, -0.0026178, 0.0024181
7: -0.0081562, 0.0042033, -0.0072660, 0.0045252, -0.0097691, 0.0090241
8: -0.0124643, -0.0028449, -0.0127148, -0.0035377, -0.0070235, 0.0076033
9: -0.0037643, -0.0029344, -0.0037045, -0.0029128, -0.0006560, 0.0006060

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_B2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018826, upper bound: 0.0019901
time: 2.49 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018826, upper bound: 0.0019900
time: 2.40 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0132586, -0.0049235, -0.0132980, -0.0055252, -0.0060285, 0.0065983
1: -0.0066768, -0.0043268, -0.0066879, -0.0044964, -0.0016997, 0.0018603
2: -0.0107028, 0.0066360, -0.0107848, 0.0053844, -0.0125406, 0.0137258
3: 0.0002109, 0.0025055, 0.0002001, 0.0023398, -0.0016595, 0.0018164
4: 0.0011325, 0.0140905, 0.0020679, 0.0141518, -0.0102578, 0.0093720
5: 0.9958209, 0.9994210, 0.9960808, 0.9994380, -0.0028499, 0.0026038
6: 0.0040903, 0.0073581, 0.0043262, 0.0073735, -0.0025869, 0.0023635
7: -0.0081173, 0.0040776, -0.0072370, 0.0041352, -0.0096538, 0.0088201
8: -0.0123665, -0.0028751, -0.0124113, -0.0035603, -0.0068647, 0.0075135
9: -0.0037617, -0.0029428, -0.0037026, -0.0029389, -0.0006482, 0.0005923

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018126, upper bound: 0.0019428
time: 2.28 seconds

## Relational analysis of NS_A2_B2_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018126, upper bound: 0.0019333
time: 1.86 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0133459, -0.0048987, -0.0135143, -0.0055075, -0.0061427, 0.0067550
1: -0.0067014, -0.0043198, -0.0067488, -0.0044914, -0.0017319, 0.0019045
2: -0.0108844, 0.0066876, -0.0112346, 0.0054212, -0.0127782, 0.0140518
3: 0.0001869, 0.0025123, 0.0001406, 0.0023447, -0.0016910, 0.0018595
4: 0.0010940, 0.0142262, 0.0020404, 0.0144879, -0.0105014, 0.0095496
5: 0.9958102, 0.9994587, 0.9960731, 0.9995314, -0.0029176, 0.0026532
6: 0.0040806, 0.0073923, 0.0043192, 0.0074583, -0.0026483, 0.0024083
7: -0.0081536, 0.0042053, -0.0072629, 0.0044516, -0.0098830, 0.0089872
8: -0.0124659, -0.0028469, -0.0126575, -0.0035401, -0.0069948, 0.0076920
9: -0.0037641, -0.0029342, -0.0037043, -0.0029177, -0.0006636, 0.0006035

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018836, upper bound: 0.0019744
time: 2.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018836, upper bound: 0.0019744
time: 2.33 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.65 seconds
NS_A1_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0019457, upper bound: 0.0019050
NS_A1_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0019866, upper bound: 0.0019060
NS_A1_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0019924, upper bound: 0.0018543
NS_A1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0019928, upper bound: 0.0018962
NS_A1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0019568, upper bound: 0.0018190
NS_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0019616, upper bound: 0.0018718
NS_A1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0019615, upper bound: 0.0018101
NS_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0019659, upper bound: 0.0018614
NS_A2_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0019051, upper bound: 0.0019457
NS_A2_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0019060, upper bound: 0.0019866
NS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018543, upper bound: 0.0019923
NS_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018962, upper bound: 0.0019928
NS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018190, upper bound: 0.0019568
NS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018719, upper bound: 0.0019616
NS_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018101, upper bound: 0.0019615
NS_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018615, upper bound: 0.0019659
NS_A2_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018694, upper bound: 0.0019570
NS_A2_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018694, upper bound: 0.0019616
NS_A2_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018589, upper bound: 0.0019626
NS_A2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018589, upper bound: 0.0019659
NS_A2_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0019153, upper bound: 0.0019490
NS_A2_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0019161, upper bound: 0.0019902
NS_A2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0019059, upper bound: 0.0019501
NS_A2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0019070, upper bound: 0.0019963
NS_A2_B2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018970, upper bound: 0.0019229
NS_A2_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0019018, upper bound: 0.0019668
NS_A2_B2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018858, upper bound: 0.0019238
NS_A2_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018913, upper bound: 0.0019721
NS_A2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018124, upper bound: 0.0019595
NS_A2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018124, upper bound: 0.0019501
NS_A2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018826, upper bound: 0.0019901
NS_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018826, upper bound: 0.0019900
NS_A2_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018126, upper bound: 0.0019428
NS_A2_B2_A2_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018126, upper bound: 0.0019333
NS_A2_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018836, upper bound: 0.0019744
NS_A2_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.65
Output dim: 5, lower bound: -0.0018836, upper bound: 0.0019744

## BFS NS instance: NS_A1_A1_B2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0130558, -0.0054085, -0.0133378, -0.0055343, -0.0059925, 0.0062976
1: -0.0066196, -0.0044635, -0.0066991, -0.0044990, -0.0016895, 0.0017755
2: -0.0102808, 0.0056271, -0.0108675, 0.0053653, -0.0124656, 0.0131002
3: 0.0002668, 0.0023720, 0.0001892, 0.0023373, -0.0016496, 0.0017336
4: 0.0018865, 0.0137751, 0.0020822, 0.0142136, -0.0097903, 0.0093160
5: 0.9960304, 0.9993334, 0.9960847, 0.9994552, -0.0027200, 0.0025883
6: 0.0042804, 0.0072785, 0.0043298, 0.0073891, -0.0024690, 0.0023494
7: -0.0074077, 0.0037807, -0.0072236, 0.0041934, -0.0092138, 0.0087674
8: -0.0121354, -0.0034274, -0.0124566, -0.0035707, -0.0068237, 0.0071711
9: -0.0037140, -0.0029627, -0.0037017, -0.0029350, -0.0006187, 0.0005887

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_A1_B2_B1_B1_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018166, upper bound: 0.0018666
time: 2.70 seconds

## Relational analysis of NS_A1_A1_B2_B1_B1_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019113, upper bound: 0.0018670
time: 2.51 seconds

## BFS NS instance: NS_A1_A1_B2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0131487, -0.0053854, -0.0135725, -0.0055061, -0.0060886, 0.0064269
1: -0.0066458, -0.0044570, -0.0067653, -0.0044910, -0.0017166, 0.0018120
2: -0.0104741, 0.0056751, -0.0113557, 0.0054241, -0.0126655, 0.0133693
3: 0.0002412, 0.0023783, 0.0001245, 0.0023451, -0.0016761, 0.0017692
4: 0.0018506, 0.0139196, 0.0020383, 0.0145784, -0.0099914, 0.0094654
5: 0.9960204, 0.9993734, 0.9960725, 0.9995565, -0.0027759, 0.0026298
6: 0.0042714, 0.0073150, 0.0043187, 0.0074811, -0.0025197, 0.0023870
7: -0.0074415, 0.0039167, -0.0072649, 0.0045368, -0.0094030, 0.0089080
8: -0.0122413, -0.0034011, -0.0127239, -0.0035386, -0.0069331, 0.0073184
9: -0.0037163, -0.0029536, -0.0037044, -0.0029120, -0.0006314, 0.0005982

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_A1_B2_B1_B1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019534, upper bound: 0.0018728
time: 2.20 seconds

## Relational analysis of NS_A1_A1_B2_B1_B1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019534, upper bound: 0.0018722
time: 2.50 seconds

## BFS NS instance: NS_A1_A1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0128665, -0.0054356, -0.0134780, -0.0055065, -0.0057819, 0.0065066
1: -0.0065662, -0.0044712, -0.0067386, -0.0044912, -0.0016301, 0.0018345
2: -0.0098871, 0.0055706, -0.0111590, 0.0054231, -0.0120276, 0.0135350
3: 0.0003189, 0.0023645, 0.0001506, 0.0023450, -0.0015917, 0.0017911
4: 0.0019287, 0.0134809, 0.0020390, 0.0144314, -0.0101152, 0.0089887
5: 0.9960421, 0.9992516, 0.9960727, 0.9995157, -0.0028103, 0.0024973
6: 0.0042911, 0.0072043, 0.0043189, 0.0074441, -0.0025509, 0.0022668
7: -0.0073680, 0.0035039, -0.0072643, 0.0043985, -0.0095196, 0.0084593
8: -0.0119199, -0.0034583, -0.0126162, -0.0035391, -0.0065839, 0.0074091
9: -0.0037114, -0.0029813, -0.0037044, -0.0029213, -0.0006392, 0.0005680

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_A1_B2_B1_B2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019652, upper bound: 0.0018125
time: 2.15 seconds

## Relational analysis of NS_A1_A1_B2_B1_B2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019570, upper bound: 0.0018125
time: 2.00 seconds

## BFS NS instance: NS_A1_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0130917, -0.0054114, -0.0135688, -0.0054816, -0.0059111, 0.0066038
1: -0.0066297, -0.0044643, -0.0067642, -0.0044841, -0.0016666, 0.0018619
2: -0.0103555, 0.0056211, -0.0113480, 0.0054749, -0.0122963, 0.0137372
3: 0.0002569, 0.0023712, 0.0001256, 0.0023518, -0.0016272, 0.0018179
4: 0.0018910, 0.0138309, 0.0020003, 0.0145727, -0.0102664, 0.0091895
5: 0.9960316, 0.9993489, 0.9960620, 0.9995549, -0.0028523, 0.0025531
6: 0.0042816, 0.0072926, 0.0043091, 0.0074797, -0.0025890, 0.0023174
7: -0.0074035, 0.0038333, -0.0073007, 0.0045314, -0.0096618, 0.0086483
8: -0.0121763, -0.0034307, -0.0127196, -0.0035107, -0.0067310, 0.0075198
9: -0.0037137, -0.0029592, -0.0037068, -0.0029123, -0.0006488, 0.0005807

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_A1_B2_B1_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019601, upper bound: 0.0018626
time: 2.57 seconds

## Relational analysis of NS_A1_A1_B2_B1_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019601, upper bound: 0.0018622
time: 2.36 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0128650, -0.0054345, -0.0132943, -0.0050718, -0.0063127, 0.0062891
1: -0.0065658, -0.0044709, -0.0066868, -0.0043686, -0.0017798, 0.0017731
2: -0.0098840, 0.0055729, -0.0107770, 0.0063276, -0.0131317, 0.0130826
3: 0.0003193, 0.0023648, 0.0002011, 0.0024646, -0.0017378, 0.0017313
4: 0.0019270, 0.0134785, 0.0013630, 0.0141459, -0.0097771, 0.0098138
5: 0.9960416, 0.9992509, 0.9958850, 0.9994364, -0.0027164, 0.0027266
6: 0.0042906, 0.0072038, 0.0041484, 0.0073721, -0.0024657, 0.0024749
7: -0.0073696, 0.0035017, -0.0079004, 0.0041298, -0.0092014, 0.0092359
8: -0.0119182, -0.0034571, -0.0124071, -0.0030440, -0.0071883, 0.0071614
9: -0.0037115, -0.0029815, -0.0037471, -0.0029393, -0.0006179, 0.0006202

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_A1_B2_B2_B1_A1_A1

### Relational analysis result of NS_A1_A1_B2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019276, upper bound: 0.0017696
time: 2.21 seconds

## Relational analysis of NS_A1_A1_B2_B2_B1_A1_A2

### Relational analysis result of NS_A1_A1_B2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019159, upper bound: 0.0017677
time: 2.04 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0130904, -0.0054089, -0.0133849, -0.0050473, -0.0064614, 0.0063953
1: -0.0066293, -0.0044636, -0.0067123, -0.0043617, -0.0018217, 0.0018031
2: -0.0103529, 0.0056262, -0.0109654, 0.0063783, -0.0134410, 0.0133035
3: 0.0002573, 0.0023718, 0.0001762, 0.0024714, -0.0017787, 0.0017605
4: 0.0018872, 0.0138290, 0.0013251, 0.0142867, -0.0099422, 0.0100449
5: 0.9960306, 0.9993483, 0.9958743, 0.9994755, -0.0027622, 0.0027908
6: 0.0042806, 0.0072921, 0.0041388, 0.0074076, -0.0025073, 0.0025332
7: -0.0074071, 0.0038315, -0.0079361, 0.0042622, -0.0093567, 0.0094534
8: -0.0121749, -0.0034279, -0.0125102, -0.0030162, -0.0073576, 0.0072823
9: -0.0037140, -0.0029593, -0.0037495, -0.0029304, -0.0006283, 0.0006348

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_A1_B2_B2_B1_A2_A1

### Relational analysis result of NS_A1_A1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019367, upper bound: 0.0018354
time: 1.81 seconds

## Relational analysis of NS_A1_A1_B2_B2_B1_A2_A2

### Relational analysis result of NS_A1_A1_B2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019260, upper bound: 0.0018353
time: 2.31 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0128665, -0.0054356, -0.0132519, -0.0050532, -0.0062781, 0.0063280
1: -0.0065662, -0.0044712, -0.0066749, -0.0043634, -0.0017700, 0.0017841
2: -0.0098871, 0.0055706, -0.0106887, 0.0063661, -0.0130598, 0.0131635
3: 0.0003189, 0.0023645, 0.0002128, 0.0024697, -0.0017283, 0.0017420
4: 0.0019287, 0.0134809, 0.0013343, 0.0140800, -0.0098375, 0.0097601
5: 0.9960421, 0.9992516, 0.9958770, 0.9994181, -0.0027332, 0.0027116
6: 0.0042911, 0.0072043, 0.0041411, 0.0073554, -0.0024809, 0.0024613
7: -0.0073680, 0.0035039, -0.0079275, 0.0040677, -0.0092582, 0.0091853
8: -0.0119199, -0.0034583, -0.0123587, -0.0030229, -0.0071489, 0.0072057
9: -0.0037114, -0.0029813, -0.0037489, -0.0029435, -0.0006217, 0.0006168

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_A1_B2_B2_B2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019322, upper bound: 0.0017600
time: 1.88 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019212, upper bound: 0.0017583
time: 1.85 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0130917, -0.0054114, -0.0133392, -0.0050280, -0.0064213, 0.0064209
1: -0.0066297, -0.0044643, -0.0066995, -0.0043562, -0.0018104, 0.0018103
2: -0.0103555, 0.0056211, -0.0108705, 0.0064187, -0.0133577, 0.0133569
3: 0.0002569, 0.0023712, 0.0001888, 0.0024767, -0.0017677, 0.0017676
4: 0.0018910, 0.0138309, 0.0012949, 0.0142158, -0.0099821, 0.0099827
5: 0.9960316, 0.9993489, 0.9958660, 0.9994557, -0.0027733, 0.0027735
6: 0.0042816, 0.0072926, 0.0041312, 0.0073897, -0.0025173, 0.0025175
7: -0.0074035, 0.0038333, -0.0079645, 0.0041955, -0.0093943, 0.0093948
8: -0.0121763, -0.0034307, -0.0124582, -0.0029941, -0.0073120, 0.0073116
9: -0.0037137, -0.0029592, -0.0037514, -0.0029349, -0.0006308, 0.0006308

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_A1_B2_B2_B2_A2_A1

### Relational analysis result of NS_A1_A1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019409, upper bound: 0.0018242
time: 2.38 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2_A2_A2

### Relational analysis result of NS_A1_A1_B2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019305, upper bound: 0.0018239
time: 1.83 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0133378, -0.0055343, -0.0130558, -0.0054085, -0.0062976, 0.0059925
1: -0.0066991, -0.0044990, -0.0066196, -0.0044635, -0.0017755, 0.0016895
2: -0.0108675, 0.0053653, -0.0102808, 0.0056271, -0.0131002, 0.0124656
3: 0.0001892, 0.0023373, 0.0002668, 0.0023720, -0.0017336, 0.0016496
4: 0.0020822, 0.0142136, 0.0018865, 0.0137751, -0.0093160, 0.0097903
5: 0.9960847, 0.9994552, 0.9960304, 0.9993334, -0.0025883, 0.0027200
6: 0.0043298, 0.0073891, 0.0042804, 0.0072785, -0.0023494, 0.0024690
7: -0.0072236, 0.0041934, -0.0074077, 0.0037807, -0.0087674, 0.0092138
8: -0.0124566, -0.0035707, -0.0121354, -0.0034274, -0.0071711, 0.0068237
9: -0.0037017, -0.0029350, -0.0037140, -0.0029627, -0.0005887, 0.0006187

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B1_A1_A1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018667, upper bound: 0.0019113
time: 1.72 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018670, upper bound: 0.0019113
time: 2.36 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0135725, -0.0055061, -0.0131487, -0.0053854, -0.0064269, 0.0060886
1: -0.0067653, -0.0044910, -0.0066458, -0.0044570, -0.0018120, 0.0017166
2: -0.0113557, 0.0054241, -0.0104741, 0.0056751, -0.0133693, 0.0126655
3: 0.0001245, 0.0023451, 0.0002412, 0.0023783, -0.0017692, 0.0016761
4: 0.0020383, 0.0145784, 0.0018506, 0.0139196, -0.0094654, 0.0099914
5: 0.9960725, 0.9995565, 0.9960204, 0.9993734, -0.0026298, 0.0027759
6: 0.0043187, 0.0074811, 0.0042714, 0.0073150, -0.0023870, 0.0025197
7: -0.0072649, 0.0045368, -0.0074415, 0.0039167, -0.0089080, 0.0094030
8: -0.0127239, -0.0035386, -0.0122413, -0.0034011, -0.0073184, 0.0069331
9: -0.0037044, -0.0029120, -0.0037163, -0.0029536, -0.0005982, 0.0006314

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B1_A1_A1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018727, upper bound: 0.0019534
time: 1.67 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018722, upper bound: 0.0019534
time: 1.76 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_B1

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

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018125, upper bound: 0.0019652
time: 2.23 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018125, upper bound: 0.0019570
time: 1.66 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_B2

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

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018114, upper bound: 0.0019601
time: 2.57 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018622, upper bound: 0.0019601
time: 2.28 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_B1

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

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017696, upper bound: 0.0019276
time: 2.18 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017677, upper bound: 0.0019159
time: 2.31 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_B2

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

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018354, upper bound: 0.0019367
time: 2.10 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_B2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018352, upper bound: 0.0019260
time: 1.89 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_B1

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

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017600, upper bound: 0.0019322
time: 2.15 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017583, upper bound: 0.0019212
time: 2.38 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_B2

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

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018242, upper bound: 0.0019409
time: 2.25 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B2_B2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018239, upper bound: 0.0019305
time: 1.75 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0135725, -0.0055061, -0.0129207, -0.0049225, -0.0069750, 0.0059387
1: -0.0067653, -0.0044910, -0.0065815, -0.0043265, -0.0019665, 0.0016744
2: -0.0113557, 0.0054241, -0.0099999, 0.0066381, -0.0145094, 0.0123538
3: 0.0001245, 0.0023451, 0.0003040, 0.0025057, -0.0019201, 0.0016348
4: 0.0020383, 0.0145784, 0.0011310, 0.0135652, -0.0092324, 0.0108434
5: 0.9960725, 0.9995565, 0.9958204, 0.9992751, -0.0025650, 0.0030126
6: 0.0043187, 0.0074811, 0.0040899, 0.0072256, -0.0023283, 0.0027346
7: -0.0072649, 0.0045368, -0.0081188, 0.0035832, -0.0086888, 0.0102049
8: -0.0127239, -0.0035386, -0.0119817, -0.0028740, -0.0079425, 0.0067625
9: -0.0037044, -0.0029120, -0.0037618, -0.0029760, -0.0005834, 0.0006852

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B2_A1_A2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018339, upper bound: 0.0019233
time: 2.18 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018315, upper bound: 0.0019233
time: 2.44 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0133281, -0.0050721, -0.0129207, -0.0049225, -0.0065509, 0.0062082
1: -0.0066963, -0.0043687, -0.0065815, -0.0043265, -0.0018469, 0.0017503
2: -0.0108473, 0.0063268, -0.0099999, 0.0066381, -0.0136271, 0.0129143
3: 0.0001918, 0.0024645, 0.0003040, 0.0025057, -0.0018033, 0.0017090
4: 0.0013636, 0.0141985, 0.0011310, 0.0135652, -0.0096514, 0.0101841
5: 0.9958851, 0.9994510, 0.9958204, 0.9992751, -0.0026814, 0.0028294
6: 0.0041486, 0.0073853, 0.0040899, 0.0072256, -0.0024339, 0.0025683
7: -0.0078998, 0.0041792, -0.0081188, 0.0035832, -0.0090830, 0.0095843
8: -0.0124455, -0.0030444, -0.0119817, -0.0028740, -0.0074595, 0.0070693
9: -0.0037471, -0.0029360, -0.0037618, -0.0029760, -0.0006099, 0.0006436

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B2_A1_A2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018339, upper bound: 0.0019259
time: 1.65 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018315, upper bound: 0.0019259
time: 2.14 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0135116, -0.0055080, -0.0129220, -0.0049225, -0.0070177, 0.0059190
1: -0.0067481, -0.0044916, -0.0065818, -0.0043265, -0.0019785, 0.0016688
2: -0.0112291, 0.0054201, -0.0100025, 0.0066381, -0.0145982, 0.0123126
3: 0.0001413, 0.0023446, 0.0003036, 0.0025057, -0.0019318, 0.0016294
4: 0.0020413, 0.0144838, 0.0011310, 0.0135671, -0.0092017, 0.0109098
5: 0.9960733, 0.9995303, 0.9958205, 0.9992756, -0.0025565, 0.0030311
6: 0.0043194, 0.0074573, 0.0040899, 0.0072261, -0.0023205, 0.0027513
7: -0.0072621, 0.0044477, -0.0081188, 0.0035850, -0.0086598, 0.0102673
8: -0.0126545, -0.0035408, -0.0119831, -0.0028740, -0.0079911, 0.0067400
9: -0.0037043, -0.0029180, -0.0037618, -0.0029759, -0.0005815, 0.0006894

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B2_A2_A2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018228, upper bound: 0.0019285
time: 2.62 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018201, upper bound: 0.0019285
time: 2.54 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0132797, -0.0050548, -0.0129220, -0.0049225, -0.0066041, 0.0061921
1: -0.0066827, -0.0043638, -0.0065818, -0.0043265, -0.0018619, 0.0017458
2: -0.0107467, 0.0063628, -0.0100025, 0.0066381, -0.0137379, 0.0128808
3: 0.0002051, 0.0024693, 0.0003036, 0.0025057, -0.0018180, 0.0017046
4: 0.0013367, 0.0141233, 0.0011310, 0.0135671, -0.0096263, 0.0102668
5: 0.9958776, 0.9994301, 0.9958205, 0.9992756, -0.0026745, 0.0028524
6: 0.0041418, 0.0073664, 0.0040899, 0.0072261, -0.0024276, 0.0025891
7: -0.0079251, 0.0041084, -0.0081188, 0.0035850, -0.0090594, 0.0096622
8: -0.0123905, -0.0030247, -0.0119831, -0.0028740, -0.0075201, 0.0070510
9: -0.0037488, -0.0029407, -0.0037618, -0.0029759, -0.0006083, 0.0006488

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018228, upper bound: 0.0019305
time: 1.92 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018201, upper bound: 0.0019305
time: 2.29 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0133378, -0.0055343, -0.0134836, -0.0053665, -0.0059904, 0.0060741
1: -0.0066991, -0.0044990, -0.0067402, -0.0044517, -0.0016889, 0.0017125
2: -0.0108675, 0.0053653, -0.0111709, 0.0057145, -0.0124613, 0.0126354
3: 0.0001892, 0.0023373, 0.0001490, 0.0023835, -0.0016491, 0.0016721
4: 0.0020822, 0.0142136, 0.0018212, 0.0144403, -0.0094429, 0.0093128
5: 0.9960847, 0.9994552, 0.9960122, 0.9995182, -0.0026235, 0.0025874
6: 0.0043298, 0.0073891, 0.0042640, 0.0074463, -0.0023814, 0.0023486
7: -0.0072236, 0.0041934, -0.0074692, 0.0044068, -0.0088868, 0.0087644
8: -0.0124566, -0.0035707, -0.0126227, -0.0033796, -0.0068213, 0.0069166
9: -0.0037017, -0.0029350, -0.0037182, -0.0029207, -0.0005967, 0.0005885

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_B1_A1_A1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018846, upper bound: 0.0019176
time: 1.73 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_A1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018856, upper bound: 0.0019176
time: 1.87 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0135725, -0.0055061, -0.0135745, -0.0053422, -0.0061257, 0.0061792
1: -0.0067653, -0.0044910, -0.0067658, -0.0044448, -0.0017271, 0.0017421
2: -0.0113557, 0.0054241, -0.0113598, 0.0057649, -0.0127427, 0.0128539
3: 0.0001245, 0.0023451, 0.0001240, 0.0023902, -0.0016863, 0.0017010
4: 0.0020383, 0.0145784, 0.0017835, 0.0145815, -0.0096062, 0.0095231
5: 0.9960725, 0.9995565, 0.9960018, 0.9995574, -0.0026689, 0.0026458
6: 0.0043187, 0.0074811, 0.0042544, 0.0074819, -0.0024225, 0.0024016
7: -0.0072649, 0.0045368, -0.0075046, 0.0045397, -0.0090405, 0.0089623
8: -0.0127239, -0.0035386, -0.0127261, -0.0033520, -0.0069754, 0.0070363
9: -0.0037044, -0.0029120, -0.0037205, -0.0029118, -0.0006071, 0.0006018

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_B1_A1_A2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018888, upper bound: 0.0019598
time: 2.48 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_A2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018889, upper bound: 0.0019598
time: 2.01 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0132905, -0.0055254, -0.0134851, -0.0053724, -0.0060654, 0.0060424
1: -0.0066857, -0.0044965, -0.0067406, -0.0044534, -0.0017101, 0.0017036
2: -0.0107691, 0.0053838, -0.0111739, 0.0057021, -0.0126172, 0.0125694
3: 0.0002022, 0.0023398, 0.0001486, 0.0023819, -0.0016697, 0.0016634
4: 0.0020683, 0.0141400, 0.0018305, 0.0144425, -0.0093936, 0.0094293
5: 0.9960809, 0.9994348, 0.9960148, 0.9995188, -0.0026098, 0.0026197
6: 0.0043263, 0.0073706, 0.0042663, 0.0074469, -0.0023689, 0.0023779
7: -0.0072366, 0.0041242, -0.0074604, 0.0044089, -0.0088404, 0.0088740
8: -0.0124027, -0.0035606, -0.0126243, -0.0033864, -0.0069067, 0.0068805
9: -0.0037025, -0.0029397, -0.0037176, -0.0029206, -0.0005936, 0.0005959

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_B1_A2_A1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018744, upper bound: 0.0019192
time: 2.38 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_A1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018755, upper bound: 0.0019192
time: 1.97 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0135116, -0.0055080, -0.0135759, -0.0053481, -0.0062041, 0.0061541
1: -0.0067481, -0.0044916, -0.0067662, -0.0044465, -0.0017492, 0.0017351
2: -0.0112291, 0.0054201, -0.0113628, 0.0057526, -0.0129058, 0.0128018
3: 0.0001413, 0.0023446, 0.0001236, 0.0023886, -0.0017079, 0.0016941
4: 0.0020413, 0.0144838, 0.0017927, 0.0145837, -0.0095673, 0.0096450
5: 0.9960733, 0.9995303, 0.9960043, 0.9995580, -0.0026581, 0.0026797
6: 0.0043194, 0.0074573, 0.0042568, 0.0074825, -0.0024127, 0.0024323
7: -0.0072621, 0.0044477, -0.0074960, 0.0045418, -0.0090039, 0.0090770
8: -0.0126545, -0.0035408, -0.0127277, -0.0033587, -0.0070646, 0.0070077
9: -0.0037043, -0.0029180, -0.0037200, -0.0029116, -0.0006046, 0.0006095

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_B1_A2_A2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018793, upper bound: 0.0019659
time: 2.32 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_A2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018796, upper bound: 0.0019659
time: 1.86 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0135725, -0.0055061, -0.0133446, -0.0048969, -0.0066655, 0.0060233
1: -0.0067653, -0.0044910, -0.0067010, -0.0043193, -0.0018792, 0.0016982
2: -0.0113557, 0.0054241, -0.0108815, 0.0066914, -0.0138655, 0.0125297
3: 0.0001245, 0.0023451, 0.0001873, 0.0025128, -0.0018349, 0.0016581
4: 0.0020383, 0.0145784, 0.0010911, 0.0142241, -0.0093639, 0.0103622
5: 0.9960725, 0.9995565, 0.9958094, 0.9994581, -0.0026016, 0.0028789
6: 0.0043187, 0.0074811, 0.0040798, 0.0073918, -0.0023614, 0.0026132
7: -0.0072649, 0.0045368, -0.0081562, 0.0042033, -0.0088125, 0.0097520
8: -0.0127239, -0.0035386, -0.0124643, -0.0028449, -0.0075900, 0.0068588
9: -0.0037044, -0.0029120, -0.0037643, -0.0029344, -0.0005917, 0.0006548

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_B2_A1_A2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018750, upper bound: 0.0019372
time: 2.28 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_A2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018738, upper bound: 0.0019372
time: 2.36 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0135116, -0.0055080, -0.0133459, -0.0048987, -0.0067476, 0.0060058
1: -0.0067481, -0.0044916, -0.0067014, -0.0043198, -0.0019024, 0.0016932
2: -0.0112291, 0.0054201, -0.0108844, 0.0066876, -0.0140363, 0.0124932
3: 0.0001413, 0.0023446, 0.0001869, 0.0025123, -0.0018575, 0.0016533
4: 0.0020413, 0.0144838, 0.0010940, 0.0142262, -0.0093366, 0.0104899
5: 0.9960733, 0.9995303, 0.9958102, 0.9994587, -0.0025940, 0.0029144
6: 0.0043194, 0.0074573, 0.0040806, 0.0073923, -0.0023546, 0.0026454
7: -0.0072621, 0.0044477, -0.0081536, 0.0042053, -0.0087868, 0.0098721
8: -0.0126545, -0.0035408, -0.0124659, -0.0028469, -0.0076835, 0.0068388
9: -0.0037043, -0.0029180, -0.0037641, -0.0029342, -0.0005900, 0.0006629

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_B2_A2_A2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018646, upper bound: 0.0019428
time: 2.19 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_A2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018638, upper bound: 0.0019427
time: 2.56 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0132557, -0.0049369, -0.0133254, -0.0056603, -0.0058867, 0.0064500
1: -0.0066759, -0.0043306, -0.0066956, -0.0045345, -0.0016597, 0.0018185
2: -0.0106966, 0.0066080, -0.0108417, 0.0051033, -0.0122455, 0.0134173
3: 0.0002118, 0.0025018, 0.0001926, 0.0023026, -0.0016205, 0.0017756
4: 0.0011534, 0.0140859, 0.0022780, 0.0141943, -0.0100273, 0.0091515
5: 0.9958267, 0.9994197, 0.9961390, 0.9994498, -0.0027859, 0.0025426
6: 0.0040955, 0.0073569, 0.0043791, 0.0073843, -0.0025287, 0.0023079
7: -0.0080976, 0.0040732, -0.0070393, 0.0041753, -0.0094368, 0.0086126
8: -0.0123631, -0.0028905, -0.0124425, -0.0037142, -0.0067032, 0.0073447
9: -0.0037604, -0.0029431, -0.0036893, -0.0029363, -0.0006337, 0.0005783

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018050, upper bound: 0.0019595
time: 2.39 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018050, upper bound: 0.0019595
time: 2.44 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0132529, -0.0049930, -0.0133701, -0.0054893, -0.0060779, 0.0064663
1: -0.0066751, -0.0043464, -0.0067082, -0.0044863, -0.0017136, 0.0018231
2: -0.0106908, 0.0064914, -0.0109347, 0.0054590, -0.0126433, 0.0134511
3: 0.0002125, 0.0024863, 0.0001803, 0.0023497, -0.0016731, 0.0017800
4: 0.0012406, 0.0140815, 0.0020122, 0.0142638, -0.0100525, 0.0094488
5: 0.9958509, 0.9994185, 0.9960653, 0.9994691, -0.0027929, 0.0026252
6: 0.0041175, 0.0073558, 0.0043121, 0.0074018, -0.0025351, 0.0023828
7: -0.0080156, 0.0040691, -0.0072895, 0.0042407, -0.0094606, 0.0088924
8: -0.0123599, -0.0029543, -0.0124934, -0.0035195, -0.0069209, 0.0073632
9: -0.0037548, -0.0029434, -0.0037061, -0.0029319, -0.0006353, 0.0005971

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018049, upper bound: 0.0019501
time: 2.43 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_B2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018050, upper bound: 0.0019501
time: 2.58 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0133446, -0.0048969, -0.0135725, -0.0055061, -0.0060233, 0.0066655
1: -0.0067010, -0.0043193, -0.0067653, -0.0044910, -0.0016982, 0.0018792
2: -0.0108815, 0.0066914, -0.0113557, 0.0054241, -0.0125297, 0.0138655
3: 0.0001873, 0.0025128, 0.0001245, 0.0023451, -0.0016581, 0.0018349
4: 0.0010911, 0.0142241, 0.0020383, 0.0145784, -0.0103622, 0.0093639
5: 0.9958094, 0.9994581, 0.9960725, 0.9995565, -0.0028789, 0.0026016
6: 0.0040798, 0.0073918, 0.0043187, 0.0074811, -0.0026132, 0.0023615
7: -0.0081562, 0.0042033, -0.0072649, 0.0045368, -0.0097520, 0.0088125
8: -0.0124643, -0.0028449, -0.0127239, -0.0035386, -0.0068588, 0.0075900
9: -0.0037643, -0.0029344, -0.0037044, -0.0029120, -0.0006548, 0.0005917

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A2_B1_B2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018554, upper bound: 0.0019667
time: 2.08 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018554, upper bound: 0.0019577
time: 2.22 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0133446, -0.0048969, -0.0133281, -0.0050721, -0.0062825, 0.0062409
1: -0.0067010, -0.0043193, -0.0066963, -0.0043687, -0.0017713, 0.0017596
2: -0.0108815, 0.0066914, -0.0108473, 0.0063268, -0.0130689, 0.0129824
3: 0.0001873, 0.0025128, 0.0001918, 0.0024645, -0.0017295, 0.0017180
4: 0.0010911, 0.0142241, 0.0013636, 0.0141985, -0.0097023, 0.0097669
5: 0.9958094, 0.9994581, 0.9958851, 0.9994510, -0.0026956, 0.0027135
6: 0.0040798, 0.0073918, 0.0041486, 0.0073853, -0.0024468, 0.0024631
7: -0.0081562, 0.0042033, -0.0078998, 0.0041792, -0.0091309, 0.0091917
8: -0.0124643, -0.0028449, -0.0124455, -0.0030444, -0.0071539, 0.0071066
9: -0.0037643, -0.0029344, -0.0037471, -0.0029360, -0.0006131, 0.0006172

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A2_B1_B2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018554, upper bound: 0.0019667
time: 2.30 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018554, upper bound: 0.0019577
time: 2.80 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0132571, -0.0049390, -0.0132883, -0.0056503, -0.0058524, 0.0065247
1: -0.0066763, -0.0043311, -0.0066851, -0.0045317, -0.0016500, 0.0018396
2: -0.0106995, 0.0066038, -0.0107645, 0.0051240, -0.0121741, 0.0135727
3: 0.0002114, 0.0025012, 0.0002028, 0.0023054, -0.0016111, 0.0017961
4: 0.0011566, 0.0140880, 0.0022625, 0.0141366, -0.0101434, 0.0090982
5: 0.9958276, 0.9994203, 0.9961348, 0.9994338, -0.0028181, 0.0025277
6: 0.0040963, 0.0073575, 0.0043752, 0.0073697, -0.0025580, 0.0022944
7: -0.0080946, 0.0040753, -0.0070539, 0.0041210, -0.0095461, 0.0085624
8: -0.0123647, -0.0028928, -0.0124002, -0.0037028, -0.0066641, 0.0074297
9: -0.0037602, -0.0029430, -0.0036903, -0.0029399, -0.0006410, 0.0005749

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_B1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018054, upper bound: 0.0019427
time: 2.60 seconds

## Relational analysis of NS_A2_B2_A2_B2_B1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018054, upper bound: 0.0019427
time: 2.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0133459, -0.0048987, -0.0135116, -0.0055080, -0.0060058, 0.0067476
1: -0.0067014, -0.0043198, -0.0067481, -0.0044916, -0.0016932, 0.0019024
2: -0.0108844, 0.0066876, -0.0112291, 0.0054201, -0.0124932, 0.0140363
3: 0.0001869, 0.0025123, 0.0001413, 0.0023446, -0.0016533, 0.0018575
4: 0.0010940, 0.0142262, 0.0020413, 0.0144838, -0.0104899, 0.0093366
5: 0.9958102, 0.9994587, 0.9960733, 0.9995303, -0.0029144, 0.0025940
6: 0.0040806, 0.0073923, 0.0043194, 0.0074573, -0.0026454, 0.0023546
7: -0.0081536, 0.0042053, -0.0072621, 0.0044477, -0.0098721, 0.0087868
8: -0.0124659, -0.0028469, -0.0126545, -0.0035408, -0.0068388, 0.0076835
9: -0.0037641, -0.0029342, -0.0037043, -0.0029180, -0.0006629, 0.0005900

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A2_B2_B2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018568, upper bound: 0.0019513
time: 2.18 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018568, upper bound: 0.0019430
time: 2.39 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0133459, -0.0048987, -0.0132797, -0.0050548, -0.0062672, 0.0063277
1: -0.0067014, -0.0043198, -0.0066827, -0.0043638, -0.0017670, 0.0017840
2: -0.0108844, 0.0066876, -0.0107467, 0.0063628, -0.0130370, 0.0131630
3: 0.0001869, 0.0025123, 0.0002051, 0.0024693, -0.0017252, 0.0017419
4: 0.0010940, 0.0142262, 0.0013367, 0.0141233, -0.0098372, 0.0097430
5: 0.9958102, 0.9994587, 0.9958776, 0.9994301, -0.0027331, 0.0027069
6: 0.0040806, 0.0073923, 0.0041418, 0.0073664, -0.0024808, 0.0024571
7: -0.0081536, 0.0042053, -0.0079251, 0.0041084, -0.0092579, 0.0091693
8: -0.0124659, -0.0028469, -0.0123905, -0.0030247, -0.0071365, 0.0072054
9: -0.0037641, -0.0029342, -0.0037488, -0.0029407, -0.0006216, 0.0006157

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A2_B2_B2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018568, upper bound: 0.0019513
time: 2.31 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018568, upper bound: 0.0019430
time: 2.29 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 6.83 seconds
NS_A1_A1_B2_B1_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018166, upper bound: 0.0018666
NS_A1_A1_B2_B1_B1_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019113, upper bound: 0.0018670
NS_A1_A1_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019534, upper bound: 0.0018728
NS_A1_A1_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019534, upper bound: 0.0018722
NS_A1_A1_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019652, upper bound: 0.0018125
NS_A1_A1_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019570, upper bound: 0.0018125
NS_A1_A1_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019601, upper bound: 0.0018626
NS_A1_A1_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019601, upper bound: 0.0018622
NS_A1_A1_B2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019276, upper bound: 0.0017696
NS_A1_A1_B2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019159, upper bound: 0.0017677
NS_A1_A1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019367, upper bound: 0.0018354
NS_A1_A1_B2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019260, upper bound: 0.0018353
NS_A1_A1_B2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019322, upper bound: 0.0017600
NS_A1_A1_B2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019212, upper bound: 0.0017583
NS_A1_A1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019409, upper bound: 0.0018242
NS_A1_A1_B2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0019305, upper bound: 0.0018239
NS_A2_B1_B1_A1_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018667, upper bound: 0.0019113
NS_A2_B1_B1_A1_A1_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018670, upper bound: 0.0019113
NS_A2_B1_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018727, upper bound: 0.0019534
NS_A2_B1_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018722, upper bound: 0.0019534
NS_A2_B1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018125, upper bound: 0.0019652
NS_A2_B1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018125, upper bound: 0.0019570
NS_A2_B1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018114, upper bound: 0.0019601
NS_A2_B1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018622, upper bound: 0.0019601
NS_A2_B1_B1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017696, upper bound: 0.0019276
NS_A2_B1_B1_A2_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017677, upper bound: 0.0019159
NS_A2_B1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018354, upper bound: 0.0019367
NS_A2_B1_B1_A2_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018352, upper bound: 0.0019260
NS_A2_B1_B1_A2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017600, upper bound: 0.0019322
NS_A2_B1_B1_A2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0017583, upper bound: 0.0019212
NS_A2_B1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018242, upper bound: 0.0019409
NS_A2_B1_B1_A2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018239, upper bound: 0.0019305
NS_A2_B1_B2_A1_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018339, upper bound: 0.0019233
NS_A2_B1_B2_A1_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018315, upper bound: 0.0019233
NS_A2_B1_B2_A1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018339, upper bound: 0.0019259
NS_A2_B1_B2_A1_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018315, upper bound: 0.0019259
NS_A2_B1_B2_A2_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018228, upper bound: 0.0019285
NS_A2_B1_B2_A2_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018201, upper bound: 0.0019285
NS_A2_B1_B2_A2_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018228, upper bound: 0.0019305
NS_A2_B1_B2_A2_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018201, upper bound: 0.0019305
NS_A2_B2_A1_B1_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018846, upper bound: 0.0019176
NS_A2_B2_A1_B1_A1_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018856, upper bound: 0.0019176
NS_A2_B2_A1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018888, upper bound: 0.0019598
NS_A2_B2_A1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018889, upper bound: 0.0019598
NS_A2_B2_A1_B1_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018744, upper bound: 0.0019192
NS_A2_B2_A1_B1_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018755, upper bound: 0.0019192
NS_A2_B2_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018793, upper bound: 0.0019659
NS_A2_B2_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018796, upper bound: 0.0019659
NS_A2_B2_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018750, upper bound: 0.0019372
NS_A2_B2_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018738, upper bound: 0.0019372
NS_A2_B2_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018646, upper bound: 0.0019428
NS_A2_B2_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018638, upper bound: 0.0019427
NS_A2_B2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018050, upper bound: 0.0019595
NS_A2_B2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018050, upper bound: 0.0019595
NS_A2_B2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018049, upper bound: 0.0019501
NS_A2_B2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018050, upper bound: 0.0019501
NS_A2_B2_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018554, upper bound: 0.0019667
NS_A2_B2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018554, upper bound: 0.0019577
NS_A2_B2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018554, upper bound: 0.0019667
NS_A2_B2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018554, upper bound: 0.0019577
NS_A2_B2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018054, upper bound: 0.0019427
NS_A2_B2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018054, upper bound: 0.0019427
NS_A2_B2_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018568, upper bound: 0.0019513
NS_A2_B2_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018568, upper bound: 0.0019430
NS_A2_B2_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018568, upper bound: 0.0019513
NS_A2_B2_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 5, lower bound: -0.0018568, upper bound: 0.0019430

## BFS NS instance: NS_A1_A1_B2_B1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0131474, -0.0054021, -0.0135625, -0.0056284, -0.0059188, 0.0063457
1: -0.0066454, -0.0044617, -0.0067624, -0.0045255, -0.0016687, 0.0017891
2: -0.0104714, 0.0056403, -0.0113349, 0.0051696, -0.0123122, 0.0132003
3: 0.0002416, 0.0023737, 0.0001273, 0.0023114, -0.0016293, 0.0017469
4: 0.0018767, 0.0139176, 0.0022285, 0.0145629, -0.0098651, 0.0092014
5: 0.9960276, 0.9993729, 0.9961253, 0.9995522, -0.0027408, 0.0025564
6: 0.0042779, 0.0073145, 0.0043667, 0.0074772, -0.0024878, 0.0023205
7: -0.0074170, 0.0039148, -0.0070859, 0.0045222, -0.0092841, 0.0086595
8: -0.0122398, -0.0034202, -0.0127125, -0.0036779, -0.0067397, 0.0072259
9: -0.0037147, -0.0029537, -0.0036924, -0.0029130, -0.0006234, 0.0005815

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_A1_B2_B1_B1_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018884, upper bound: 0.0017939
time: 2.10 seconds

## Relational analysis of NS_A1_A1_B2_B1_B1_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018826, upper bound: 0.0017932
time: 2.44 seconds

## BFS NS instance: NS_A1_A1_B2_B1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0131450, -0.0054552, -0.0136053, -0.0054619, -0.0061267, 0.0063688
1: -0.0066447, -0.0044767, -0.0067745, -0.0044786, -0.0017274, 0.0017956
2: -0.0104665, 0.0055300, -0.0114240, 0.0055159, -0.0127449, 0.0132484
3: 0.0002422, 0.0023591, 0.0001155, 0.0023572, -0.0016866, 0.0017532
4: 0.0019591, 0.0139139, 0.0019696, 0.0146294, -0.0099010, 0.0095247
5: 0.9960505, 0.9993719, 0.9960534, 0.9995707, -0.0027508, 0.0026462
6: 0.0042987, 0.0073135, 0.0043014, 0.0074940, -0.0024969, 0.0024020
7: -0.0073394, 0.0039114, -0.0073295, 0.0045848, -0.0093180, 0.0089638
8: -0.0122371, -0.0034806, -0.0127612, -0.0034883, -0.0069765, 0.0072522
9: -0.0037094, -0.0029540, -0.0037088, -0.0029088, -0.0006257, 0.0006019

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 101

## Relational analysis of NS_A1_A1_B2_B1_B1_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019100, upper bound: 0.0018084
time: 2.32 seconds

## Relational analysis of NS_A1_A1_B2_B1_B1_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019092, upper bound: 0.0018214
time: 2.44 seconds

## BFS NS instance: NS_A1_A1_B2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0128566, -0.0055619, -0.0134767, -0.0055227, -0.0057089, 0.0063154
1: -0.0065634, -0.0045068, -0.0067382, -0.0044957, -0.0016096, 0.0017806
2: -0.0098664, 0.0053080, -0.0111564, 0.0053895, -0.0118757, 0.0131374
3: 0.0003216, 0.0023297, 0.0001509, 0.0023405, -0.0015716, 0.0017385
4: 0.0021250, 0.0134654, 0.0020641, 0.0144294, -0.0098181, 0.0088751
5: 0.9960967, 0.9992474, 0.9960797, 0.9995152, -0.0027278, 0.0024658
6: 0.0043406, 0.0072005, 0.0043252, 0.0074436, -0.0024760, 0.0022382
7: -0.0071833, 0.0034893, -0.0072406, 0.0043966, -0.0092399, 0.0083525
8: -0.0119086, -0.0036021, -0.0126147, -0.0035575, -0.0065008, 0.0071914
9: -0.0036990, -0.0029823, -0.0037028, -0.0029214, -0.0006204, 0.0005609

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_A1_B2_B1_B2_A1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018911, upper bound: 0.0017432
time: 1.73 seconds

## Relational analysis of NS_A1_A1_B2_B1_B2_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018902, upper bound: 0.0017159
time: 2.09 seconds

## BFS NS instance: NS_A1_A1_B2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0129117, -0.0053946, -0.0134739, -0.0055809, -0.0057358, 0.0065041
1: -0.0065789, -0.0044596, -0.0067375, -0.0045121, -0.0016171, 0.0018337
2: -0.0099811, 0.0056560, -0.0111507, 0.0052684, -0.0119317, 0.0135297
3: 0.0003065, 0.0023758, 0.0001517, 0.0023245, -0.0015790, 0.0017904
4: 0.0018649, 0.0135511, 0.0021546, 0.0144252, -0.0101113, 0.0089170
5: 0.9960243, 0.9992711, 0.9961049, 0.9995140, -0.0028092, 0.0024774
6: 0.0042750, 0.0072221, 0.0043480, 0.0074425, -0.0025499, 0.0022487
7: -0.0074281, 0.0035699, -0.0071554, 0.0043926, -0.0095158, 0.0083919
8: -0.0119714, -0.0034116, -0.0126116, -0.0036238, -0.0065314, 0.0074062
9: -0.0037154, -0.0029769, -0.0036971, -0.0029217, -0.0006390, 0.0005635

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_A1_B2_B1_B2_A1_A2_A1

### Relational analysis result of NS_A1_A1_B2_B1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018856, upper bound: 0.0017128
time: 1.86 seconds

## Relational analysis of NS_A1_A1_B2_B1_B2_A1_A2_A2

### Relational analysis result of NS_A1_A1_B2_B1_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018794, upper bound: 0.0017117
time: 2.39 seconds

## BFS NS instance: NS_A1_A1_B2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0130904, -0.0054280, -0.0135588, -0.0056020, -0.0057391, 0.0065231
1: -0.0066293, -0.0044690, -0.0067614, -0.0045181, -0.0016181, 0.0018391
2: -0.0103529, 0.0055865, -0.0113272, 0.0052245, -0.0119385, 0.0135694
3: 0.0002573, 0.0023666, 0.0001283, 0.0023187, -0.0015799, 0.0017957
4: 0.0019169, 0.0138290, 0.0021874, 0.0145571, -0.0101409, 0.0089221
5: 0.9960388, 0.9993483, 0.9961139, 0.9995507, -0.0028175, 0.0024788
6: 0.0042881, 0.0072921, 0.0043563, 0.0074758, -0.0025574, 0.0022500
7: -0.0073792, 0.0038314, -0.0071246, 0.0045167, -0.0095437, 0.0083967
8: -0.0121749, -0.0034497, -0.0127082, -0.0036478, -0.0065351, 0.0074279
9: -0.0037121, -0.0029593, -0.0036950, -0.0029133, -0.0006408, 0.0005638

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_A1_B2_B1_B2_A2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018940, upper bound: 0.0017797
time: 2.54 seconds

## Relational analysis of NS_A1_A1_B2_B1_B2_A2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018887, upper bound: 0.0017793
time: 2.62 seconds

## BFS NS instance: NS_A1_A1_B2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0130879, -0.0054819, -0.0136031, -0.0054359, -0.0059497, 0.0065347
1: -0.0066286, -0.0044842, -0.0067739, -0.0044712, -0.0016774, 0.0018424
2: -0.0103477, 0.0054744, -0.0114194, 0.0055702, -0.0123765, 0.0135934
3: 0.0002579, 0.0023517, 0.0001161, 0.0023644, -0.0016378, 0.0017989
4: 0.0020006, 0.0138251, 0.0019291, 0.0146260, -0.0101589, 0.0092494
5: 0.9960620, 0.9993473, 0.9960421, 0.9995698, -0.0028224, 0.0025698
6: 0.0043092, 0.0072911, 0.0042912, 0.0074931, -0.0025619, 0.0023326
7: -0.0073003, 0.0038278, -0.0073677, 0.0045816, -0.0095606, 0.0087047
8: -0.0121720, -0.0035110, -0.0127587, -0.0034586, -0.0067749, 0.0074411
9: -0.0037068, -0.0029596, -0.0037113, -0.0029090, -0.0006420, 0.0005845

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_A1_B2_B1_B2_A2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018905, upper bound: 0.0017982
time: 1.95 seconds

## Relational analysis of NS_A1_A1_B2_B1_B2_A2_B2_B2

### Relational analysis result of NS_A1_A1_B2_B1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018887, upper bound: 0.0017758
time: 2.31 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0130806, -0.0055347, -0.0133833, -0.0050629, -0.0063903, 0.0062046
1: -0.0066266, -0.0044991, -0.0067119, -0.0043661, -0.0018017, 0.0017493
2: -0.0103324, 0.0053645, -0.0109621, 0.0063460, -0.0132932, 0.0129067
3: 0.0002600, 0.0023372, 0.0001766, 0.0024671, -0.0017591, 0.0017080
4: 0.0020828, 0.0138137, 0.0013492, 0.0142843, -0.0096457, 0.0099345
5: 0.9960849, 0.9993441, 0.9958810, 0.9994748, -0.0026799, 0.0027601
6: 0.0043299, 0.0072883, 0.0041449, 0.0074069, -0.0024325, 0.0025053
7: -0.0072230, 0.0038171, -0.0079133, 0.0042599, -0.0090777, 0.0093495
8: -0.0121637, -0.0035712, -0.0125084, -0.0030339, -0.0072767, 0.0070652
9: -0.0037016, -0.0029603, -0.0037480, -0.0029306, -0.0006095, 0.0006278

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_A1_B2_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_A1_B2_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018187, upper bound: 0.0017304
time: 2.30 seconds

## Relational analysis of NS_A1_A1_B2_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_A1_B2_B2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018105, upper bound: 0.0016783
time: 2.31 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0130819, -0.0055365, -0.0133377, -0.0050433, -0.0063484, 0.0062308
1: -0.0066269, -0.0044996, -0.0066990, -0.0043605, -0.0017898, 0.0017567
2: -0.0103352, 0.0053607, -0.0108672, 0.0063868, -0.0132059, 0.0129613
3: 0.0002596, 0.0023367, 0.0001892, 0.0024725, -0.0017476, 0.0017152
4: 0.0020856, 0.0138157, 0.0013188, 0.0142133, -0.0096864, 0.0098693
5: 0.9960856, 0.9993446, 0.9958726, 0.9994551, -0.0026912, 0.0027420
6: 0.0043306, 0.0072888, 0.0041372, 0.0073891, -0.0024428, 0.0024889
7: -0.0072204, 0.0038190, -0.0079420, 0.0041932, -0.0091160, 0.0092881
8: -0.0121652, -0.0035733, -0.0124564, -0.0030116, -0.0072289, 0.0070950
9: -0.0037015, -0.0029602, -0.0037499, -0.0029351, -0.0006121, 0.0006237

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_A1_B2_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_A1_B2_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018218, upper bound: 0.0017173
time: 1.67 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2_A2_A1_B2

### Relational analysis result of NS_A1_A1_B2_B2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018127, upper bound: 0.0016638
time: 1.91 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0135625, -0.0056284, -0.0131474, -0.0054021, -0.0063457, 0.0059188
1: -0.0067624, -0.0045255, -0.0066454, -0.0044617, -0.0017891, 0.0016687
2: -0.0113349, 0.0051696, -0.0104714, 0.0056403, -0.0132003, 0.0123122
3: 0.0001273, 0.0023114, 0.0002416, 0.0023737, -0.0017469, 0.0016293
4: 0.0022285, 0.0145629, 0.0018767, 0.0139176, -0.0092014, 0.0098651
5: 0.9961253, 0.9995522, 0.9960276, 0.9993729, -0.0025564, 0.0027408
6: 0.0043667, 0.0074772, 0.0042779, 0.0073145, -0.0023205, 0.0024878
7: -0.0070859, 0.0045222, -0.0074170, 0.0039148, -0.0086595, 0.0092841
8: -0.0127125, -0.0036779, -0.0122398, -0.0034202, -0.0072259, 0.0067397
9: -0.0036924, -0.0029130, -0.0037147, -0.0029537, -0.0005815, 0.0006234

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_B1_A1_A1_A2_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017939, upper bound: 0.0018885
time: 2.10 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_A2_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017932, upper bound: 0.0018826
time: 1.78 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0136053, -0.0054619, -0.0131450, -0.0054552, -0.0063688, 0.0061267
1: -0.0067745, -0.0044786, -0.0066447, -0.0044767, -0.0017956, 0.0017274
2: -0.0114240, 0.0055159, -0.0104665, 0.0055300, -0.0132484, 0.0127449
3: 0.0001155, 0.0023572, 0.0002422, 0.0023591, -0.0017532, 0.0016866
4: 0.0019696, 0.0146294, 0.0019591, 0.0139139, -0.0095247, 0.0099010
5: 0.9960534, 0.9995707, 0.9960505, 0.9993719, -0.0026462, 0.0027508
6: 0.0043014, 0.0074940, 0.0042987, 0.0073135, -0.0024020, 0.0024969
7: -0.0073295, 0.0045848, -0.0073394, 0.0039114, -0.0089638, 0.0093180
8: -0.0127612, -0.0034883, -0.0122371, -0.0034806, -0.0072522, 0.0069766
9: -0.0037088, -0.0029088, -0.0037094, -0.0029540, -0.0006019, 0.0006257

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 101

## Relational analysis of NS_A2_B1_B1_A1_A1_A2_A2_B1

### Relational analysis result of NS_A2_B1_B1_A1_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018084, upper bound: 0.0019099
time: 2.11 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_A2_A2_B2

### Relational analysis result of NS_A2_B1_B1_A1_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018214, upper bound: 0.0019091
time: 2.54 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0134767, -0.0055227, -0.0128566, -0.0055619, -0.0063154, 0.0057089
1: -0.0067382, -0.0044957, -0.0065634, -0.0045068, -0.0017806, 0.0016095
2: -0.0111564, 0.0053895, -0.0098664, 0.0053080, -0.0131374, 0.0118757
3: 0.0001509, 0.0023405, 0.0003216, 0.0023297, -0.0017385, 0.0015716
4: 0.0020641, 0.0144294, 0.0021250, 0.0134654, -0.0088751, 0.0098181
5: 0.9960797, 0.9995152, 0.9960967, 0.9992474, -0.0024658, 0.0027278
6: 0.0043252, 0.0074436, 0.0043406, 0.0072005, -0.0022382, 0.0024760
7: -0.0072406, 0.0043966, -0.0071833, 0.0034893, -0.0083525, 0.0092399
8: -0.0126147, -0.0035575, -0.0119086, -0.0036021, -0.0071914, 0.0065008
9: -0.0037028, -0.0029214, -0.0036990, -0.0029823, -0.0005609, 0.0006204

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017433, upper bound: 0.0018911
time: 2.24 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017160, upper bound: 0.0018902
time: 1.66 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0134739, -0.0055809, -0.0129117, -0.0053946, -0.0065040, 0.0057358
1: -0.0067375, -0.0045121, -0.0065789, -0.0044596, -0.0018337, 0.0016171
2: -0.0111507, 0.0052684, -0.0099811, 0.0056560, -0.0135297, 0.0119317
3: 0.0001517, 0.0023245, 0.0003065, 0.0023758, -0.0017904, 0.0015790
4: 0.0021546, 0.0144252, 0.0018649, 0.0135511, -0.0089170, 0.0101113
5: 0.9961049, 0.9995140, 0.9960243, 0.9992711, -0.0024774, 0.0028092
6: 0.0043480, 0.0074425, 0.0042750, 0.0072221, -0.0022487, 0.0025499
7: -0.0071554, 0.0043926, -0.0074281, 0.0035699, -0.0083919, 0.0095158
8: -0.0126116, -0.0036238, -0.0119714, -0.0034116, -0.0074062, 0.0065314
9: -0.0036971, -0.0029217, -0.0037154, -0.0029769, -0.0005635, 0.0006390

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017128, upper bound: 0.0018856
time: 2.08 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017117, upper bound: 0.0018794
time: 2.06 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0135588, -0.0056020, -0.0130904, -0.0054280, -0.0065231, 0.0057391
1: -0.0067614, -0.0045181, -0.0066293, -0.0044690, -0.0018391, 0.0016181
2: -0.0113272, 0.0052245, -0.0103529, 0.0055865, -0.0135694, 0.0119385
3: 0.0001283, 0.0023187, 0.0002573, 0.0023666, -0.0017957, 0.0015799
4: 0.0021874, 0.0145571, 0.0019169, 0.0138290, -0.0089221, 0.0101409
5: 0.9961139, 0.9995507, 0.9960388, 0.9993483, -0.0024788, 0.0028175
6: 0.0043563, 0.0074758, 0.0042881, 0.0072921, -0.0022500, 0.0025574
7: -0.0071246, 0.0045167, -0.0073792, 0.0038314, -0.0083967, 0.0095437
8: -0.0127082, -0.0036478, -0.0121749, -0.0034497, -0.0074279, 0.0065351
9: -0.0036950, -0.0029133, -0.0037121, -0.0029593, -0.0005638, 0.0006408

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017797, upper bound: 0.0018941
time: 2.13 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017793, upper bound: 0.0018887
time: 2.37 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0136031, -0.0054359, -0.0130879, -0.0054819, -0.0065347, 0.0059497
1: -0.0067739, -0.0044712, -0.0066286, -0.0044842, -0.0018424, 0.0016774
2: -0.0114194, 0.0055702, -0.0103477, 0.0054744, -0.0135934, 0.0123765
3: 0.0001161, 0.0023644, 0.0002579, 0.0023517, -0.0017989, 0.0016378
4: 0.0019291, 0.0146260, 0.0020006, 0.0138251, -0.0092494, 0.0101589
5: 0.9960421, 0.9995698, 0.9960620, 0.9993473, -0.0025698, 0.0028224
6: 0.0042912, 0.0074931, 0.0043092, 0.0072911, -0.0023326, 0.0025619
7: -0.0073677, 0.0045816, -0.0073003, 0.0038278, -0.0087047, 0.0095606
8: -0.0127587, -0.0034586, -0.0121720, -0.0035110, -0.0074411, 0.0067749
9: -0.0037113, -0.0029090, -0.0037068, -0.0029596, -0.0005845, 0.0006420

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_A2_A1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017982, upper bound: 0.0018906
time: 2.13 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_A2_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017759, upper bound: 0.0018887
time: 2.34 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0133833, -0.0050629, -0.0130806, -0.0055347, -0.0062046, 0.0063903
1: -0.0067119, -0.0043661, -0.0066266, -0.0044991, -0.0017493, 0.0018017
2: -0.0109621, 0.0063460, -0.0103324, 0.0053645, -0.0129067, 0.0132932
3: 0.0001766, 0.0024671, 0.0002600, 0.0023372, -0.0017080, 0.0017591
4: 0.0013492, 0.0142843, 0.0020828, 0.0138137, -0.0099345, 0.0096457
5: 0.9958810, 0.9994748, 0.9960849, 0.9993441, -0.0027601, 0.0026799
6: 0.0041449, 0.0074069, 0.0043299, 0.0072883, -0.0025053, 0.0024325
7: -0.0079133, 0.0042599, -0.0072230, 0.0038171, -0.0093495, 0.0090777
8: -0.0125084, -0.0030339, -0.0121637, -0.0035712, -0.0070652, 0.0072767
9: -0.0037480, -0.0029306, -0.0037016, -0.0029603, -0.0006278, 0.0006095

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017304, upper bound: 0.0018187
time: 1.84 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016783, upper bound: 0.0018105
time: 1.88 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0133377, -0.0050433, -0.0130819, -0.0055365, -0.0062308, 0.0063484
1: -0.0066990, -0.0043605, -0.0066269, -0.0044996, -0.0017567, 0.0017898
2: -0.0108672, 0.0063868, -0.0103352, 0.0053607, -0.0129613, 0.0132059
3: 0.0001892, 0.0024725, 0.0002596, 0.0023367, -0.0017152, 0.0017476
4: 0.0013188, 0.0142133, 0.0020856, 0.0138157, -0.0098693, 0.0096864
5: 0.9958726, 0.9994551, 0.9960856, 0.9993446, -0.0027420, 0.0026912
6: 0.0041372, 0.0073891, 0.0043306, 0.0072888, -0.0024889, 0.0024428
7: -0.0079420, 0.0041932, -0.0072204, 0.0038190, -0.0092881, 0.0091160
8: -0.0124564, -0.0030116, -0.0121652, -0.0035733, -0.0070950, 0.0072289
9: -0.0037499, -0.0029351, -0.0037015, -0.0029602, -0.0006237, 0.0006121

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_B1_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017172, upper bound: 0.0018218
time: 1.89 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016638, upper bound: 0.0018127
time: 1.85 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0135625, -0.0056284, -0.0135732, -0.0053583, -0.0060492, 0.0060017
1: -0.0067624, -0.0045255, -0.0067654, -0.0044494, -0.0017055, 0.0016921
2: -0.0113349, 0.0051696, -0.0113571, 0.0057315, -0.0125835, 0.0124847
3: 0.0001273, 0.0023114, 0.0001244, 0.0023858, -0.0016652, 0.0016522
4: 0.0022285, 0.0145629, 0.0018085, 0.0145795, -0.0093303, 0.0094041
5: 0.9961253, 0.9995522, 0.9960087, 0.9995568, -0.0025922, 0.0026127
6: 0.0043667, 0.0074772, 0.0042607, 0.0074814, -0.0023530, 0.0023716
7: -0.0070859, 0.0045222, -0.0074812, 0.0045378, -0.0087808, 0.0088503
8: -0.0127125, -0.0036779, -0.0127246, -0.0033703, -0.0068882, 0.0068341
9: -0.0036924, -0.0029130, -0.0037190, -0.0029119, -0.0005896, 0.0005943

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A1_B1_A1_A2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018175, upper bound: 0.0018931
time: 2.31 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_A2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018173, upper bound: 0.0018883
time: 1.80 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0136053, -0.0054619, -0.0135705, -0.0054169, -0.0060720, 0.0061998
1: -0.0067745, -0.0044786, -0.0067647, -0.0044659, -0.0017119, 0.0017479
2: -0.0114240, 0.0055159, -0.0113515, 0.0056095, -0.0126309, 0.0128968
3: 0.0001155, 0.0023572, 0.0001251, 0.0023696, -0.0016715, 0.0017067
4: 0.0019696, 0.0146294, 0.0018997, 0.0145753, -0.0096382, 0.0094396
5: 0.9960534, 0.9995707, 0.9960341, 0.9995556, -0.0026778, 0.0026226
6: 0.0043014, 0.0074940, 0.0042837, 0.0074803, -0.0024306, 0.0023805
7: -0.0073295, 0.0045848, -0.0073953, 0.0045338, -0.0090707, 0.0088837
8: -0.0127612, -0.0034883, -0.0127216, -0.0034371, -0.0069142, 0.0070597
9: -0.0037088, -0.0029088, -0.0037132, -0.0029122, -0.0006091, 0.0005965

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A1_B1_A1_A2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018172, upper bound: 0.0018931
time: 2.50 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_A2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018169, upper bound: 0.0018883
time: 1.91 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.83 + 599.25 = 604.08 seconds
