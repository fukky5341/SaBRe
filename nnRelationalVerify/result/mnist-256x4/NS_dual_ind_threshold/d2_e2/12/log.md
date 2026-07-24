## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.068045589


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0045828, 0.0243426, 0.0045828, 0.0243426, -0.0197598, 0.0197598)
1: (-0.0056720, 0.0211549, -0.0056720, 0.0211549, -0.0268269, 0.0268269)
2: (0.0024190, 0.0394507, 0.0024190, 0.0394507, -0.0370317, 0.0370317)
3: (-0.0105444, -0.0010774, -0.0105444, -0.0010774, -0.0094671, 0.0094671)
4: (-0.0098809, 0.0091565, -0.0098809, 0.0091565, -0.0190374, 0.0190374)
5: (-0.0112748, 0.0139717, -0.0112748, 0.0139717, -0.0252465, 0.0252465)
6: (0.9184468, 0.9928610, 0.9184468, 0.9928610, -0.0744143, 0.0744143)
7: (-0.0241657, 0.0032827, -0.0241657, 0.0032827, -0.0274484, 0.0274484)
8: (-0.0217363, 0.0139040, -0.0217363, 0.0139040, -0.0356403, 0.0356403)
9: (-0.0066065, 0.0159613, -0.0066065, 0.0159613, -0.0225677, 0.0225677)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.62 + 2.72 = 4.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0731673, upper bound: 0.0731670

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 211

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0705525, upper bound: 0.0704885
time: 1.67 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0704887, upper bound: 0.0704889
time: 1.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.48 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.48
Output dim: 6, lower bound: -0.0705525, upper bound: 0.0704885
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.48
Output dim: 6, lower bound: -0.0704887, upper bound: 0.0704889

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0050161, 0.0092801, 0.0045828, 0.0243426, -0.0193266, 0.0046972
1: -0.0042865, 0.0051613, -0.0056720, 0.0211549, -0.0254414, 0.0108334
2: 0.0096787, 0.0287005, 0.0024190, 0.0394507, -0.0297720, 0.0262814
3: -0.0078428, -0.0022435, -0.0105444, -0.0010774, -0.0067655, 0.0083009
4: -0.0064776, 0.0051736, -0.0098809, 0.0091565, -0.0156341, 0.0150545
5: -0.0046951, 0.0087411, -0.0112748, 0.0139717, -0.0186668, 0.0200159
6: 0.9810222, 0.9925619, 0.9184468, 0.9928610, -0.0118389, 0.0741152
7: -0.0210998, -0.0044199, -0.0241657, 0.0032827, -0.0243825, 0.0197458
8: -0.0166465, 0.0037897, -0.0217363, 0.0139040, -0.0305505, 0.0255260
9: -0.0065381, 0.0069701, -0.0066065, 0.0159613, -0.0224993, 0.0135766

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 211

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0702678, upper bound: 0.0702676
time: 1.70 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0702678, upper bound: 0.0704890
time: 1.62 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0050722, 0.0091536, 0.0048011, 0.0196690, -0.0145968, 0.0043525
1: -0.0041239, 0.0041639, -0.0053243, 0.0164911, -0.0206150, 0.0094882
2: 0.0093247, 0.0285399, 0.0046080, 0.0360108, -0.0266861, 0.0239319
3: -0.0073531, -0.0023092, -0.0098523, -0.0013915, -0.0059616, 0.0075431
4: -0.0060019, 0.0051049, -0.0088771, 0.0080991, -0.0141010, 0.0139820
5: -0.0040834, 0.0084132, -0.0094502, 0.0122953, -0.0163788, 0.0178634
6: 0.9814776, 0.9926040, 0.9378989, 0.9925925, -0.0111149, 0.0547051
7: -0.0205634, -0.0041422, -0.0233246, 0.0000010, -0.0205644, 0.0191824
8: -0.0163421, 0.0030192, -0.0203502, 0.0110404, -0.0273825, 0.0233693
9: -0.0067117, 0.0062617, -0.0065837, 0.0131923, -0.0199039, 0.0128454

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0704890, upper bound: 0.0702678
time: 1.70 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0704890, upper bound: 0.0704891
time: 1.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.02 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.02
Output dim: 6, lower bound: -0.0702678, upper bound: 0.0702676
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.02
Output dim: 6, lower bound: -0.0702678, upper bound: 0.0704890
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.02
Output dim: 6, lower bound: -0.0704890, upper bound: 0.0702678
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.02
Output dim: 6, lower bound: -0.0704890, upper bound: 0.0704891

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0050161, 0.0092801, 0.0050161, 0.0092801, -0.0042640, 0.0042640
1: -0.0042865, 0.0051613, -0.0042865, 0.0051613, -0.0094479, 0.0094479
2: 0.0096787, 0.0287005, 0.0096787, 0.0287005, -0.0190217, 0.0190217
3: -0.0078428, -0.0022435, -0.0078428, -0.0022435, -0.0055993, 0.0055993
4: -0.0064776, 0.0051736, -0.0064776, 0.0051736, -0.0116513, 0.0116513
5: -0.0046951, 0.0087411, -0.0046951, 0.0087411, -0.0134362, 0.0134362
6: 0.9810222, 0.9925619, 0.9810222, 0.9925619, -0.0115398, 0.0115398
7: -0.0210998, -0.0044199, -0.0210998, -0.0044199, -0.0166799, 0.0166799
8: -0.0166465, 0.0037897, -0.0166465, 0.0037897, -0.0204362, 0.0204362
9: -0.0065381, 0.0069701, -0.0065381, 0.0069701, -0.0135082, 0.0135082

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0705496, upper bound: 0.0700038
time: 1.69 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0701754, upper bound: 0.0700028
time: 1.51 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.0050161, 0.0092801, 0.0050722, 0.0091536, -0.0041375, 0.0042078
1: -0.0042865, 0.0051613, -0.0041239, 0.0041639, -0.0084505, 0.0092852
2: 0.0096787, 0.0287005, 0.0093247, 0.0285399, -0.0188612, 0.0193757
3: -0.0078428, -0.0022435, -0.0073531, -0.0023092, -0.0055336, 0.0051096
4: -0.0064776, 0.0051736, -0.0060019, 0.0051049, -0.0115825, 0.0111755
5: -0.0046951, 0.0087411, -0.0040834, 0.0084132, -0.0131083, 0.0128246
6: 0.9810222, 0.9925619, 0.9814776, 0.9926040, -0.0115818, 0.0110844
7: -0.0210998, -0.0044199, -0.0205634, -0.0041422, -0.0169576, 0.0161436
8: -0.0166465, 0.0037897, -0.0163421, 0.0030192, -0.0196656, 0.0201319
9: -0.0065381, 0.0069701, -0.0067117, 0.0062617, -0.0127998, 0.0136818

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0705496, upper bound: 0.0703524
time: 1.73 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0701754, upper bound: 0.0703518
time: 1.61 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0050722, 0.0091536, 0.0050161, 0.0092801, -0.0042078, 0.0041375
1: -0.0041239, 0.0041639, -0.0042865, 0.0051613, -0.0092852, 0.0084505
2: 0.0093247, 0.0285399, 0.0096787, 0.0287005, -0.0193757, 0.0188612
3: -0.0073531, -0.0023092, -0.0078428, -0.0022435, -0.0051096, 0.0055336
4: -0.0060019, 0.0051049, -0.0064776, 0.0051736, -0.0111755, 0.0115825
5: -0.0040834, 0.0084132, -0.0046951, 0.0087411, -0.0128246, 0.0131083
6: 0.9814776, 0.9926040, 0.9810222, 0.9925619, -0.0110844, 0.0115818
7: -0.0205634, -0.0041422, -0.0210998, -0.0044199, -0.0161436, 0.0169576
8: -0.0163421, 0.0030192, -0.0166465, 0.0037897, -0.0201319, 0.0196656
9: -0.0067117, 0.0062617, -0.0065381, 0.0069701, -0.0136818, 0.0127998

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0704862, upper bound: 0.0700024
time: 1.86 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0703515, upper bound: 0.0700016
time: 1.98 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0050722, 0.0091536, 0.0050722, 0.0091536, -0.0040813, 0.0040813
1: -0.0041239, 0.0041639, -0.0041239, 0.0041639, -0.0082878, 0.0082878
2: 0.0093247, 0.0285399, 0.0093247, 0.0285399, -0.0192152, 0.0192152
3: -0.0073531, -0.0023092, -0.0073531, -0.0023092, -0.0050439, 0.0050439
4: -0.0060019, 0.0051049, -0.0060019, 0.0051049, -0.0111068, 0.0111068
5: -0.0040834, 0.0084132, -0.0040834, 0.0084132, -0.0124967, 0.0124967
6: 0.9814776, 0.9926040, 0.9814776, 0.9926040, -0.0111265, 0.0111265
7: -0.0205634, -0.0041422, -0.0205634, -0.0041422, -0.0164212, 0.0164212
8: -0.0163421, 0.0030192, -0.0163421, 0.0030192, -0.0193613, 0.0193613
9: -0.0067117, 0.0062617, -0.0067117, 0.0062617, -0.0129734, 0.0129734

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0704862, upper bound: 0.0703508
time: 1.93 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0703515, upper bound: 0.0703507
time: 1.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.59 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 6, lower bound: -0.0705496, upper bound: 0.0700038
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 6, lower bound: -0.0701754, upper bound: 0.0700028
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 6, lower bound: -0.0705496, upper bound: 0.0703524
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 6, lower bound: -0.0701754, upper bound: 0.0703518
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 6, lower bound: -0.0704862, upper bound: 0.0700024
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 6, lower bound: -0.0703515, upper bound: 0.0700016
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 6, lower bound: -0.0704862, upper bound: 0.0703508
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 6, lower bound: -0.0703515, upper bound: 0.0703507

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0051091, 0.0090706, 0.0050171, 0.0092779, -0.0041688, 0.0040535
1: -0.0040172, 0.0035094, -0.0042837, 0.0051441, -0.0091613, 0.0077931
2: 0.0101538, 0.0284345, 0.0096839, 0.0286977, -0.0185439, 0.0187506
3: -0.0070317, -0.0023523, -0.0078343, -0.0022446, -0.0047871, 0.0054820
4: -0.0056897, 0.0047456, -0.0064694, 0.0051665, -0.0108562, 0.0112149
5: -0.0036821, 0.0081981, -0.0046845, 0.0087354, -0.0124175, 0.0128826
6: 0.9817764, 0.9925056, 0.9810300, 0.9925614, -0.0107850, 0.0114755
7: -0.0202115, -0.0047925, -0.0210905, -0.0044239, -0.0157875, 0.0162980
8: -0.0161425, 0.0025135, -0.0166412, 0.0037764, -0.0199188, 0.0191547
9: -0.0063050, 0.0057969, -0.0065355, 0.0069578, -0.0132629, 0.0123324

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0701796, upper bound: 0.0701798
time: 1.68 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0701796, upper bound: 0.0701796
time: 1.68 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0048045, 0.0097567, 0.0050214, 0.0092682, -0.0044637, 0.0047353
1: -0.0048993, 0.0089195, -0.0042713, 0.0050679, -0.0099671, 0.0131908
2: 0.0074419, 0.0293054, 0.0097108, 0.0286854, -0.0212435, 0.0195946
3: -0.0096881, -0.0019959, -0.0077969, -0.0022496, -0.0074385, 0.0058010
4: -0.0082701, 0.0067226, -0.0064330, 0.0051351, -0.0134052, 0.0131556
5: -0.0069998, 0.0099765, -0.0046378, 0.0087104, -0.0157102, 0.0146143
6: 0.9793062, 0.9925203, 0.9810647, 0.9925581, -0.0132518, 0.0114556
7: -0.0231209, -0.0046950, -0.0210496, -0.0044450, -0.0186758, 0.0163546
8: -0.0177931, 0.0066932, -0.0166180, 0.0037175, -0.0215106, 0.0233111
9: -0.0063660, 0.0096392, -0.0065223, 0.0069037, -0.0132698, 0.0161616

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0701798, upper bound: 0.0701798
time: 1.55 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0701798, upper bound: 0.0701798
time: 1.81 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0051091, 0.0090706, 0.0050732, 0.0091515, -0.0040424, 0.0039974
1: -0.0040172, 0.0035094, -0.0041212, 0.0041474, -0.0081646, 0.0076306
2: 0.0101538, 0.0284345, 0.0093299, 0.0285372, -0.0183834, 0.0191047
3: -0.0070317, -0.0023523, -0.0073449, -0.0023103, -0.0047214, 0.0049927
4: -0.0056897, 0.0047456, -0.0059940, 0.0051026, -0.0107924, 0.0107396
5: -0.0036821, 0.0081981, -0.0040733, 0.0084078, -0.0120899, 0.0122714
6: 0.9817764, 0.9925056, 0.9814851, 0.9926034, -0.0108270, 0.0110204
7: -0.0202115, -0.0047925, -0.0205545, -0.0041462, -0.0160653, 0.0157620
8: -0.0161425, 0.0025135, -0.0163371, 0.0030064, -0.0191488, 0.0188506
9: -0.0063050, 0.0057969, -0.0067092, 0.0062500, -0.0125550, 0.0125061

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0701753, upper bound: 0.0703518
time: 1.79 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0701753, upper bound: 0.0703513
time: 1.63 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0048045, 0.0097567, 0.0050776, 0.0091416, -0.0043371, 0.0046791
1: -0.0048993, 0.0089195, -0.0041085, 0.0040693, -0.0089685, 0.0130280
2: 0.0074419, 0.0293054, 0.0093573, 0.0285247, -0.0210828, 0.0199481
3: -0.0096881, -0.0019959, -0.0073066, -0.0023154, -0.0073727, 0.0053106
4: -0.0082701, 0.0067226, -0.0059567, 0.0050907, -0.0133608, 0.0126793
5: -0.0069998, 0.0099765, -0.0040254, 0.0083821, -0.0153820, 0.0140019
6: 0.9793062, 0.9925203, 0.9815207, 0.9926000, -0.0132937, 0.0109996
7: -0.0231209, -0.0046950, -0.0205125, -0.0041678, -0.0189531, 0.0158175
8: -0.0177931, 0.0066932, -0.0163133, 0.0029460, -0.0207391, 0.0230065
9: -0.0063660, 0.0096392, -0.0066957, 0.0061945, -0.0125605, 0.0163349

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0701754, upper bound: 0.0703518
time: 2.24 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0701754, upper bound: 0.0703518
time: 1.66 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0051588, 0.0089585, 0.0050171, 0.0092779, -0.0041190, 0.0039415
1: -0.0038732, 0.0026258, -0.0042837, 0.0051441, -0.0090172, 0.0069095
2: 0.0097897, 0.0282923, 0.0096839, 0.0286977, -0.0189080, 0.0186084
3: -0.0065978, -0.0024105, -0.0078343, -0.0022446, -0.0043532, 0.0054238
4: -0.0052683, 0.0049034, -0.0064694, 0.0051665, -0.0104348, 0.0113727
5: -0.0031402, 0.0079076, -0.0046845, 0.0087354, -0.0118756, 0.0125921
6: 0.9821798, 0.9925488, 0.9810300, 0.9925614, -0.0103816, 0.0115188
7: -0.0197363, -0.0045069, -0.0210905, -0.0044239, -0.0153123, 0.0165836
8: -0.0158729, 0.0018309, -0.0166412, 0.0037764, -0.0196492, 0.0184720
9: -0.0064836, 0.0051694, -0.0065355, 0.0069578, -0.0134415, 0.0117049

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0703515, upper bound: 0.0701754
time: 1.97 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0703515, upper bound: 0.0701754
time: 1.73 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0048550, 0.0096429, 0.0050214, 0.0092682, -0.0044132, 0.0046215
1: -0.0047530, 0.0080223, -0.0042713, 0.0050679, -0.0098209, 0.0122935
2: 0.0082823, 0.0291610, 0.0097108, 0.0286854, -0.0204031, 0.0194502
3: -0.0092475, -0.0020551, -0.0077969, -0.0022496, -0.0069979, 0.0057419
4: -0.0078421, 0.0063528, -0.0064330, 0.0051351, -0.0129773, 0.0127858
5: -0.0064496, 0.0096816, -0.0046378, 0.0087104, -0.0151600, 0.0143194
6: 0.9797159, 0.9925621, 0.9810647, 0.9925581, -0.0128422, 0.0114974
7: -0.0226383, -0.0044196, -0.0210496, -0.0044450, -0.0181933, 0.0166300
8: -0.0175194, 0.0060000, -0.0166180, 0.0037175, -0.0212369, 0.0226179
9: -0.0065382, 0.0090020, -0.0065223, 0.0069037, -0.0134420, 0.0155243

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0703518, upper bound: 0.0701754
time: 1.68 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0703518, upper bound: 0.0701754
time: 1.73 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0051588, 0.0089585, 0.0050732, 0.0091515, -0.0039927, 0.0038854
1: -0.0038732, 0.0026258, -0.0041212, 0.0041474, -0.0080205, 0.0067471
2: 0.0097897, 0.0282923, 0.0093299, 0.0285372, -0.0187476, 0.0189625
3: -0.0065978, -0.0024105, -0.0073449, -0.0023103, -0.0042876, 0.0049345
4: -0.0052683, 0.0049034, -0.0059940, 0.0051026, -0.0103709, 0.0108974
5: -0.0031402, 0.0079076, -0.0040733, 0.0084078, -0.0115480, 0.0119809
6: 0.9821798, 0.9925488, 0.9814851, 0.9926034, -0.0104235, 0.0110637
7: -0.0197363, -0.0045069, -0.0205545, -0.0041462, -0.0155901, 0.0160476
8: -0.0158729, 0.0018309, -0.0163371, 0.0030064, -0.0188792, 0.0181680
9: -0.0064836, 0.0051694, -0.0067092, 0.0062500, -0.0127336, 0.0118785

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0703518, upper bound: 0.0703504
time: 1.81 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0703518, upper bound: 0.0703507
time: 1.55 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0048550, 0.0096429, 0.0050776, 0.0091416, -0.0042866, 0.0045653
1: -0.0047530, 0.0080223, -0.0041085, 0.0040693, -0.0088222, 0.0121307
2: 0.0082823, 0.0291610, 0.0093573, 0.0285247, -0.0202423, 0.0198036
3: -0.0092475, -0.0020551, -0.0073066, -0.0023154, -0.0069321, 0.0052515
4: -0.0078421, 0.0063528, -0.0059567, 0.0050907, -0.0129329, 0.0123095
5: -0.0064496, 0.0096816, -0.0040254, 0.0083821, -0.0148317, 0.0137070
6: 0.9797159, 0.9925621, 0.9815207, 0.9926000, -0.0128841, 0.0110414
7: -0.0226383, -0.0044196, -0.0205125, -0.0041678, -0.0184706, 0.0160929
8: -0.0175194, 0.0060000, -0.0163133, 0.0029460, -0.0204654, 0.0223132
9: -0.0065382, 0.0090020, -0.0066957, 0.0061945, -0.0127327, 0.0156977

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0703521, upper bound: 0.0703502
time: 1.67 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0703521, upper bound: 0.0703506
time: 1.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.02 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0701796, upper bound: 0.0701798
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0701796, upper bound: 0.0701796
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0701798, upper bound: 0.0701798
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0701798, upper bound: 0.0701798
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0701753, upper bound: 0.0703518
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0701753, upper bound: 0.0703513
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0701754, upper bound: 0.0703518
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0701754, upper bound: 0.0703518
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0703515, upper bound: 0.0701754
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0703515, upper bound: 0.0701754
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0703518, upper bound: 0.0701754
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0703518, upper bound: 0.0701754
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0703518, upper bound: 0.0703504
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0703518, upper bound: 0.0703507
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0703521, upper bound: 0.0703502
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.02
Output dim: 6, lower bound: -0.0703521, upper bound: 0.0703506

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0051091, 0.0090706, 0.0051091, 0.0090706, -0.0039615, 0.0039615
1: -0.0040172, 0.0035094, -0.0040172, 0.0035094, -0.0075266, 0.0075266
2: 0.0101538, 0.0284345, 0.0101538, 0.0284345, -0.0182808, 0.0182808
3: -0.0070317, -0.0023523, -0.0070317, -0.0023523, -0.0046794, 0.0046794
4: -0.0056897, 0.0047456, -0.0056897, 0.0047456, -0.0104353, 0.0104353
5: -0.0036821, 0.0081981, -0.0036821, 0.0081981, -0.0118802, 0.0118802
6: 0.9817764, 0.9925056, 0.9817764, 0.9925056, -0.0107292, 0.0107292
7: -0.0202115, -0.0047925, -0.0202115, -0.0047925, -0.0154189, 0.0154189
8: -0.0161425, 0.0025135, -0.0161425, 0.0025135, -0.0186560, 0.0186560
9: -0.0063050, 0.0057969, -0.0063050, 0.0057969, -0.0121019, 0.0121019

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0666429, upper bound: 0.0693945
time: 1.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0699007, upper bound: 0.0695403
time: 1.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0051091, 0.0090706, 0.0048045, 0.0097567, -0.0046476, 0.0042661
1: -0.0040172, 0.0035094, -0.0048993, 0.0089195, -0.0129367, 0.0084087
2: 0.0101538, 0.0284345, 0.0074419, 0.0293054, -0.0191516, 0.0209926
3: -0.0070317, -0.0023523, -0.0096881, -0.0019959, -0.0050358, 0.0073358
4: -0.0056897, 0.0047456, -0.0082701, 0.0067226, -0.0124123, 0.0130157
5: -0.0036821, 0.0081981, -0.0069998, 0.0099765, -0.0136586, 0.0151979
6: 0.9817764, 0.9925056, 0.9793062, 0.9925203, -0.0107439, 0.0131993
7: -0.0202115, -0.0047925, -0.0231209, -0.0046950, -0.0155165, 0.0183283
8: -0.0161425, 0.0025135, -0.0177931, 0.0066932, -0.0228357, 0.0203066
9: -0.0063050, 0.0057969, -0.0063660, 0.0096392, -0.0159443, 0.0121629

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0666429, upper bound: 0.0693945
time: 1.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0699007, upper bound: 0.0695405
time: 1.51 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0048045, 0.0097567, 0.0051091, 0.0090706, -0.0042661, 0.0046476
1: -0.0048993, 0.0089195, -0.0040172, 0.0035094, -0.0084087, 0.0129367
2: 0.0074419, 0.0293054, 0.0101538, 0.0284345, -0.0209926, 0.0191516
3: -0.0096881, -0.0019959, -0.0070317, -0.0023523, -0.0073358, 0.0050358
4: -0.0082701, 0.0067226, -0.0056897, 0.0047456, -0.0130157, 0.0124123
5: -0.0069998, 0.0099765, -0.0036821, 0.0081981, -0.0151979, 0.0136586
6: 0.9793062, 0.9925203, 0.9817764, 0.9925056, -0.0131993, 0.0107439
7: -0.0231209, -0.0046950, -0.0202115, -0.0047925, -0.0183283, 0.0155165
8: -0.0177931, 0.0066932, -0.0161425, 0.0025135, -0.0203066, 0.0228357
9: -0.0063660, 0.0096392, -0.0063050, 0.0057969, -0.0121629, 0.0159443

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0658996, upper bound: 0.0693785
time: 1.62 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695395, upper bound: 0.0695396
time: 1.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0048045, 0.0097567, 0.0048045, 0.0097567, -0.0049522, 0.0049522
1: -0.0048993, 0.0089195, -0.0048993, 0.0089195, -0.0138188, 0.0138188
2: 0.0074419, 0.0293054, 0.0074419, 0.0293054, -0.0218635, 0.0218635
3: -0.0096881, -0.0019959, -0.0096881, -0.0019959, -0.0076922, 0.0076922
4: -0.0082701, 0.0067226, -0.0082701, 0.0067226, -0.0149927, 0.0149927
5: -0.0069998, 0.0099765, -0.0069998, 0.0099765, -0.0169764, 0.0169764
6: 0.9793062, 0.9925203, 0.9793062, 0.9925203, -0.0132141, 0.0132141
7: -0.0231209, -0.0046950, -0.0231209, -0.0046950, -0.0184259, 0.0184259
8: -0.0177931, 0.0066932, -0.0177931, 0.0066932, -0.0244863, 0.0244863
9: -0.0063660, 0.0096392, -0.0063660, 0.0096392, -0.0160053, 0.0160053

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0658996, upper bound: 0.0693787
time: 1.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695395, upper bound: 0.0695395
time: 1.78 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0051091, 0.0090706, 0.0051588, 0.0089585, -0.0038494, 0.0039118
1: -0.0040172, 0.0035094, -0.0038732, 0.0026258, -0.0066430, 0.0073826
2: 0.0101538, 0.0284345, 0.0097897, 0.0282923, -0.0181385, 0.0186449
3: -0.0070317, -0.0023523, -0.0065978, -0.0024105, -0.0046212, 0.0042456
4: -0.0056897, 0.0047456, -0.0052683, 0.0049034, -0.0105931, 0.0100139
5: -0.0036821, 0.0081981, -0.0031402, 0.0079076, -0.0115897, 0.0113383
6: 0.9817764, 0.9925056, 0.9821798, 0.9925488, -0.0107725, 0.0103257
7: -0.0202115, -0.0047925, -0.0197363, -0.0045069, -0.0157046, 0.0149438
8: -0.0161425, 0.0025135, -0.0158729, 0.0018309, -0.0179733, 0.0183864
9: -0.0063050, 0.0057969, -0.0064836, 0.0051694, -0.0114744, 0.0122805

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0666382, upper bound: 0.0696392
time: 1.67 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0698969, upper bound: 0.0697903
time: 1.53 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0051091, 0.0090706, 0.0048550, 0.0096429, -0.0045338, 0.0042156
1: -0.0040172, 0.0035094, -0.0047530, 0.0080223, -0.0120395, 0.0082624
2: 0.0101538, 0.0284345, 0.0082823, 0.0291610, -0.0190072, 0.0201522
3: -0.0070317, -0.0023523, -0.0092475, -0.0020551, -0.0049767, 0.0068952
4: -0.0056897, 0.0047456, -0.0078421, 0.0063528, -0.0120425, 0.0125877
5: -0.0036821, 0.0081981, -0.0064496, 0.0096816, -0.0133636, 0.0146477
6: 0.9817764, 0.9925056, 0.9797159, 0.9925621, -0.0107858, 0.0127897
7: -0.0202115, -0.0047925, -0.0226383, -0.0044196, -0.0157919, 0.0178458
8: -0.0161425, 0.0025135, -0.0175194, 0.0060000, -0.0221424, 0.0200329
9: -0.0063050, 0.0057969, -0.0065382, 0.0090020, -0.0153070, 0.0123351

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0666382, upper bound: 0.0696392
time: 1.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0698969, upper bound: 0.0697898
time: 2.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0048045, 0.0097567, 0.0051588, 0.0089585, -0.0041541, 0.0045978
1: -0.0048993, 0.0089195, -0.0038732, 0.0026258, -0.0075251, 0.0127927
2: 0.0074419, 0.0293054, 0.0097897, 0.0282923, -0.0208504, 0.0195157
3: -0.0096881, -0.0019959, -0.0065978, -0.0024105, -0.0072776, 0.0046019
4: -0.0082701, 0.0067226, -0.0052683, 0.0049034, -0.0131735, 0.0119909
5: -0.0069998, 0.0099765, -0.0031402, 0.0079076, -0.0149074, 0.0131167
6: 0.9793062, 0.9925203, 0.9821798, 0.9925488, -0.0132426, 0.0103405
7: -0.0231209, -0.0046950, -0.0197363, -0.0045069, -0.0186140, 0.0150413
8: -0.0177931, 0.0066932, -0.0158729, 0.0018309, -0.0196240, 0.0225660
9: -0.0063660, 0.0096392, -0.0064836, 0.0051694, -0.0115354, 0.0161229

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0658947, upper bound: 0.0696208
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695360, upper bound: 0.0697890
time: 1.66 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0048045, 0.0097567, 0.0048550, 0.0096429, -0.0048384, 0.0049017
1: -0.0048993, 0.0089195, -0.0047530, 0.0080223, -0.0129215, 0.0136725
2: 0.0074419, 0.0293054, 0.0082823, 0.0291610, -0.0217191, 0.0210231
3: -0.0096881, -0.0019959, -0.0092475, -0.0020551, -0.0076331, 0.0072516
4: -0.0082701, 0.0067226, -0.0078421, 0.0063528, -0.0146229, 0.0145647
5: -0.0069998, 0.0099765, -0.0064496, 0.0096816, -0.0166814, 0.0164261
6: 0.9793062, 0.9925203, 0.9797159, 0.9925621, -0.0132559, 0.0128044
7: -0.0231209, -0.0046950, -0.0226383, -0.0044196, -0.0187013, 0.0179433
8: -0.0177931, 0.0066932, -0.0175194, 0.0060000, -0.0237931, 0.0242125
9: -0.0063660, 0.0096392, -0.0065382, 0.0090020, -0.0153680, 0.0161775

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0658947, upper bound: 0.0696207
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695360, upper bound: 0.0697893
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0051588, 0.0089585, 0.0051091, 0.0090706, -0.0039118, 0.0038494
1: -0.0038732, 0.0026258, -0.0040172, 0.0035094, -0.0073826, 0.0066430
2: 0.0097897, 0.0282923, 0.0101538, 0.0284345, -0.0186449, 0.0181385
3: -0.0065978, -0.0024105, -0.0070317, -0.0023523, -0.0042456, 0.0046212
4: -0.0052683, 0.0049034, -0.0056897, 0.0047456, -0.0100139, 0.0105931
5: -0.0031402, 0.0079076, -0.0036821, 0.0081981, -0.0113383, 0.0115897
6: 0.9821798, 0.9925488, 0.9817764, 0.9925056, -0.0103257, 0.0107725
7: -0.0197363, -0.0045069, -0.0202115, -0.0047925, -0.0149438, 0.0157046
8: -0.0158729, 0.0018309, -0.0161425, 0.0025135, -0.0183864, 0.0179733
9: -0.0064836, 0.0051694, -0.0063050, 0.0057969, -0.0122805, 0.0114744

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0663205, upper bound: 0.0692421
time: 1.59 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0699203, upper bound: 0.0695364
time: 1.88 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0051588, 0.0089585, 0.0048045, 0.0097567, -0.0045978, 0.0041541
1: -0.0038732, 0.0026258, -0.0048993, 0.0089195, -0.0127927, 0.0075251
2: 0.0097897, 0.0282923, 0.0074419, 0.0293054, -0.0195157, 0.0208504
3: -0.0065978, -0.0024105, -0.0096881, -0.0019959, -0.0046019, 0.0072776
4: -0.0052683, 0.0049034, -0.0082701, 0.0067226, -0.0119909, 0.0131735
5: -0.0031402, 0.0079076, -0.0069998, 0.0099765, -0.0131167, 0.0149074
6: 0.9821798, 0.9925488, 0.9793062, 0.9925203, -0.0103405, 0.0132426
7: -0.0197363, -0.0045069, -0.0231209, -0.0046950, -0.0150413, 0.0186140
8: -0.0158729, 0.0018309, -0.0177931, 0.0066932, -0.0225660, 0.0196240
9: -0.0064836, 0.0051694, -0.0063660, 0.0096392, -0.0161229, 0.0115354

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0663205, upper bound: 0.0692419
time: 1.66 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0699203, upper bound: 0.0695365
time: 2.14 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0048550, 0.0096429, 0.0051091, 0.0090706, -0.0042156, 0.0045338
1: -0.0047530, 0.0080223, -0.0040172, 0.0035094, -0.0082624, 0.0120395
2: 0.0082823, 0.0291610, 0.0101538, 0.0284345, -0.0201522, 0.0190072
3: -0.0092475, -0.0020551, -0.0070317, -0.0023523, -0.0068952, 0.0049767
4: -0.0078421, 0.0063528, -0.0056897, 0.0047456, -0.0125877, 0.0120425
5: -0.0064496, 0.0096816, -0.0036821, 0.0081981, -0.0146477, 0.0133636
6: 0.9797159, 0.9925621, 0.9817764, 0.9925056, -0.0127897, 0.0107858
7: -0.0226383, -0.0044196, -0.0202115, -0.0047925, -0.0178458, 0.0157919
8: -0.0175194, 0.0060000, -0.0161425, 0.0025135, -0.0200329, 0.0221424
9: -0.0065382, 0.0090020, -0.0063050, 0.0057969, -0.0123351, 0.0153070

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656996, upper bound: 0.0692385
time: 1.48 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697890, upper bound: 0.0695359
time: 2.06 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0048550, 0.0096429, 0.0048045, 0.0097567, -0.0049017, 0.0048384
1: -0.0047530, 0.0080223, -0.0048993, 0.0089195, -0.0136725, 0.0129215
2: 0.0082823, 0.0291610, 0.0074419, 0.0293054, -0.0210231, 0.0217191
3: -0.0092475, -0.0020551, -0.0096881, -0.0019959, -0.0072516, 0.0076331
4: -0.0078421, 0.0063528, -0.0082701, 0.0067226, -0.0145647, 0.0146229
5: -0.0064496, 0.0096816, -0.0069998, 0.0099765, -0.0164261, 0.0166814
6: 0.9797159, 0.9925621, 0.9793062, 0.9925203, -0.0128044, 0.0132559
7: -0.0226383, -0.0044196, -0.0231209, -0.0046950, -0.0179433, 0.0187013
8: -0.0175194, 0.0060000, -0.0177931, 0.0066932, -0.0242125, 0.0237931
9: -0.0065382, 0.0090020, -0.0063660, 0.0096392, -0.0161775, 0.0153680

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656996, upper bound: 0.0692385
time: 1.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697890, upper bound: 0.0695361
time: 1.53 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0051588, 0.0089585, 0.0051588, 0.0089585, -0.0037997, 0.0037997
1: -0.0038732, 0.0026258, -0.0038732, 0.0026258, -0.0064990, 0.0064990
2: 0.0097897, 0.0282923, 0.0097897, 0.0282923, -0.0185027, 0.0185027
3: -0.0065978, -0.0024105, -0.0065978, -0.0024105, -0.0041874, 0.0041874
4: -0.0052683, 0.0049034, -0.0052683, 0.0049034, -0.0101717, 0.0101717
5: -0.0031402, 0.0079076, -0.0031402, 0.0079076, -0.0110478, 0.0110478
6: 0.9821798, 0.9925488, 0.9821798, 0.9925488, -0.0103690, 0.0103690
7: -0.0197363, -0.0045069, -0.0197363, -0.0045069, -0.0152294, 0.0152294
8: -0.0158729, 0.0018309, -0.0158729, 0.0018309, -0.0177037, 0.0177037
9: -0.0064836, 0.0051694, -0.0064836, 0.0051694, -0.0116530, 0.0116530

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0663208, upper bound: 0.0694911
time: 1.94 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0699209, upper bound: 0.0697882
time: 1.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0051588, 0.0089585, 0.0048550, 0.0096429, -0.0044840, 0.0041035
1: -0.0038732, 0.0026258, -0.0047530, 0.0080223, -0.0118954, 0.0073788
2: 0.0097897, 0.0282923, 0.0082823, 0.0291610, -0.0193713, 0.0200100
3: -0.0065978, -0.0024105, -0.0092475, -0.0020551, -0.0045428, 0.0068370
4: -0.0052683, 0.0049034, -0.0078421, 0.0063528, -0.0116211, 0.0127455
5: -0.0031402, 0.0079076, -0.0064496, 0.0096816, -0.0128218, 0.0143572
6: 0.9821798, 0.9925488, 0.9797159, 0.9925621, -0.0103823, 0.0128329
7: -0.0197363, -0.0045069, -0.0226383, -0.0044196, -0.0153167, 0.0181315
8: -0.0158729, 0.0018309, -0.0175194, 0.0060000, -0.0218728, 0.0193502
9: -0.0064836, 0.0051694, -0.0065382, 0.0090020, -0.0154856, 0.0117076

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0663208, upper bound: 0.0694912
time: 1.62 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0699209, upper bound: 0.0697884
time: 2.02 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0048550, 0.0096429, 0.0051588, 0.0089585, -0.0041035, 0.0044840
1: -0.0047530, 0.0080223, -0.0038732, 0.0026258, -0.0073788, 0.0118954
2: 0.0082823, 0.0291610, 0.0097897, 0.0282923, -0.0200100, 0.0193713
3: -0.0092475, -0.0020551, -0.0065978, -0.0024105, -0.0068370, 0.0045428
4: -0.0078421, 0.0063528, -0.0052683, 0.0049034, -0.0127455, 0.0116211
5: -0.0064496, 0.0096816, -0.0031402, 0.0079076, -0.0143572, 0.0128218
6: 0.9797159, 0.9925621, 0.9821798, 0.9925488, -0.0128329, 0.0103823
7: -0.0226383, -0.0044196, -0.0197363, -0.0045069, -0.0181315, 0.0153167
8: -0.0175194, 0.0060000, -0.0158729, 0.0018309, -0.0193502, 0.0218728
9: -0.0065382, 0.0090020, -0.0064836, 0.0051694, -0.0117076, 0.0154856

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656999, upper bound: 0.0694842
time: 1.61 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697893, upper bound: 0.0697882
time: 1.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0048550, 0.0096429, 0.0048550, 0.0096429, -0.0047879, 0.0047879
1: -0.0047530, 0.0080223, -0.0047530, 0.0080223, -0.0127752, 0.0127752
2: 0.0082823, 0.0291610, 0.0082823, 0.0291610, -0.0208786, 0.0208786
3: -0.0092475, -0.0020551, -0.0092475, -0.0020551, -0.0071925, 0.0071925
4: -0.0078421, 0.0063528, -0.0078421, 0.0063528, -0.0141949, 0.0141949
5: -0.0064496, 0.0096816, -0.0064496, 0.0096816, -0.0161311, 0.0161311
6: 0.9797159, 0.9925621, 0.9797159, 0.9925621, -0.0128462, 0.0128462
7: -0.0226383, -0.0044196, -0.0226383, -0.0044196, -0.0182188, 0.0182188
8: -0.0175194, 0.0060000, -0.0175194, 0.0060000, -0.0235193, 0.0235193
9: -0.0065382, 0.0090020, -0.0065382, 0.0090020, -0.0155402, 0.0155402

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656999, upper bound: 0.0694842
time: 1.99 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697893, upper bound: 0.0697878
time: 1.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.63 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0666429, upper bound: 0.0693945
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0699007, upper bound: 0.0695403
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0666429, upper bound: 0.0693945
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0699007, upper bound: 0.0695405
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0658996, upper bound: 0.0693785
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0695395, upper bound: 0.0695396
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0658996, upper bound: 0.0693787
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0695395, upper bound: 0.0695395
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0666382, upper bound: 0.0696392
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0698969, upper bound: 0.0697903
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0666382, upper bound: 0.0696392
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0698969, upper bound: 0.0697898
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0658947, upper bound: 0.0696208
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0695360, upper bound: 0.0697890
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0658947, upper bound: 0.0696207
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0695360, upper bound: 0.0697893
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0663205, upper bound: 0.0692421
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0699203, upper bound: 0.0695364
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0663205, upper bound: 0.0692419
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0699203, upper bound: 0.0695365
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0656996, upper bound: 0.0692385
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0697890, upper bound: 0.0695359
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0656996, upper bound: 0.0692385
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0697890, upper bound: 0.0695361
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0663208, upper bound: 0.0694911
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0699209, upper bound: 0.0697882
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0663208, upper bound: 0.0694912
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0699209, upper bound: 0.0697884
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0656999, upper bound: 0.0694842
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0697893, upper bound: 0.0697882
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0656999, upper bound: 0.0694842
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 6, lower bound: -0.0697893, upper bound: 0.0697878

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0055103, 0.0081807, 0.0051768, 0.0089181, -0.0034078, 0.0030039
1: -0.0028554, 0.0008601, -0.0038212, 0.0023070, -0.0051624, 0.0046813
2: 0.0099901, 0.0272875, 0.0101589, 0.0282410, -0.0182509, 0.0171286
3: -0.0047990, -0.0028216, -0.0064413, -0.0024315, -0.0023675, 0.0036197
4: -0.0032308, 0.0048165, -0.0051162, 0.0047434, -0.0079741, 0.0099327
5: -0.0012388, 0.0058556, -0.0029447, 0.0078028, -0.0090415, 0.0088003
6: 0.9845367, 0.9925250, 0.9823254, 0.9925051, -0.0079684, 0.0101997
7: -0.0174291, -0.0046641, -0.0195648, -0.0047965, -0.0126327, 0.0149007
8: -0.0139683, -0.0004729, -0.0157756, 0.0015845, -0.0155529, 0.0153027
9: -0.0063853, 0.0024155, -0.0063025, 0.0049429, -0.0113282, 0.0087180

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661517, upper bound: 0.0697144
time: 1.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661516, upper bound: 0.0693127
time: 1.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0052724, 0.0087028, 0.0051091, 0.0090706, -0.0037982, 0.0035937
1: -0.0035443, 0.0008382, -0.0040172, 0.0035094, -0.0070537, 0.0048554
2: 0.0101671, 0.0279676, 0.0101538, 0.0284345, -0.0182675, 0.0178138
3: -0.0056075, -0.0025433, -0.0070317, -0.0023523, -0.0032553, 0.0044884
4: -0.0043063, 0.0047398, -0.0056897, 0.0047456, -0.0090519, 0.0104296
5: -0.0019033, 0.0072446, -0.0036821, 0.0081981, -0.0101014, 0.0109267
6: 0.9831008, 0.9925041, 0.9817764, 0.9925056, -0.0094048, 0.0107277
7: -0.0186516, -0.0048029, -0.0202115, -0.0047925, -0.0138591, 0.0154085
8: -0.0152575, 0.0002727, -0.0161425, 0.0025135, -0.0177710, 0.0164152
9: -0.0062985, 0.0037369, -0.0063050, 0.0057969, -0.0120954, 0.0100419

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0698102, upper bound: 0.0666474
time: 1.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0698102, upper bound: 0.0699049
time: 1.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0055103, 0.0081807, 0.0048710, 0.0096068, -0.0040965, 0.0033097
1: -0.0028554, 0.0008601, -0.0047066, 0.0077375, -0.0105929, 0.0055667
2: 0.0099901, 0.0272875, 0.0085490, 0.0291151, -0.0191251, 0.0187385
3: -0.0047990, -0.0028216, -0.0091077, -0.0020738, -0.0027252, 0.0062861
4: -0.0032308, 0.0048165, -0.0077063, 0.0062354, -0.0094662, 0.0125229
5: -0.0012388, 0.0058556, -0.0062750, 0.0095880, -0.0108267, 0.0121306
6: 0.9845367, 0.9925250, 0.9798459, 0.9925198, -0.0079831, 0.0126791
7: -0.0174291, -0.0046641, -0.0224852, -0.0046989, -0.0127302, 0.0178211
8: -0.0139683, -0.0004729, -0.0174325, 0.0057800, -0.0197483, 0.0169596
9: -0.0063853, 0.0024155, -0.0063635, 0.0087998, -0.0151851, 0.0087790

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661475, upper bound: 0.0693098
time: 1.49 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661477, upper bound: 0.0691154
time: 1.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0052724, 0.0087028, 0.0048045, 0.0097567, -0.0044843, 0.0038983
1: -0.0035443, 0.0008382, -0.0048993, 0.0089195, -0.0124638, 0.0057375
2: 0.0101671, 0.0279676, 0.0074419, 0.0293054, -0.0191383, 0.0205257
3: -0.0056075, -0.0025433, -0.0096881, -0.0019959, -0.0036116, 0.0071448
4: -0.0043063, 0.0047398, -0.0082701, 0.0067226, -0.0110289, 0.0130099
5: -0.0019033, 0.0072446, -0.0069998, 0.0099765, -0.0118798, 0.0142444
6: 0.9831008, 0.9925041, 0.9793062, 0.9925203, -0.0094195, 0.0131978
7: -0.0186516, -0.0048029, -0.0231209, -0.0046950, -0.0139567, 0.0183179
8: -0.0152575, 0.0002727, -0.0177931, 0.0066932, -0.0219507, 0.0180658
9: -0.0062985, 0.0037369, -0.0063660, 0.0096392, -0.0159378, 0.0101029

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697507, upper bound: 0.0659002
time: 1.64 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697507, upper bound: 0.0695405
time: 1.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0052009, 0.0088638, 0.0051768, 0.0089181, -0.0037172, 0.0036870
1: -0.0037513, 0.0018787, -0.0038212, 0.0023070, -0.0060583, 0.0056999
2: 0.0098702, 0.0281720, 0.0101589, 0.0282410, -0.0183708, 0.0180132
3: -0.0062310, -0.0024597, -0.0064413, -0.0024315, -0.0037995, 0.0039816
4: -0.0049120, 0.0048685, -0.0051162, 0.0047434, -0.0096553, 0.0099847
5: -0.0026821, 0.0076620, -0.0029447, 0.0078028, -0.0104848, 0.0106067
6: 0.9825209, 0.9925392, 0.9823254, 0.9925051, -0.0099842, 0.0102139
7: -0.0193345, -0.0045700, -0.0195648, -0.0047965, -0.0145380, 0.0149948
8: -0.0156449, 0.0012537, -0.0157756, 0.0015845, -0.0172294, 0.0170293
9: -0.0064441, 0.0046388, -0.0063025, 0.0049429, -0.0113870, 0.0109413

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656313, upper bound: 0.0696826
time: 1.52 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656312, upper bound: 0.0692817
time: 1.59 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0049721, 0.0093792, 0.0051091, 0.0090706, -0.0040985, 0.0042701
1: -0.0044140, 0.0059430, -0.0040172, 0.0035094, -0.0079234, 0.0099602
2: 0.0100427, 0.0288263, 0.0101538, 0.0284345, -0.0183918, 0.0186725
3: -0.0082266, -0.0021920, -0.0070317, -0.0023523, -0.0058743, 0.0048397
4: -0.0068504, 0.0054958, -0.0056897, 0.0047456, -0.0115960, 0.0111856
5: -0.0051745, 0.0089981, -0.0036821, 0.0081981, -0.0133725, 0.0126801
6: 0.9806653, 0.9925188, 0.9817764, 0.9925056, -0.0118402, 0.0107424
7: -0.0215201, -0.0047054, -0.0202115, -0.0047925, -0.0167276, 0.0155061
8: -0.0168850, 0.0043936, -0.0161425, 0.0025135, -0.0193985, 0.0205361
9: -0.0063595, 0.0075252, -0.0063050, 0.0057969, -0.0121564, 0.0138303

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0693945, upper bound: 0.0666427
time: 1.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0693945, upper bound: 0.0699007
time: 1.77 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0052009, 0.0088638, 0.0048710, 0.0096068, -0.0044059, 0.0039928
1: -0.0037513, 0.0018787, -0.0047066, 0.0077375, -0.0114888, 0.0065853
2: 0.0098702, 0.0281720, 0.0085490, 0.0291151, -0.0192450, 0.0196230
3: -0.0062310, -0.0024597, -0.0091077, -0.0020738, -0.0041572, 0.0066480
4: -0.0049120, 0.0048685, -0.0077063, 0.0062354, -0.0111474, 0.0125748
5: -0.0026821, 0.0076620, -0.0062750, 0.0095880, -0.0122700, 0.0139370
6: 0.9825209, 0.9925392, 0.9798459, 0.9925198, -0.0099989, 0.0126933
7: -0.0193345, -0.0045700, -0.0224852, -0.0046989, -0.0146355, 0.0179152
8: -0.0156449, 0.0012537, -0.0174325, 0.0057800, -0.0214249, 0.0186862
9: -0.0064441, 0.0046388, -0.0063635, 0.0087998, -0.0152439, 0.0110023

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656308, upper bound: 0.0693099
time: 1.56 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656307, upper bound: 0.0691108
time: 1.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0049721, 0.0093792, 0.0048045, 0.0097567, -0.0047846, 0.0045747
1: -0.0044140, 0.0059430, -0.0048993, 0.0089195, -0.0133335, 0.0108422
2: 0.0100427, 0.0288263, 0.0074419, 0.0293054, -0.0192627, 0.0213844
3: -0.0082266, -0.0021920, -0.0096881, -0.0019959, -0.0062306, 0.0074961
4: -0.0068504, 0.0054958, -0.0082701, 0.0067226, -0.0135730, 0.0137659
5: -0.0051745, 0.0089981, -0.0069998, 0.0099765, -0.0151510, 0.0159979
6: 0.9806653, 0.9925188, 0.9793062, 0.9925203, -0.0118549, 0.0132126
7: -0.0215201, -0.0047054, -0.0231209, -0.0046950, -0.0168252, 0.0184155
8: -0.0168850, 0.0043936, -0.0177931, 0.0066932, -0.0235781, 0.0221867
9: -0.0063595, 0.0075252, -0.0063660, 0.0096392, -0.0159987, 0.0138913

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0693788, upper bound: 0.0659000
time: 3.38 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0693788, upper bound: 0.0695396
time: 1.61 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0055103, 0.0081807, 0.0052253, 0.0088088, -0.0032984, 0.0029554
1: -0.0028554, 0.0008601, -0.0036806, 0.0014449, -0.0043003, 0.0045408
2: 0.0099901, 0.0272875, 0.0097949, 0.0281022, -0.0181122, 0.0174927
3: -0.0047990, -0.0028216, -0.0060180, -0.0024882, -0.0023108, 0.0031964
4: -0.0032308, 0.0048165, -0.0047051, 0.0049011, -0.0081319, 0.0095216
5: -0.0012388, 0.0058556, -0.0024160, 0.0075194, -0.0087581, 0.0082716
6: 0.9845367, 0.9925250, 0.9827189, 0.9925482, -0.0080115, 0.0098062
7: -0.0174291, -0.0046641, -0.0191012, -0.0045110, -0.0129182, 0.0144371
8: -0.0139683, -0.0004729, -0.0155126, 0.0009185, -0.0148869, 0.0150397
9: -0.0063853, 0.0024155, -0.0064811, 0.0043306, -0.0107160, 0.0088966

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661478, upper bound: 0.0697398
time: 1.42 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661478, upper bound: 0.0695630
time: 2.06 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0052724, 0.0087028, 0.0051588, 0.0089585, -0.0036861, 0.0035439
1: -0.0035443, 0.0008382, -0.0038732, 0.0026258, -0.0061701, 0.0047114
2: 0.0101671, 0.0279676, 0.0097897, 0.0282923, -0.0181252, 0.0181780
3: -0.0056075, -0.0025433, -0.0065978, -0.0024105, -0.0031971, 0.0040545
4: -0.0043063, 0.0047398, -0.0052683, 0.0049034, -0.0092097, 0.0100081
5: -0.0019033, 0.0072446, -0.0031402, 0.0079076, -0.0098109, 0.0103848
6: 0.9831008, 0.9925041, 0.9821798, 0.9925488, -0.0094481, 0.0103242
7: -0.0186516, -0.0048029, -0.0197363, -0.0045069, -0.0141448, 0.0149334
8: -0.0152575, 0.0002727, -0.0158729, 0.0018309, -0.0170884, 0.0161455
9: -0.0062985, 0.0037369, -0.0064836, 0.0051694, -0.0114679, 0.0102205

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0696451, upper bound: 0.0663261
time: 1.51 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0696451, upper bound: 0.0699232
time: 2.03 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0055103, 0.0081807, 0.0049210, 0.0094942, -0.0039839, 0.0032597
1: -0.0028554, 0.0008601, -0.0045618, 0.0068500, -0.0097054, 0.0054220
2: 0.0099901, 0.0272875, 0.0093803, 0.0289723, -0.0189822, 0.0179072
3: -0.0047990, -0.0028216, -0.0086720, -0.0021322, -0.0026668, 0.0058503
4: -0.0032308, 0.0048165, -0.0072830, 0.0058696, -0.0091004, 0.0120996
5: -0.0012388, 0.0058556, -0.0057307, 0.0092962, -0.0105350, 0.0115863
6: 0.9845367, 0.9925250, 0.9802511, 0.9925613, -0.0080246, 0.0122739
7: -0.0174291, -0.0046641, -0.0220079, -0.0044236, -0.0130056, 0.0173438
8: -0.0139683, -0.0004729, -0.0171617, 0.0050943, -0.0190627, 0.0166888
9: -0.0063853, 0.0024155, -0.0065357, 0.0081695, -0.0145548, 0.0089512

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661439, upper bound: 0.0695924
time: 1.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661444, upper bound: 0.0695040
time: 1.67 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0052724, 0.0087028, 0.0048550, 0.0096429, -0.0043705, 0.0038478
1: -0.0035443, 0.0008382, -0.0047530, 0.0080223, -0.0115666, 0.0055912
2: 0.0101671, 0.0279676, 0.0082823, 0.0291610, -0.0189939, 0.0196853
3: -0.0056075, -0.0025433, -0.0092475, -0.0020551, -0.0035525, 0.0067042
4: -0.0043063, 0.0047398, -0.0078421, 0.0063528, -0.0106591, 0.0125820
5: -0.0019033, 0.0072446, -0.0064496, 0.0096816, -0.0115849, 0.0136942
6: 0.9831008, 0.9925041, 0.9797159, 0.9925621, -0.0094613, 0.0127882
7: -0.0186516, -0.0048029, -0.0226383, -0.0044196, -0.0142321, 0.0178354
8: -0.0152575, 0.0002727, -0.0175194, 0.0060000, -0.0212575, 0.0177920
9: -0.0062985, 0.0037369, -0.0065382, 0.0090020, -0.0153005, 0.0102751

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695671, upper bound: 0.0657002
time: 1.54 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695671, upper bound: 0.0657004
time: 1.72 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0052009, 0.0088638, 0.0052253, 0.0088088, -0.0036079, 0.0036385
1: -0.0037513, 0.0018787, -0.0036806, 0.0014449, -0.0051962, 0.0055593
2: 0.0098702, 0.0281720, 0.0097949, 0.0281022, -0.0182321, 0.0183772
3: -0.0062310, -0.0024597, -0.0060180, -0.0024882, -0.0037428, 0.0035583
4: -0.0049120, 0.0048685, -0.0047051, 0.0049011, -0.0098131, 0.0095735
5: -0.0026821, 0.0076620, -0.0024160, 0.0075194, -0.0102014, 0.0100780
6: 0.9825209, 0.9925392, 0.9827189, 0.9925482, -0.0100273, 0.0098203
7: -0.0193345, -0.0045700, -0.0191012, -0.0045110, -0.0148235, 0.0145312
8: -0.0156449, 0.0012537, -0.0155126, 0.0009185, -0.0165634, 0.0167663
9: -0.0064441, 0.0046388, -0.0064811, 0.0043306, -0.0107748, 0.0111198

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656270, upper bound: 0.0697200
time: 1.56 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656269, upper bound: 0.0695387
time: 1.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0049721, 0.0093792, 0.0051588, 0.0089585, -0.0039865, 0.0042204
1: -0.0044140, 0.0059430, -0.0038732, 0.0026258, -0.0070398, 0.0098161
2: 0.0100427, 0.0288263, 0.0097897, 0.0282923, -0.0182496, 0.0190366
3: -0.0082266, -0.0021920, -0.0065978, -0.0024105, -0.0058161, 0.0044059
4: -0.0068504, 0.0054958, -0.0052683, 0.0049034, -0.0117538, 0.0107641
5: -0.0051745, 0.0089981, -0.0031402, 0.0079076, -0.0130821, 0.0121383
6: 0.9806653, 0.9925188, 0.9821798, 0.9925488, -0.0118835, 0.0103390
7: -0.0215201, -0.0047054, -0.0197363, -0.0045069, -0.0170133, 0.0150309
8: -0.0168850, 0.0043936, -0.0158729, 0.0018309, -0.0187158, 0.0202665
9: -0.0063595, 0.0075252, -0.0064836, 0.0051694, -0.0115289, 0.0140089

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0692422, upper bound: 0.0663209
time: 1.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0692422, upper bound: 0.0699206
time: 1.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0052009, 0.0088638, 0.0049210, 0.0094942, -0.0042933, 0.0039428
1: -0.0037513, 0.0018787, -0.0045618, 0.0068500, -0.0106014, 0.0064406
2: 0.0098702, 0.0281720, 0.0093803, 0.0289723, -0.0191021, 0.0187918
3: -0.0062310, -0.0024597, -0.0086720, -0.0021322, -0.0040988, 0.0062123
4: -0.0049120, 0.0048685, -0.0072830, 0.0058696, -0.0107816, 0.0121515
5: -0.0026821, 0.0076620, -0.0057307, 0.0092962, -0.0119783, 0.0133927
6: 0.9825209, 0.9925392, 0.9802511, 0.9925613, -0.0100405, 0.0122881
7: -0.0193345, -0.0045700, -0.0220079, -0.0044236, -0.0149109, 0.0174379
8: -0.0156449, 0.0012537, -0.0171617, 0.0050943, -0.0207393, 0.0184154
9: -0.0064441, 0.0046388, -0.0065357, 0.0081695, -0.0146136, 0.0111745

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656266, upper bound: 0.0695908
time: 1.51 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656267, upper bound: 0.0694958
time: 1.56 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0049721, 0.0093792, 0.0048550, 0.0096429, -0.0046708, 0.0045242
1: -0.0044140, 0.0059430, -0.0047530, 0.0080223, -0.0124362, 0.0106959
2: 0.0100427, 0.0288263, 0.0082823, 0.0291610, -0.0191182, 0.0205439
3: -0.0082266, -0.0021920, -0.0092475, -0.0020551, -0.0061715, 0.0070555
4: -0.0068504, 0.0054958, -0.0078421, 0.0063528, -0.0132032, 0.0133380
5: -0.0051745, 0.0089981, -0.0064496, 0.0096816, -0.0148560, 0.0154476
6: 0.9806653, 0.9925188, 0.9797159, 0.9925621, -0.0118968, 0.0128029
7: -0.0215201, -0.0047054, -0.0226383, -0.0044196, -0.0171006, 0.0179329
8: -0.0168850, 0.0043936, -0.0175194, 0.0060000, -0.0228849, 0.0219130
9: -0.0063595, 0.0075252, -0.0065382, 0.0090020, -0.0153615, 0.0140635

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0692385, upper bound: 0.0656993
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0692385, upper bound: 0.0656999
time: 1.70 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0055544, 0.0080854, 0.0051768, 0.0089181, -0.0033637, 0.0029086
1: -0.0027278, 0.0009054, -0.0038212, 0.0023070, -0.0050347, 0.0047266
2: 0.0096249, 0.0271615, 0.0101589, 0.0282410, -0.0186161, 0.0170026
3: -0.0047817, -0.0028732, -0.0064413, -0.0024315, -0.0023503, 0.0035681
4: -0.0031298, 0.0049748, -0.0051162, 0.0047434, -0.0078732, 0.0100910
5: -0.0012624, 0.0055982, -0.0029447, 0.0078028, -0.0090652, 0.0085429
6: 0.9847512, 0.9925683, 0.9823254, 0.9925051, -0.0077539, 0.0102429
7: -0.0173125, -0.0043776, -0.0195648, -0.0047965, -0.0125160, 0.0151872
8: -0.0137295, -0.0003831, -0.0157756, 0.0015845, -0.0153140, 0.0153924
9: -0.0065645, 0.0023464, -0.0063025, 0.0049429, -0.0115074, 0.0086489

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661574, upper bound: 0.0695933
time: 1.42 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661574, upper bound: 0.0692265
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0053224, 0.0085902, 0.0051091, 0.0090706, -0.0037482, 0.0034811
1: -0.0033996, 0.0008834, -0.0040172, 0.0035094, -0.0069090, 0.0049006
2: 0.0098025, 0.0278248, 0.0101538, 0.0284345, -0.0186320, 0.0176710
3: -0.0051717, -0.0026018, -0.0070317, -0.0023523, -0.0028194, 0.0044299
4: -0.0038829, 0.0048978, -0.0056897, 0.0047456, -0.0086285, 0.0105875
5: -0.0013589, 0.0069528, -0.0036821, 0.0081981, -0.0095570, 0.0106348
6: 0.9835059, 0.9925473, 0.9817764, 0.9925056, -0.0089996, 0.0107709
7: -0.0181742, -0.0045170, -0.0202115, -0.0047925, -0.0133817, 0.0156945
8: -0.0149867, -0.0004131, -0.0161425, 0.0025135, -0.0175002, 0.0157293
9: -0.0064773, 0.0031064, -0.0063050, 0.0057969, -0.0122742, 0.0094115

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697953, upper bound: 0.0666423
time: 1.81 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697953, upper bound: 0.0699004
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0055544, 0.0080854, 0.0048710, 0.0096068, -0.0040524, 0.0032144
1: -0.0027278, 0.0009054, -0.0047066, 0.0077375, -0.0104653, 0.0056120
2: 0.0096249, 0.0271615, 0.0085490, 0.0291151, -0.0194903, 0.0186124
3: -0.0047817, -0.0028732, -0.0091077, -0.0020738, -0.0027079, 0.0062345
4: -0.0031298, 0.0049748, -0.0077063, 0.0062354, -0.0093653, 0.0126811
5: -0.0012624, 0.0055982, -0.0062750, 0.0095880, -0.0108503, 0.0118732
6: 0.9847512, 0.9925683, 0.9798459, 0.9925198, -0.0077686, 0.0127223
7: -0.0173125, -0.0043776, -0.0224852, -0.0046989, -0.0126135, 0.0181076
8: -0.0137295, -0.0003831, -0.0174325, 0.0057800, -0.0195094, 0.0170493
9: -0.0065645, 0.0023464, -0.0063635, 0.0087998, -0.0153642, 0.0087099

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661534, upper bound: 0.0692179
time: 1.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661535, upper bound: 0.0690317
time: 1.42 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0053224, 0.0085902, 0.0048045, 0.0097567, -0.0044343, 0.0037857
1: -0.0033996, 0.0008834, -0.0048993, 0.0089195, -0.0123191, 0.0057826
2: 0.0098025, 0.0278248, 0.0074419, 0.0293054, -0.0195029, 0.0203829
3: -0.0051717, -0.0026018, -0.0096881, -0.0019959, -0.0031757, 0.0070863
4: -0.0038829, 0.0048978, -0.0082701, 0.0067226, -0.0106055, 0.0131679
5: -0.0013589, 0.0069528, -0.0069998, 0.0099765, -0.0113355, 0.0139526
6: 0.9835059, 0.9925473, 0.9793062, 0.9925203, -0.0090144, 0.0132411
7: -0.0181742, -0.0045170, -0.0231209, -0.0046950, -0.0134793, 0.0186039
8: -0.0149867, -0.0004131, -0.0177931, 0.0066932, -0.0216798, 0.0173800
9: -0.0064773, 0.0031064, -0.0063660, 0.0096392, -0.0161166, 0.0094724

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697451, upper bound: 0.0658952
time: 1.45 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697451, upper bound: 0.0695364
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0052499, 0.0087534, 0.0051768, 0.0089181, -0.0036682, 0.0035766
1: -0.0036094, 0.0010080, -0.0038212, 0.0023070, -0.0059164, 0.0048292
2: 0.0095092, 0.0280319, 0.0101589, 0.0282410, -0.0187318, 0.0178730
3: -0.0058035, -0.0025170, -0.0064413, -0.0024315, -0.0033720, 0.0039243
4: -0.0044967, 0.0050249, -0.0051162, 0.0047434, -0.0092401, 0.0101411
5: -0.0021481, 0.0073758, -0.0029447, 0.0078028, -0.0099509, 0.0103205
6: 0.9829183, 0.9925821, 0.9823254, 0.9925051, -0.0095868, 0.0102568
7: -0.0188663, -0.0042869, -0.0195648, -0.0047965, -0.0140698, 0.0152779
8: -0.0153793, 0.0005810, -0.0157756, 0.0015845, -0.0169638, 0.0163566
9: -0.0066212, 0.0040204, -0.0063025, 0.0049429, -0.0115641, 0.0103229

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656892, upper bound: 0.0695382
time: 1.53 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656894, upper bound: 0.0691907
time: 1.46 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0050248, 0.0092606, 0.0051091, 0.0090706, -0.0040458, 0.0041515
1: -0.0042614, 0.0050073, -0.0040172, 0.0035094, -0.0077708, 0.0090245
2: 0.0096915, 0.0286757, 0.0101538, 0.0284345, -0.0187430, 0.0185219
3: -0.0077672, -0.0022536, -0.0070317, -0.0023523, -0.0054149, 0.0047781
4: -0.0064042, 0.0051102, -0.0056897, 0.0047456, -0.0111497, 0.0107999
5: -0.0046007, 0.0086905, -0.0036821, 0.0081981, -0.0127988, 0.0123726
6: 0.9810924, 0.9925604, 0.9817764, 0.9925056, -0.0114132, 0.0107840
7: -0.0210170, -0.0044299, -0.0202115, -0.0047925, -0.0162245, 0.0157816
8: -0.0165995, 0.0036707, -0.0161425, 0.0025135, -0.0191130, 0.0198132
9: -0.0065318, 0.0068607, -0.0063050, 0.0057969, -0.0123287, 0.0131658

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0696393, upper bound: 0.0666381
time: 1.82 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0696393, upper bound: 0.0666382
time: 1.86 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0052499, 0.0087534, 0.0048710, 0.0096068, -0.0043568, 0.0038824
1: -0.0036094, 0.0010080, -0.0047066, 0.0077375, -0.0113469, 0.0057146
2: 0.0095092, 0.0280319, 0.0085490, 0.0291151, -0.0196059, 0.0194829
3: -0.0058035, -0.0025170, -0.0091077, -0.0020738, -0.0037297, 0.0065907
4: -0.0044967, 0.0050249, -0.0077063, 0.0062354, -0.0107321, 0.0127312
5: -0.0021481, 0.0073758, -0.0062750, 0.0095880, -0.0117360, 0.0136508
6: 0.9829183, 0.9925821, 0.9798459, 0.9925198, -0.0096015, 0.0127362
7: -0.0188663, -0.0042869, -0.0224852, -0.0046989, -0.0141673, 0.0181983
8: -0.0153793, 0.0005810, -0.0174325, 0.0057800, -0.0211593, 0.0180135
9: -0.0066212, 0.0040204, -0.0063635, 0.0087998, -0.0154210, 0.0103839

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656887, upper bound: 0.0692177
time: 1.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656883, upper bound: 0.0690253
time: 1.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0050248, 0.0092606, 0.0048045, 0.0097567, -0.0047319, 0.0044561
1: -0.0042614, 0.0050073, -0.0048993, 0.0089195, -0.0131809, 0.0099066
2: 0.0096915, 0.0286757, 0.0074419, 0.0293054, -0.0196139, 0.0212338
3: -0.0077672, -0.0022536, -0.0096881, -0.0019959, -0.0057712, 0.0074345
4: -0.0064042, 0.0051102, -0.0082701, 0.0067226, -0.0131268, 0.0133803
5: -0.0046007, 0.0086905, -0.0069998, 0.0099765, -0.0145772, 0.0156903
6: 0.9810924, 0.9925604, 0.9793062, 0.9925203, -0.0114279, 0.0132542
7: -0.0210170, -0.0044299, -0.0231209, -0.0046950, -0.0163220, 0.0186910
8: -0.0165995, 0.0036707, -0.0177931, 0.0066932, -0.0232926, 0.0214639
9: -0.0065318, 0.0068607, -0.0063660, 0.0096392, -0.0161710, 0.0132267

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0696209, upper bound: 0.0658951
time: 1.91 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0696209, upper bound: 0.0695359
time: 1.97 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0055544, 0.0080854, 0.0052253, 0.0088088, -0.0032544, 0.0028601
1: -0.0027278, 0.0009054, -0.0036806, 0.0014449, -0.0041726, 0.0045860
2: 0.0096249, 0.0271615, 0.0097949, 0.0281022, -0.0184774, 0.0173666
3: -0.0047817, -0.0028732, -0.0060180, -0.0024882, -0.0022935, 0.0031448
4: -0.0031298, 0.0049748, -0.0047051, 0.0049011, -0.0080309, 0.0096798
5: -0.0012624, 0.0055982, -0.0024160, 0.0075194, -0.0087818, 0.0080142
6: 0.9847512, 0.9925683, 0.9827189, 0.9925482, -0.0077971, 0.0098494
7: -0.0173125, -0.0043776, -0.0191012, -0.0045110, -0.0128015, 0.0147236
8: -0.0137295, -0.0003831, -0.0155126, 0.0009185, -0.0146480, 0.0151294
9: -0.0065645, 0.0023464, -0.0064811, 0.0043306, -0.0108951, 0.0088275

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661575, upper bound: 0.0696657
time: 1.62 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661575, upper bound: 0.0694472
time: 1.42 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0053224, 0.0085902, 0.0051588, 0.0089585, -0.0036361, 0.0034314
1: -0.0033996, 0.0008834, -0.0038732, 0.0026258, -0.0060254, 0.0047565
2: 0.0098025, 0.0278248, 0.0097897, 0.0282923, -0.0184898, 0.0180351
3: -0.0051717, -0.0026018, -0.0065978, -0.0024105, -0.0027612, 0.0039961
4: -0.0038829, 0.0048978, -0.0052683, 0.0049034, -0.0087863, 0.0101661
5: -0.0013589, 0.0069528, -0.0031402, 0.0079076, -0.0092665, 0.0100930
6: 0.9835059, 0.9925473, 0.9821798, 0.9925488, -0.0090429, 0.0103675
7: -0.0181742, -0.0045170, -0.0197363, -0.0045069, -0.0136674, 0.0152193
8: -0.0149867, -0.0004131, -0.0158729, 0.0018309, -0.0168175, 0.0154597
9: -0.0064773, 0.0031064, -0.0064836, 0.0051694, -0.0116467, 0.0095901

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697950, upper bound: 0.0663248
time: 1.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697950, upper bound: 0.0699216
time: 1.80 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0055544, 0.0080854, 0.0049210, 0.0094942, -0.0039398, 0.0031644
1: -0.0027278, 0.0009054, -0.0045618, 0.0068500, -0.0095778, 0.0054673
2: 0.0096249, 0.0271615, 0.0093803, 0.0289723, -0.0193474, 0.0177812
3: -0.0047817, -0.0028732, -0.0086720, -0.0021322, -0.0026495, 0.0057988
4: -0.0031298, 0.0049748, -0.0072830, 0.0058696, -0.0089995, 0.0122578
5: -0.0012624, 0.0055982, -0.0057307, 0.0092962, -0.0105586, 0.0113289
6: 0.9847512, 0.9925683, 0.9802511, 0.9925613, -0.0078102, 0.0123171
7: -0.0173125, -0.0043776, -0.0220079, -0.0044236, -0.0128889, 0.0176303
8: -0.0137295, -0.0003831, -0.0171617, 0.0050943, -0.0188238, 0.0167786
9: -0.0065645, 0.0023464, -0.0065357, 0.0081695, -0.0147339, 0.0088821

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661538, upper bound: 0.0694857
time: 1.52 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661539, upper bound: 0.0693811
time: 1.59 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0053224, 0.0085902, 0.0048550, 0.0096429, -0.0043205, 0.0037352
1: -0.0033996, 0.0008834, -0.0047530, 0.0080223, -0.0114218, 0.0056364
2: 0.0098025, 0.0278248, 0.0082823, 0.0291610, -0.0193584, 0.0195424
3: -0.0051717, -0.0026018, -0.0092475, -0.0020551, -0.0031166, 0.0066457
4: -0.0038829, 0.0048978, -0.0078421, 0.0063528, -0.0102357, 0.0127399
5: -0.0013589, 0.0069528, -0.0064496, 0.0096816, -0.0110405, 0.0134024
6: 0.9835059, 0.9925473, 0.9797159, 0.9925621, -0.0090562, 0.0128314
7: -0.0181742, -0.0045170, -0.0226383, -0.0044196, -0.0137547, 0.0181213
8: -0.0149867, -0.0004131, -0.0175194, 0.0060000, -0.0209866, 0.0171062
9: -0.0064773, 0.0031064, -0.0065382, 0.0090020, -0.0154793, 0.0096447

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697452, upper bound: 0.0656994
time: 1.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697452, upper bound: 0.0697884
time: 1.76 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0052499, 0.0087534, 0.0052253, 0.0088088, -0.0035588, 0.0035280
1: -0.0036094, 0.0010080, -0.0036806, 0.0014449, -0.0050543, 0.0046887
2: 0.0095092, 0.0280319, 0.0097949, 0.0281022, -0.0185930, 0.0182370
3: -0.0058035, -0.0025170, -0.0060180, -0.0024882, -0.0033153, 0.0035010
4: -0.0044967, 0.0050249, -0.0047051, 0.0049011, -0.0093978, 0.0097299
5: -0.0021481, 0.0073758, -0.0024160, 0.0075194, -0.0096675, 0.0097918
6: 0.9829183, 0.9925821, 0.9827189, 0.9925482, -0.0096299, 0.0098633
7: -0.0188663, -0.0042869, -0.0191012, -0.0045110, -0.0143553, 0.0148143
8: -0.0153793, 0.0005810, -0.0155126, 0.0009185, -0.0162978, 0.0160936
9: -0.0066212, 0.0040204, -0.0064811, 0.0043306, -0.0109518, 0.0105015

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656894, upper bound: 0.0696397
time: 6.31 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656893, upper bound: 0.0694220
time: 1.56 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0050248, 0.0092606, 0.0051588, 0.0089585, -0.0039338, 0.0041017
1: -0.0042614, 0.0050073, -0.0038732, 0.0026258, -0.0068872, 0.0088805
2: 0.0096915, 0.0286757, 0.0097897, 0.0282923, -0.0186008, 0.0188860
3: -0.0077672, -0.0022536, -0.0065978, -0.0024105, -0.0053567, 0.0043442
4: -0.0064042, 0.0051102, -0.0052683, 0.0049034, -0.0113075, 0.0103785
5: -0.0046007, 0.0086905, -0.0031402, 0.0079076, -0.0125083, 0.0118307
6: 0.9810924, 0.9925604, 0.9821798, 0.9925488, -0.0114564, 0.0103806
7: -0.0210170, -0.0044299, -0.0197363, -0.0045069, -0.0165101, 0.0153064
8: -0.0165995, 0.0036707, -0.0158729, 0.0018309, -0.0184303, 0.0195436
9: -0.0065318, 0.0068607, -0.0064836, 0.0051694, -0.0117011, 0.0133444

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0696394, upper bound: 0.0663203
time: 1.87 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0696394, upper bound: 0.0663201
time: 1.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0052499, 0.0087534, 0.0049210, 0.0094942, -0.0042443, 0.0038324
1: -0.0036094, 0.0010080, -0.0045618, 0.0068500, -0.0104594, 0.0055699
2: 0.0095092, 0.0280319, 0.0093803, 0.0289723, -0.0194631, 0.0186516
3: -0.0058035, -0.0025170, -0.0086720, -0.0021322, -0.0036713, 0.0061549
4: -0.0044967, 0.0050249, -0.0072830, 0.0058696, -0.0103663, 0.0123079
5: -0.0021481, 0.0073758, -0.0057307, 0.0092962, -0.0114443, 0.0131065
6: 0.9829183, 0.9925821, 0.9802511, 0.9925613, -0.0096430, 0.0123310
7: -0.0188663, -0.0042869, -0.0220079, -0.0044236, -0.0144427, 0.0177210
8: -0.0153793, 0.0005810, -0.0171617, 0.0050943, -0.0204736, 0.0177427
9: -0.0066212, 0.0040204, -0.0065357, 0.0081695, -0.0147906, 0.0105561

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656891, upper bound: 0.0694810
time: 1.54 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656887, upper bound: 0.0693755
time: 1.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0050248, 0.0092606, 0.0048550, 0.0096429, -0.0046181, 0.0044056
1: -0.0042614, 0.0050073, -0.0047530, 0.0080223, -0.0122837, 0.0097603
2: 0.0096915, 0.0286757, 0.0082823, 0.0291610, -0.0194694, 0.0203933
3: -0.0077672, -0.0022536, -0.0092475, -0.0020551, -0.0057121, 0.0069939
4: -0.0064042, 0.0051102, -0.0078421, 0.0063528, -0.0127569, 0.0129523
5: -0.0046007, 0.0086905, -0.0064496, 0.0096816, -0.0142822, 0.0151401
6: 0.9810924, 0.9925604, 0.9797159, 0.9925621, -0.0114697, 0.0128445
7: -0.0210170, -0.0044299, -0.0226383, -0.0044196, -0.0165974, 0.0182084
8: -0.0165995, 0.0036707, -0.0175194, 0.0060000, -0.0225994, 0.0211901
9: -0.0065318, 0.0068607, -0.0065382, 0.0090020, -0.0155337, 0.0133990

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0696210, upper bound: 0.0656987
time: 1.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0696210, upper bound: 0.0656991
time: 1.56 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.89 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661517, upper bound: 0.0697144
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661516, upper bound: 0.0693127
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0698102, upper bound: 0.0666474
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0698102, upper bound: 0.0699049
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661475, upper bound: 0.0693098
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661477, upper bound: 0.0691154
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0697507, upper bound: 0.0659002
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0697507, upper bound: 0.0695405
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656313, upper bound: 0.0696826
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656312, upper bound: 0.0692817
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0693945, upper bound: 0.0666427
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0693945, upper bound: 0.0699007
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656308, upper bound: 0.0693099
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656307, upper bound: 0.0691108
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0693788, upper bound: 0.0659000
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0693788, upper bound: 0.0695396
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661478, upper bound: 0.0697398
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661478, upper bound: 0.0695630
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0696451, upper bound: 0.0663261
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0696451, upper bound: 0.0699232
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661439, upper bound: 0.0695924
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661444, upper bound: 0.0695040
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0695671, upper bound: 0.0657002
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0695671, upper bound: 0.0657004
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656270, upper bound: 0.0697200
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656269, upper bound: 0.0695387
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0692422, upper bound: 0.0663209
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0692422, upper bound: 0.0699206
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656266, upper bound: 0.0695908
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656267, upper bound: 0.0694958
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0692385, upper bound: 0.0656993
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0692385, upper bound: 0.0656999
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661574, upper bound: 0.0695933
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661574, upper bound: 0.0692265
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0697953, upper bound: 0.0666423
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0697953, upper bound: 0.0699004
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661534, upper bound: 0.0692179
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661535, upper bound: 0.0690317
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0697451, upper bound: 0.0658952
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0697451, upper bound: 0.0695364
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656892, upper bound: 0.0695382
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656894, upper bound: 0.0691907
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0696393, upper bound: 0.0666381
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0696393, upper bound: 0.0666382
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656887, upper bound: 0.0692177
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656883, upper bound: 0.0690253
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0696209, upper bound: 0.0658951
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0696209, upper bound: 0.0695359
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661575, upper bound: 0.0696657
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661575, upper bound: 0.0694472
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0697950, upper bound: 0.0663248
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0697950, upper bound: 0.0699216
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661538, upper bound: 0.0694857
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0661539, upper bound: 0.0693811
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0697452, upper bound: 0.0656994
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0697452, upper bound: 0.0697884
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656894, upper bound: 0.0696397
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656893, upper bound: 0.0694220
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0696394, upper bound: 0.0663203
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0696394, upper bound: 0.0663201
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656891, upper bound: 0.0694810
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0656887, upper bound: 0.0693755
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0696210, upper bound: 0.0656987
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 6, lower bound: -0.0696210, upper bound: 0.0656991

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0055103, 0.0081807, 0.0052480, 0.0087578, -0.0032474, 0.0029327
1: -0.0028554, 0.0008601, -0.0036150, 0.0010426, -0.0038981, 0.0044752
2: 0.0099901, 0.0272875, 0.0104637, 0.0280375, -0.0180474, 0.0168238
3: -0.0047990, -0.0028216, -0.0058205, -0.0025148, -0.0022843, 0.0029989
4: -0.0032308, 0.0048165, -0.0045132, 0.0046113, -0.0078421, 0.0093297
5: -0.0012388, 0.0058556, -0.0021693, 0.0073872, -0.0086259, 0.0080249
6: 0.9845367, 0.9925250, 0.9829026, 0.9924688, -0.0079321, 0.0096225
7: -0.0174291, -0.0046641, -0.0188849, -0.0050356, -0.0123936, 0.0142208
8: -0.0139683, -0.0004729, -0.0153898, 0.0006077, -0.0145761, 0.0149169
9: -0.0063853, 0.0024155, -0.0061531, 0.0040449, -0.0104303, 0.0085685

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661517, upper bound: 0.0693127
time: 1.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661517, upper bound: 0.0693127
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0055173, 0.0081657, 0.0050628, 0.0091748, -0.0036575, 0.0031028
1: -0.0028353, 0.0008553, -0.0041512, 0.0043310, -0.0071663, 0.0050064
2: 0.0100293, 0.0272676, 0.0104624, 0.0285668, -0.0185375, 0.0168052
3: -0.0047963, -0.0028297, -0.0074351, -0.0022982, -0.0024981, 0.0046054
4: -0.0032149, 0.0047995, -0.0060816, 0.0048315, -0.0080463, 0.0108811
5: -0.0012362, 0.0058150, -0.0041859, 0.0084682, -0.0097044, 0.0100009
6: 0.9845705, 0.9925203, 0.9814012, 0.9924690, -0.0078984, 0.0111191
7: -0.0174107, -0.0046949, -0.0206533, -0.0050346, -0.0123761, 0.0159584
8: -0.0139307, -0.0004825, -0.0163931, 0.0031483, -0.0170789, 0.0159106
9: -0.0063661, 0.0024046, -0.0061536, 0.0063804, -0.0127465, 0.0085582

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0661077, upper bound: 0.0664959
time: 1.46 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0633873, upper bound: 0.0664850
time: 1.63 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0052724, 0.0087028, 0.0055103, 0.0081807, -0.0029083, 0.0031924
1: -0.0035443, 0.0008382, -0.0028554, 0.0008601, -0.0044044, 0.0036936
2: 0.0101671, 0.0279676, 0.0099901, 0.0272875, -0.0171204, 0.0179776
3: -0.0056075, -0.0025433, -0.0047990, -0.0028216, -0.0027859, 0.0022557
4: -0.0043063, 0.0047398, -0.0032308, 0.0048165, -0.0091229, 0.0079706
5: -0.0019033, 0.0072446, -0.0012388, 0.0058556, -0.0077589, 0.0084833
6: 0.9831008, 0.9925041, 0.9845367, 0.9925250, -0.0094243, 0.0079674
7: -0.0186516, -0.0048029, -0.0174291, -0.0046641, -0.0139875, 0.0126262
8: -0.0152575, 0.0002727, -0.0139683, -0.0004729, -0.0147846, 0.0142410
9: -0.0062985, 0.0037369, -0.0063853, 0.0024155, -0.0087140, 0.0101222

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697144, upper bound: 0.0661518
time: 1.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0693127, upper bound: 0.0661516
time: 1.63 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0052724, 0.0087028, 0.0052724, 0.0087028, -0.0034304, 0.0034304
1: -0.0035443, 0.0008382, -0.0035443, 0.0008382, -0.0043825, 0.0043825
2: 0.0101671, 0.0279676, 0.0101671, 0.0279676, -0.0178006, 0.0178006
3: -0.0056075, -0.0025433, -0.0056075, -0.0025433, -0.0030642, 0.0030642
4: -0.0043063, 0.0047398, -0.0043063, 0.0047398, -0.0090462, 0.0090462
5: -0.0019033, 0.0072446, -0.0019033, 0.0072446, -0.0091479, 0.0091479
6: 0.9831008, 0.9925041, 0.9831008, 0.9925041, -0.0094033, 0.0094033
7: -0.0186516, -0.0048029, -0.0186516, -0.0048029, -0.0138487, 0.0138487
8: -0.0152575, 0.0002727, -0.0152575, 0.0002727, -0.0155302, 0.0155302
9: -0.0062985, 0.0037369, -0.0062985, 0.0037369, -0.0100354, 0.0100354

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697144, upper bound: 0.0694357
time: 1.78 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0693127, upper bound: 0.0694354
time: 1.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0055103, 0.0081807, 0.0049400, 0.0094513, -0.0039410, 0.0032407
1: -0.0028554, 0.0008601, -0.0045067, 0.0065117, -0.0093672, 0.0053668
2: 0.0099901, 0.0272875, 0.0096972, 0.0289178, -0.0189278, 0.0175904
3: -0.0047990, -0.0028216, -0.0085059, -0.0021545, -0.0026445, 0.0056842
4: -0.0032308, 0.0048165, -0.0071217, 0.0057302, -0.0089610, 0.0119382
5: -0.0012388, 0.0058556, -0.0055233, 0.0091850, -0.0104238, 0.0113789
6: 0.9845367, 0.9925250, 0.9804056, 0.9924793, -0.0079426, 0.0121195
7: -0.0174291, -0.0046641, -0.0218260, -0.0049665, -0.0124627, 0.0171619
8: -0.0139683, -0.0004729, -0.0170585, 0.0048330, -0.0188013, 0.0165856
9: -0.0063853, 0.0024155, -0.0061962, 0.0079292, -0.0143145, 0.0086117

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661478, upper bound: 0.0691154
time: 2.15 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661478, upper bound: 0.0691154
time: 3.70 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0055173, 0.0081657, 0.0047398, 0.0099023, -0.0043850, 0.0034259
1: -0.0028353, 0.0008553, -0.0050865, 0.0100678, -0.0129031, 0.0059417
2: 0.0100293, 0.0272676, 0.0063664, 0.0294902, -0.0194609, 0.0209012
3: -0.0047963, -0.0028297, -0.0102519, -0.0019203, -0.0028760, 0.0074222
4: -0.0032149, 0.0047995, -0.0088178, 0.0071958, -0.0104107, 0.0136173
5: -0.0012362, 0.0058150, -0.0077040, 0.0103540, -0.0115902, 0.0135190
6: 0.9845705, 0.9925203, 0.9787821, 0.9924834, -0.0079129, 0.0137382
7: -0.0174107, -0.0046949, -0.0237383, -0.0049388, -0.0124719, 0.0190435
8: -0.0139307, -0.0004825, -0.0181435, 0.0075803, -0.0215109, 0.0176609
9: -0.0063661, 0.0024046, -0.0062136, 0.0104547, -0.0168208, 0.0086181

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0660799, upper bound: 0.0663461
time: 1.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0633832, upper bound: 0.0663357
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0052724, 0.0087028, 0.0052009, 0.0088638, -0.0035914, 0.0035019
1: -0.0035443, 0.0008382, -0.0037513, 0.0018787, -0.0054230, 0.0045895
2: 0.0101671, 0.0279676, 0.0098702, 0.0281720, -0.0180050, 0.0180975
3: -0.0056075, -0.0025433, -0.0062310, -0.0024597, -0.0031479, 0.0036877
4: -0.0043063, 0.0047398, -0.0049120, 0.0048685, -0.0091748, 0.0096518
5: -0.0019033, 0.0072446, -0.0026821, 0.0076620, -0.0095653, 0.0099266
6: 0.9831008, 0.9925041, 0.9825209, 0.9925392, -0.0094385, 0.0099832
7: -0.0186516, -0.0048029, -0.0193345, -0.0045700, -0.0140816, 0.0145316
8: -0.0152575, 0.0002727, -0.0156449, 0.0012537, -0.0165112, 0.0159176
9: -0.0062985, 0.0037369, -0.0064441, 0.0046388, -0.0109373, 0.0101810

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0696828, upper bound: 0.0656316
time: 1.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0692817, upper bound: 0.0656312
time: 1.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0052724, 0.0087028, 0.0049721, 0.0093792, -0.0041068, 0.0037307
1: -0.0035443, 0.0008382, -0.0044140, 0.0059430, -0.0094873, 0.0052522
2: 0.0101671, 0.0279676, 0.0100427, 0.0288263, -0.0186592, 0.0179249
3: -0.0056075, -0.0025433, -0.0082266, -0.0021920, -0.0034155, 0.0056833
4: -0.0043063, 0.0047398, -0.0068504, 0.0054958, -0.0098022, 0.0115903
5: -0.0019033, 0.0072446, -0.0051745, 0.0089981, -0.0109014, 0.0124191
6: 0.9831008, 0.9925041, 0.9806653, 0.9925188, -0.0094180, 0.0118387
7: -0.0186516, -0.0048029, -0.0215201, -0.0047054, -0.0139462, 0.0167172
8: -0.0152575, 0.0002727, -0.0168850, 0.0043936, -0.0196511, 0.0171576
9: -0.0062985, 0.0037369, -0.0063595, 0.0075252, -0.0138238, 0.0100964

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0696828, upper bound: 0.0656316
time: 1.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0692817, upper bound: 0.0692943
time: 2.47 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0052009, 0.0088638, 0.0052480, 0.0087578, -0.0035569, 0.0036158
1: -0.0037513, 0.0018787, -0.0036150, 0.0010426, -0.0047940, 0.0054937
2: 0.0098702, 0.0281720, 0.0104637, 0.0280375, -0.0181673, 0.0177084
3: -0.0062310, -0.0024597, -0.0058205, -0.0025148, -0.0037163, 0.0033608
4: -0.0049120, 0.0048685, -0.0045132, 0.0046113, -0.0095233, 0.0093817
5: -0.0026821, 0.0076620, -0.0021693, 0.0073872, -0.0100692, 0.0098313
6: 0.9825209, 0.9925392, 0.9829026, 0.9924688, -0.0099480, 0.0096366
7: -0.0193345, -0.0045700, -0.0188849, -0.0050356, -0.0142989, 0.0143148
8: -0.0156449, 0.0012537, -0.0153898, 0.0006077, -0.0162527, 0.0166435
9: -0.0064441, 0.0046388, -0.0061531, 0.0040449, -0.0104891, 0.0107918

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656312, upper bound: 0.0692817
time: 1.87 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656312, upper bound: 0.0692817
time: 4.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0052082, 0.0088474, 0.0050628, 0.0091748, -0.0039666, 0.0037846
1: -0.0037303, 0.0017496, -0.0041512, 0.0043310, -0.0080613, 0.0059008
2: 0.0099071, 0.0281513, 0.0104624, 0.0285668, -0.0186597, 0.0176888
3: -0.0061676, -0.0024682, -0.0074351, -0.0022982, -0.0038695, 0.0049669
4: -0.0048504, 0.0048525, -0.0060816, 0.0048315, -0.0096818, 0.0109341
5: -0.0026028, 0.0076196, -0.0041859, 0.0084682, -0.0110710, 0.0118055
6: 0.9825799, 0.9925349, 0.9814012, 0.9924690, -0.0098891, 0.0111337
7: -0.0192651, -0.0045990, -0.0206533, -0.0050346, -0.0142304, 0.0160543
8: -0.0156055, 0.0011539, -0.0163931, 0.0031483, -0.0187538, 0.0175470
9: -0.0064260, 0.0045470, -0.0061536, 0.0063804, -0.0128064, 0.0107007

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0655975, upper bound: 0.0664719
time: 1.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0628726, upper bound: 0.0664579
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0049721, 0.0093792, 0.0055103, 0.0081807, -0.0032086, 0.0038689
1: -0.0044140, 0.0059430, -0.0028554, 0.0008601, -0.0052741, 0.0087984
2: 0.0100427, 0.0288263, 0.0099901, 0.0272875, -0.0172448, 0.0188362
3: -0.0082266, -0.0021920, -0.0047990, -0.0028216, -0.0054050, 0.0026070
4: -0.0068504, 0.0054958, -0.0032308, 0.0048165, -0.0116670, 0.0087266
5: -0.0051745, 0.0089981, -0.0012388, 0.0058556, -0.0110301, 0.0102368
6: 0.9806653, 0.9925188, 0.9845367, 0.9925250, -0.0118597, 0.0079821
7: -0.0215201, -0.0047054, -0.0174291, -0.0046641, -0.0168561, 0.0127238
8: -0.0168850, 0.0043936, -0.0139683, -0.0004729, -0.0164121, 0.0183619
9: -0.0063595, 0.0075252, -0.0063853, 0.0024155, -0.0087750, 0.0139106

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0693098, upper bound: 0.0661474
time: 1.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0691155, upper bound: 0.0661478
time: 4.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0049721, 0.0093792, 0.0052724, 0.0087028, -0.0037307, 0.0041068
1: -0.0044140, 0.0059430, -0.0035443, 0.0008382, -0.0052522, 0.0094873
2: 0.0100427, 0.0288263, 0.0101671, 0.0279676, -0.0179249, 0.0186592
3: -0.0082266, -0.0021920, -0.0056075, -0.0025433, -0.0056833, 0.0034155
4: -0.0068504, 0.0054958, -0.0043063, 0.0047398, -0.0115903, 0.0098022
5: -0.0051745, 0.0089981, -0.0019033, 0.0072446, -0.0124191, 0.0109014
6: 0.9806653, 0.9925188, 0.9831008, 0.9925041, -0.0118387, 0.0094180
7: -0.0215201, -0.0047054, -0.0186516, -0.0048029, -0.0167172, 0.0139462
8: -0.0168850, 0.0043936, -0.0152575, 0.0002727, -0.0171576, 0.0196511
9: -0.0063595, 0.0075252, -0.0062985, 0.0037369, -0.0100964, 0.0138238

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0693099, upper bound: 0.0661478
time: 1.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0691155, upper bound: 0.0661480
time: 1.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0052009, 0.0088638, 0.0049400, 0.0094513, -0.0042504, 0.0039238
1: -0.0037513, 0.0018787, -0.0045067, 0.0065117, -0.0102631, 0.0063854
2: 0.0098702, 0.0281720, 0.0096972, 0.0289178, -0.0190477, 0.0184749
3: -0.0062310, -0.0024597, -0.0085059, -0.0021545, -0.0040765, 0.0060462
4: -0.0049120, 0.0048685, -0.0071217, 0.0057302, -0.0106422, 0.0119902
5: -0.0026821, 0.0076620, -0.0055233, 0.0091850, -0.0118671, 0.0131853
6: 0.9825209, 0.9925392, 0.9804056, 0.9924793, -0.0099584, 0.0121337
7: -0.0193345, -0.0045700, -0.0218260, -0.0049665, -0.0143680, 0.0172560
8: -0.0156449, 0.0012537, -0.0170585, 0.0048330, -0.0204779, 0.0183122
9: -0.0064441, 0.0046388, -0.0061962, 0.0079292, -0.0143733, 0.0108350

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656303, upper bound: 0.0691108
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656303, upper bound: 0.0691108
time: 1.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0052082, 0.0088474, 0.0047398, 0.0099023, -0.0046941, 0.0041076
1: -0.0037303, 0.0017496, -0.0050865, 0.0100678, -0.0137981, 0.0068361
2: 0.0099071, 0.0281513, 0.0063664, 0.0294902, -0.0195831, 0.0217849
3: -0.0061676, -0.0024682, -0.0102519, -0.0019203, -0.0042473, 0.0077837
4: -0.0048504, 0.0048525, -0.0088178, 0.0071958, -0.0120462, 0.0136702
5: -0.0026028, 0.0076196, -0.0077040, 0.0103540, -0.0129568, 0.0153236
6: 0.9825799, 0.9925349, 0.9787821, 0.9924834, -0.0099036, 0.0137529
7: -0.0192651, -0.0045990, -0.0237383, -0.0049388, -0.0143262, 0.0191393
8: -0.0156055, 0.0011539, -0.0181435, 0.0075803, -0.0231858, 0.0192974
9: -0.0064260, 0.0045470, -0.0062136, 0.0104547, -0.0168808, 0.0107606

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0655969, upper bound: 0.0663383
time: 1.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0628720, upper bound: 0.0663268
time: 1.66 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0049721, 0.0093792, 0.0052009, 0.0088638, -0.0038917, 0.0041783
1: -0.0044140, 0.0059430, -0.0037513, 0.0018787, -0.0062927, 0.0096943
2: 0.0100427, 0.0288263, 0.0098702, 0.0281720, -0.0181293, 0.0189561
3: -0.0082266, -0.0021920, -0.0062310, -0.0024597, -0.0057669, 0.0040390
4: -0.0068504, 0.0054958, -0.0049120, 0.0048685, -0.0117189, 0.0104078
5: -0.0051745, 0.0089981, -0.0026821, 0.0076620, -0.0128365, 0.0116801
6: 0.9806653, 0.9925188, 0.9825209, 0.9925392, -0.0118739, 0.0099979
7: -0.0215201, -0.0047054, -0.0193345, -0.0045700, -0.0169501, 0.0146291
8: -0.0168850, 0.0043936, -0.0156449, 0.0012537, -0.0181386, 0.0200385
9: -0.0063595, 0.0075252, -0.0064441, 0.0046388, -0.0109983, 0.0139694

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0693098, upper bound: 0.0656308
time: 1.75 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0691108, upper bound: 0.0656307
time: 1.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0049721, 0.0093792, 0.0049721, 0.0093792, -0.0044071, 0.0044071
1: -0.0044140, 0.0059430, -0.0044140, 0.0059430, -0.0103569, 0.0103569
2: 0.0100427, 0.0288263, 0.0100427, 0.0288263, -0.0187835, 0.0187835
3: -0.0082266, -0.0021920, -0.0082266, -0.0021920, -0.0060346, 0.0060346
4: -0.0068504, 0.0054958, -0.0068504, 0.0054958, -0.0123463, 0.0123463
5: -0.0051745, 0.0089981, -0.0051745, 0.0089981, -0.0141725, 0.0141725
6: 0.9806653, 0.9925188, 0.9806653, 0.9925188, -0.0118535, 0.0118535
7: -0.0215201, -0.0047054, -0.0215201, -0.0047054, -0.0168147, 0.0168147
8: -0.0168850, 0.0043936, -0.0168850, 0.0043936, -0.0212786, 0.0212786
9: -0.0063595, 0.0075252, -0.0063595, 0.0075252, -0.0138848, 0.0138848

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0693098, upper bound: 0.0692943
time: 1.63 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0691108, upper bound: 0.0692939
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0055103, 0.0081807, 0.0052962, 0.0086492, -0.0031389, 0.0028845
1: -0.0028554, 0.0008601, -0.0034754, 0.0008471, -0.0037025, 0.0043356
2: 0.0099901, 0.0272875, 0.0100954, 0.0278996, -0.0179096, 0.0171921
3: -0.0047990, -0.0028216, -0.0054001, -0.0025712, -0.0022279, 0.0025785
4: -0.0032308, 0.0048165, -0.0041048, 0.0047709, -0.0080017, 0.0089213
5: -0.0012388, 0.0058556, -0.0016442, 0.0071057, -0.0083444, 0.0074998
6: 0.9845367, 0.9925250, 0.9832935, 0.9925126, -0.0079759, 0.0092315
7: -0.0174291, -0.0046641, -0.0184244, -0.0047467, -0.0126825, 0.0137603
8: -0.0139683, -0.0004729, -0.0151286, -0.0000538, -0.0139146, 0.0146557
9: -0.0063853, 0.0024155, -0.0063337, 0.0034368, -0.0098222, 0.0087492

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661475, upper bound: 0.0695630
time: 1.58 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661475, upper bound: 0.0695630
time: 1.63 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0055173, 0.0081657, 0.0051168, 0.0090533, -0.0035360, 0.0030489
1: -0.0028353, 0.0008553, -0.0039949, 0.0033729, -0.0062082, 0.0048502
2: 0.0100293, 0.0272676, 0.0101212, 0.0284126, -0.0183832, 0.0171465
3: -0.0047963, -0.0028297, -0.0069647, -0.0023613, -0.0024350, 0.0041349
4: -0.0032149, 0.0047995, -0.0056246, 0.0047597, -0.0079746, 0.0104241
5: -0.0012362, 0.0058150, -0.0035983, 0.0081532, -0.0093894, 0.0094133
6: 0.9845705, 0.9925203, 0.9818386, 0.9925094, -0.0079389, 0.0106817
7: -0.0174107, -0.0046949, -0.0201380, -0.0047669, -0.0126438, 0.0154431
8: -0.0139307, -0.0004825, -0.0161008, 0.0024080, -0.0163387, 0.0156183
9: -0.0063661, 0.0024046, -0.0063210, 0.0056999, -0.0120660, 0.0087256

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0660395, upper bound: 0.0665505
time: 1.52 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0633828, upper bound: 0.0665395
time: 1.40 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0052724, 0.0087028, 0.0055544, 0.0080854, -0.0028130, 0.0031484
1: -0.0035443, 0.0008382, -0.0027278, 0.0009054, -0.0044497, 0.0035660
2: 0.0101671, 0.0279676, 0.0096249, 0.0271615, -0.0169944, 0.0183428
3: -0.0056075, -0.0025433, -0.0047817, -0.0028732, -0.0027344, 0.0022384
4: -0.0043063, 0.0047398, -0.0031298, 0.0049748, -0.0092811, 0.0078697
5: -0.0019033, 0.0072446, -0.0012624, 0.0055982, -0.0075015, 0.0085070
6: 0.9831008, 0.9925041, 0.9847512, 0.9925683, -0.0094675, 0.0077529
7: -0.0186516, -0.0048029, -0.0173125, -0.0043776, -0.0142740, 0.0125096
8: -0.0152575, 0.0002727, -0.0137295, -0.0003831, -0.0148744, 0.0140021
9: -0.0062985, 0.0037369, -0.0065645, 0.0023464, -0.0086449, 0.0103014

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695934, upper bound: 0.0661571
time: 1.60 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0692265, upper bound: 0.0661573
time: 1.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0052724, 0.0087028, 0.0053224, 0.0085902, -0.0033178, 0.0033804
1: -0.0035443, 0.0008382, -0.0033996, 0.0008834, -0.0044277, 0.0042378
2: 0.0101671, 0.0279676, 0.0098025, 0.0278248, -0.0176577, 0.0181651
3: -0.0056075, -0.0025433, -0.0051717, -0.0026018, -0.0030058, 0.0026283
4: -0.0043063, 0.0047398, -0.0038829, 0.0048978, -0.0092041, 0.0086228
5: -0.0019033, 0.0072446, -0.0013589, 0.0069528, -0.0088561, 0.0086035
6: 0.9831008, 0.9925041, 0.9835059, 0.9925473, -0.0094465, 0.0089982
7: -0.0186516, -0.0048029, -0.0181742, -0.0045170, -0.0141346, 0.0133713
8: -0.0152575, 0.0002727, -0.0149867, -0.0004131, -0.0148444, 0.0152593
9: -0.0062985, 0.0037369, -0.0064773, 0.0031064, -0.0094050, 0.0102142

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695934, upper bound: 0.0696741
time: 1.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0692265, upper bound: 0.0696736
time: 2.29 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0055103, 0.0081807, 0.0049901, 0.0093386, -0.0038283, 0.0031906
1: -0.0028554, 0.0008601, -0.0043618, 0.0056230, -0.0084784, 0.0052219
2: 0.0099901, 0.0272875, 0.0100201, 0.0287748, -0.0187847, 0.0172674
3: -0.0047990, -0.0028216, -0.0080695, -0.0022131, -0.0025859, 0.0052478
4: -0.0032308, 0.0048165, -0.0066978, 0.0053639, -0.0085947, 0.0115143
5: -0.0012388, 0.0058556, -0.0049782, 0.0088929, -0.0101316, 0.0108338
6: 0.9845367, 0.9925250, 0.9808112, 0.9925215, -0.0079848, 0.0117138
7: -0.0174291, -0.0046641, -0.0213480, -0.0046877, -0.0127415, 0.0166840
8: -0.0139683, -0.0004729, -0.0167873, 0.0041464, -0.0181147, 0.0163144
9: -0.0063853, 0.0024155, -0.0063706, 0.0072980, -0.0136833, 0.0087860

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661440, upper bound: 0.0695039
time: 1.89 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661440, upper bound: 0.0695040
time: 1.66 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0055173, 0.0081657, 0.0047993, 0.0097683, -0.0042510, 0.0033664
1: -0.0028353, 0.0008553, -0.0049142, 0.0090114, -0.0118467, 0.0057695
2: 0.0100293, 0.0272676, 0.0073558, 0.0293202, -0.0192909, 0.0199118
3: -0.0047963, -0.0028297, -0.0097332, -0.0019899, -0.0028064, 0.0069035
4: -0.0032149, 0.0047995, -0.0083139, 0.0067605, -0.0099753, 0.0131135
5: -0.0012362, 0.0058150, -0.0070562, 0.0100067, -0.0112430, 0.0128712
6: 0.9845705, 0.9925203, 0.9792643, 0.9925236, -0.0079531, 0.0132560
7: -0.0174107, -0.0046949, -0.0231703, -0.0046732, -0.0127375, 0.0184754
8: -0.0139307, -0.0004825, -0.0178212, 0.0067642, -0.0206949, 0.0173386
9: -0.0063661, 0.0024046, -0.0063796, 0.0097045, -0.0160706, 0.0087842

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0660202, upper bound: 0.0664802
time: 2.16 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0633795, upper bound: 0.0664695
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0052724, 0.0087028, 0.0052499, 0.0087534, -0.0034810, 0.0034528
1: -0.0035443, 0.0008382, -0.0036094, 0.0010080, -0.0045523, 0.0044476
2: 0.0101671, 0.0279676, 0.0095092, 0.0280319, -0.0178648, 0.0184584
3: -0.0056075, -0.0025433, -0.0058035, -0.0025170, -0.0030905, 0.0032602
4: -0.0043063, 0.0047398, -0.0044967, 0.0050249, -0.0093312, 0.0092365
5: -0.0019033, 0.0072446, -0.0021481, 0.0073758, -0.0092791, 0.0093927
6: 0.9831008, 0.9925041, 0.9829183, 0.9925821, -0.0094814, 0.0095857
7: -0.0186516, -0.0048029, -0.0188663, -0.0042869, -0.0143647, 0.0140633
8: -0.0152575, 0.0002727, -0.0153793, 0.0005810, -0.0158385, 0.0156520
9: -0.0062985, 0.0037369, -0.0066212, 0.0040204, -0.0103189, 0.0103581

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695382, upper bound: 0.0656893
time: 1.91 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0691909, upper bound: 0.0656894
time: 1.76 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0052724, 0.0087028, 0.0050248, 0.0092606, -0.0039881, 0.0036780
1: -0.0035443, 0.0008382, -0.0042614, 0.0050073, -0.0085516, 0.0050996
2: 0.0101671, 0.0279676, 0.0096915, 0.0286757, -0.0185086, 0.0182761
3: -0.0056075, -0.0025433, -0.0077672, -0.0022536, -0.0033539, 0.0052239
4: -0.0043063, 0.0047398, -0.0064042, 0.0051102, -0.0094165, 0.0111440
5: -0.0019033, 0.0072446, -0.0046007, 0.0086905, -0.0105938, 0.0118453
6: 0.9831008, 0.9925041, 0.9810924, 0.9925604, -0.0094596, 0.0114117
7: -0.0186516, -0.0048029, -0.0210170, -0.0044299, -0.0142217, 0.0162141
8: -0.0152575, 0.0002727, -0.0165995, 0.0036707, -0.0189282, 0.0168722
9: -0.0062985, 0.0037369, -0.0065318, 0.0068607, -0.0131593, 0.0102687

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695382, upper bound: 0.0696423
time: 1.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0691909, upper bound: 0.0656894
time: 1.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0052009, 0.0088638, 0.0052962, 0.0086492, -0.0034483, 0.0035676
1: -0.0037513, 0.0018787, -0.0034754, 0.0008471, -0.0045984, 0.0053542
2: 0.0098702, 0.0281720, 0.0100954, 0.0278996, -0.0180295, 0.0180767
3: -0.0062310, -0.0024597, -0.0054001, -0.0025712, -0.0036599, 0.0029404
4: -0.0049120, 0.0048685, -0.0041048, 0.0047709, -0.0096829, 0.0089733
5: -0.0026821, 0.0076620, -0.0016442, 0.0071057, -0.0097877, 0.0093062
6: 0.9825209, 0.9925392, 0.9832935, 0.9925126, -0.0099917, 0.0092457
7: -0.0193345, -0.0045700, -0.0184244, -0.0047467, -0.0145878, 0.0138544
8: -0.0156449, 0.0012537, -0.0151286, -0.0000538, -0.0155912, 0.0163823
9: -0.0064441, 0.0046388, -0.0063337, 0.0034368, -0.0098810, 0.0109724

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656270, upper bound: 0.0695388
time: 1.50 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656270, upper bound: 0.0695388
time: 1.53 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0052082, 0.0088474, 0.0051168, 0.0090533, -0.0038451, 0.0037306
1: -0.0037303, 0.0017496, -0.0039949, 0.0033729, -0.0071032, 0.0057445
2: 0.0099071, 0.0281513, 0.0101212, 0.0284126, -0.0185054, 0.0180301
3: -0.0061676, -0.0024682, -0.0069647, -0.0023613, -0.0038064, 0.0044965
4: -0.0048504, 0.0048525, -0.0056246, 0.0047597, -0.0096101, 0.0104771
5: -0.0026028, 0.0076196, -0.0035983, 0.0081532, -0.0107560, 0.0112179
6: 0.9825799, 0.9925349, 0.9818386, 0.9925094, -0.0099295, 0.0106964
7: -0.0192651, -0.0045990, -0.0201380, -0.0047669, -0.0144981, 0.0155390
8: -0.0156055, 0.0011539, -0.0161008, 0.0024080, -0.0180135, 0.0172547
9: -0.0064260, 0.0045470, -0.0063210, 0.0056999, -0.0121259, 0.0108681

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0655400, upper bound: 0.0665282
time: 1.49 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0628670, upper bound: 0.0665120
time: 1.56 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0049721, 0.0093792, 0.0055544, 0.0080854, -0.0031133, 0.0038248
1: -0.0044140, 0.0059430, -0.0027278, 0.0009054, -0.0053194, 0.0086707
2: 0.0100427, 0.0288263, 0.0096249, 0.0271615, -0.0171187, 0.0192014
3: -0.0082266, -0.0021920, -0.0047817, -0.0028732, -0.0053534, 0.0025897
4: -0.0068504, 0.0054958, -0.0031298, 0.0049748, -0.0118252, 0.0086257
5: -0.0051745, 0.0089981, -0.0012624, 0.0055982, -0.0107727, 0.0102604
6: 0.9806653, 0.9925188, 0.9847512, 0.9925683, -0.0119029, 0.0077676
7: -0.0215201, -0.0047054, -0.0173125, -0.0043776, -0.0171425, 0.0126071
8: -0.0168850, 0.0043936, -0.0137295, -0.0003831, -0.0165018, 0.0181231
9: -0.0063595, 0.0075252, -0.0065645, 0.0023464, -0.0087059, 0.0140897

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0692179, upper bound: 0.0661534
time: 1.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0690317, upper bound: 0.0661536
time: 5.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0049721, 0.0093792, 0.0053224, 0.0085902, -0.0036181, 0.0040568
1: -0.0044140, 0.0059430, -0.0033996, 0.0008834, -0.0052973, 0.0093426
2: 0.0100427, 0.0288263, 0.0098025, 0.0278248, -0.0177820, 0.0190237
3: -0.0082266, -0.0021920, -0.0051717, -0.0026018, -0.0056248, 0.0029797
4: -0.0068504, 0.0054958, -0.0038829, 0.0048978, -0.0117482, 0.0093788
5: -0.0051745, 0.0089981, -0.0013589, 0.0069528, -0.0121272, 0.0103570
6: 0.9806653, 0.9925188, 0.9835059, 0.9925473, -0.0118819, 0.0090129
7: -0.0215201, -0.0047054, -0.0181742, -0.0045170, -0.0170031, 0.0134689
8: -0.0168850, 0.0043936, -0.0149867, -0.0004131, -0.0164718, 0.0193803
9: -0.0063595, 0.0075252, -0.0064773, 0.0031064, -0.0094659, 0.0140026

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0692179, upper bound: 0.0696728
time: 1.81 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0690317, upper bound: 0.0696726
time: 1.69 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0052009, 0.0088638, 0.0049901, 0.0093386, -0.0041377, 0.0038737
1: -0.0037513, 0.0018787, -0.0043618, 0.0056230, -0.0093743, 0.0062405
2: 0.0098702, 0.0281720, 0.0100201, 0.0287748, -0.0189046, 0.0181519
3: -0.0062310, -0.0024597, -0.0080695, -0.0022131, -0.0040179, 0.0056098
4: -0.0049120, 0.0048685, -0.0066978, 0.0053639, -0.0102759, 0.0115663
5: -0.0026821, 0.0076620, -0.0049782, 0.0088929, -0.0115749, 0.0126402
6: 0.9825209, 0.9925392, 0.9808112, 0.9925215, -0.0100006, 0.0117280
7: -0.0193345, -0.0045700, -0.0213480, -0.0046877, -0.0146468, 0.0167780
8: -0.0156449, 0.0012537, -0.0167873, 0.0041464, -0.0197913, 0.0180410
9: -0.0064441, 0.0046388, -0.0063706, 0.0072980, -0.0137421, 0.0110093

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656269, upper bound: 0.0694958
time: 2.12 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656269, upper bound: 0.0694958
time: 1.75 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0052082, 0.0088474, 0.0047993, 0.0097683, -0.0045601, 0.0040481
1: -0.0037303, 0.0017496, -0.0049142, 0.0090114, -0.0127417, 0.0066638
2: 0.0099071, 0.0281513, 0.0073558, 0.0293202, -0.0194131, 0.0207955
3: -0.0061676, -0.0024682, -0.0097332, -0.0019899, -0.0041777, 0.0072650
4: -0.0048504, 0.0048525, -0.0083139, 0.0067605, -0.0116108, 0.0131664
5: -0.0026028, 0.0076196, -0.0070562, 0.0100067, -0.0126096, 0.0146758
6: 0.9825799, 0.9925349, 0.9792643, 0.9925236, -0.0099437, 0.0132707
7: -0.0192651, -0.0045990, -0.0231703, -0.0046732, -0.0145918, 0.0185713
8: -0.0156055, 0.0011539, -0.0178212, 0.0067642, -0.0223697, 0.0189751
9: -0.0064260, 0.0045470, -0.0063796, 0.0097045, -0.0161305, 0.0109267

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0655400, upper bound: 0.0664718
time: 1.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0628668, upper bound: 0.0664591
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0049721, 0.0093792, 0.0052499, 0.0087534, -0.0037813, 0.0041293
1: -0.0044140, 0.0059430, -0.0036094, 0.0010080, -0.0054220, 0.0095523
2: 0.0100427, 0.0288263, 0.0095092, 0.0280319, -0.0179892, 0.0193170
3: -0.0082266, -0.0021920, -0.0058035, -0.0025170, -0.0057096, 0.0036115
4: -0.0068504, 0.0054958, -0.0044967, 0.0050249, -0.0118753, 0.0099925
5: -0.0051745, 0.0089981, -0.0021481, 0.0073758, -0.0125503, 0.0111461
6: 0.9806653, 0.9925188, 0.9829183, 0.9925821, -0.0119168, 0.0096005
7: -0.0215201, -0.0047054, -0.0188663, -0.0042869, -0.0172332, 0.0141609
8: -0.0168850, 0.0043936, -0.0153793, 0.0005810, -0.0174660, 0.0197729
9: -0.0063595, 0.0075252, -0.0066212, 0.0040204, -0.0103799, 0.0141464

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0692178, upper bound: 0.0656887
time: 1.63 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0690255, upper bound: 0.0656883
time: 1.78 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0049721, 0.0093792, 0.0050248, 0.0092606, -0.0042885, 0.0043545
1: -0.0044140, 0.0059430, -0.0042614, 0.0050073, -0.0094213, 0.0102044
2: 0.0100427, 0.0288263, 0.0096915, 0.0286757, -0.0186329, 0.0191347
3: -0.0082266, -0.0021920, -0.0077672, -0.0022536, -0.0059730, 0.0055752
4: -0.0068504, 0.0054958, -0.0064042, 0.0051102, -0.0119606, 0.0119000
5: -0.0051745, 0.0089981, -0.0046007, 0.0086905, -0.0138649, 0.0135987
6: 0.9806653, 0.9925188, 0.9810924, 0.9925604, -0.0118951, 0.0114264
7: -0.0215201, -0.0047054, -0.0210170, -0.0044299, -0.0170902, 0.0163116
8: -0.0168850, 0.0043936, -0.0165995, 0.0036707, -0.0205557, 0.0209931
9: -0.0063595, 0.0075252, -0.0065318, 0.0068607, -0.0132202, 0.0140570

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0692178, upper bound: 0.0696418
time: 1.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0690255, upper bound: 0.0696413
time: 1.98 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0055544, 0.0080854, 0.0052480, 0.0087578, -0.0032034, 0.0028374
1: -0.0027278, 0.0009054, -0.0036150, 0.0010426, -0.0037704, 0.0045204
2: 0.0096249, 0.0271615, 0.0104637, 0.0280375, -0.0184126, 0.0166978
3: -0.0047817, -0.0028732, -0.0058205, -0.0025148, -0.0022670, 0.0029473
4: -0.0031298, 0.0049748, -0.0045132, 0.0046113, -0.0077411, 0.0094880
5: -0.0012624, 0.0055982, -0.0021693, 0.0073872, -0.0086495, 0.0077675
6: 0.9847512, 0.9925683, 0.9829026, 0.9924688, -0.0077177, 0.0096657
7: -0.0173125, -0.0043776, -0.0188849, -0.0050356, -0.0122769, 0.0145073
8: -0.0137295, -0.0003831, -0.0153898, 0.0006077, -0.0143372, 0.0150067
9: -0.0065645, 0.0023464, -0.0061531, 0.0040449, -0.0106094, 0.0084994

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661570, upper bound: 0.0692266
time: 2.01 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661570, upper bound: 0.0692265
time: 1.80 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0055614, 0.0080702, 0.0050628, 0.0091748, -0.0036133, 0.0030074
1: -0.0027074, 0.0009004, -0.0041512, 0.0043310, -0.0070384, 0.0050516
2: 0.0096650, 0.0271414, 0.0104624, 0.0285668, -0.0189018, 0.0166790
3: -0.0047790, -0.0028814, -0.0074351, -0.0022982, -0.0024808, 0.0045537
4: -0.0031137, 0.0049574, -0.0060816, 0.0048315, -0.0079452, 0.0110390
5: -0.0012598, 0.0055572, -0.0041859, 0.0084682, -0.0097279, 0.0097431
6: 0.9847854, 0.9925635, 0.9814012, 0.9924690, -0.0076835, 0.0111623
7: -0.0172939, -0.0044091, -0.0206533, -0.0050346, -0.0122593, 0.0162442
8: -0.0136914, -0.0003930, -0.0163931, 0.0031483, -0.0168396, 0.0160001
9: -0.0065448, 0.0023354, -0.0061536, 0.0063804, -0.0129252, 0.0084890

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0661136, upper bound: 0.0663950
time: 1.78 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0631988, upper bound: 0.0663587
time: 1.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0053224, 0.0085902, 0.0055103, 0.0081807, -0.0028583, 0.0030799
1: -0.0033996, 0.0008834, -0.0028554, 0.0008601, -0.0042597, 0.0037388
2: 0.0098025, 0.0278248, 0.0099901, 0.0272875, -0.0174850, 0.0178347
3: -0.0051717, -0.0026018, -0.0047990, -0.0028216, -0.0023500, 0.0021972
4: -0.0038829, 0.0048978, -0.0032308, 0.0048165, -0.0086995, 0.0081286
5: -0.0013589, 0.0069528, -0.0012388, 0.0058556, -0.0072145, 0.0081915
6: 0.9835059, 0.9925473, 0.9845367, 0.9925250, -0.0090191, 0.0080106
7: -0.0181742, -0.0045170, -0.0174291, -0.0046641, -0.0135102, 0.0129122
8: -0.0149867, -0.0004131, -0.0139683, -0.0004729, -0.0145138, 0.0135552
9: -0.0064773, 0.0031064, -0.0063853, 0.0024155, -0.0088928, 0.0094918

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697398, upper bound: 0.0661477
time: 1.78 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695630, upper bound: 0.0661478
time: 1.84 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0053224, 0.0085902, 0.0052724, 0.0087028, -0.0033804, 0.0033178
1: -0.0033996, 0.0008834, -0.0035443, 0.0008382, -0.0042378, 0.0044277
2: 0.0098025, 0.0278248, 0.0101671, 0.0279676, -0.0181651, 0.0176577
3: -0.0051717, -0.0026018, -0.0056075, -0.0025433, -0.0026283, 0.0030058
4: -0.0038829, 0.0048978, -0.0043063, 0.0047398, -0.0086228, 0.0092041
5: -0.0013589, 0.0069528, -0.0019033, 0.0072446, -0.0086035, 0.0088561
6: 0.9835059, 0.9925473, 0.9831008, 0.9925041, -0.0089982, 0.0094465
7: -0.0181742, -0.0045170, -0.0186516, -0.0048029, -0.0133713, 0.0141346
8: -0.0149867, -0.0004131, -0.0152575, 0.0002727, -0.0152593, 0.0148444
9: -0.0064773, 0.0031064, -0.0062985, 0.0037369, -0.0102142, 0.0094050

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697398, upper bound: 0.0694330
time: 1.79 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695630, upper bound: 0.0694330
time: 1.86 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0055544, 0.0080854, 0.0049400, 0.0094513, -0.0038969, 0.0031454
1: -0.0027278, 0.0009054, -0.0045067, 0.0065117, -0.0092395, 0.0054121
2: 0.0096249, 0.0271615, 0.0096972, 0.0289178, -0.0192929, 0.0174643
3: -0.0047817, -0.0028732, -0.0085059, -0.0021545, -0.0026272, 0.0056327
4: -0.0031298, 0.0049748, -0.0071217, 0.0057302, -0.0088600, 0.0120965
5: -0.0012624, 0.0055982, -0.0055233, 0.0091850, -0.0104474, 0.0111215
6: 0.9847512, 0.9925683, 0.9804056, 0.9924793, -0.0077282, 0.0121627
7: -0.0173125, -0.0043776, -0.0218260, -0.0049665, -0.0123460, 0.0174484
8: -0.0137295, -0.0003831, -0.0170585, 0.0048330, -0.0185625, 0.0166753
9: -0.0065645, 0.0023464, -0.0061962, 0.0079292, -0.0144937, 0.0085426

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661535, upper bound: 0.0690316
time: 1.85 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0661535, upper bound: 0.0690317
time: 1.65 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0055614, 0.0080702, 0.0047398, 0.0099023, -0.0043408, 0.0033304
1: -0.0027074, 0.0009004, -0.0050865, 0.0100678, -0.0127752, 0.0059869
2: 0.0096650, 0.0271414, 0.0063664, 0.0294902, -0.0198253, 0.0207750
3: -0.0047790, -0.0028814, -0.0102519, -0.0019203, -0.0028586, 0.0073705
4: -0.0031137, 0.0049574, -0.0088178, 0.0071958, -0.0103096, 0.0137752
5: -0.0012598, 0.0055572, -0.0077040, 0.0103540, -0.0116137, 0.0132612
6: 0.9847854, 0.9925635, 0.9787821, 0.9924834, -0.0076980, 0.0137815
7: -0.0172939, -0.0044091, -0.0237383, -0.0049388, -0.0123551, 0.0193293
8: -0.0136914, -0.0003930, -0.0181435, 0.0075803, -0.0212717, 0.0177505
9: -0.0065448, 0.0023354, -0.0062136, 0.0104547, -0.0169996, 0.0085489

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0660962, upper bound: 0.0662374
time: 18.86 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0631958, upper bound: 0.0661981
time: 1.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0053224, 0.0085902, 0.0052009, 0.0088638, -0.0035414, 0.0033893
1: -0.0033996, 0.0008834, -0.0037513, 0.0018787, -0.0052783, 0.0046347
2: 0.0098025, 0.0278248, 0.0098702, 0.0281720, -0.0183695, 0.0179546
3: -0.0051717, -0.0026018, -0.0062310, -0.0024597, -0.0027120, 0.0036292
4: -0.0038829, 0.0048978, -0.0049120, 0.0048685, -0.0087514, 0.0098098
5: -0.0013589, 0.0069528, -0.0026821, 0.0076620, -0.0090209, 0.0096348
6: 0.9835059, 0.9925473, 0.9825209, 0.9925392, -0.0090333, 0.0100264
7: -0.0181742, -0.0045170, -0.0193345, -0.0045700, -0.0136042, 0.0148175
8: -0.0149867, -0.0004131, -0.0156449, 0.0012537, -0.0162403, 0.0152318
9: -0.0064773, 0.0031064, -0.0064441, 0.0046388, -0.0111161, 0.0095506

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697199, upper bound: 0.0656270
time: 1.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695387, upper bound: 0.0656270
time: 1.54 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0053224, 0.0085902, 0.0049721, 0.0093792, -0.0040568, 0.0036181
1: -0.0033996, 0.0008834, -0.0044140, 0.0059430, -0.0093426, 0.0052973
2: 0.0098025, 0.0278248, 0.0100427, 0.0288263, -0.0190237, 0.0177820
3: -0.0051717, -0.0026018, -0.0082266, -0.0021920, -0.0029797, 0.0056248
4: -0.0038829, 0.0048978, -0.0068504, 0.0054958, -0.0093788, 0.0117482
5: -0.0013589, 0.0069528, -0.0051745, 0.0089981, -0.0103570, 0.0121272
6: 0.9835059, 0.9925473, 0.9806653, 0.9925188, -0.0090129, 0.0118819
7: -0.0181742, -0.0045170, -0.0215201, -0.0047054, -0.0134689, 0.0170031
8: -0.0149867, -0.0004131, -0.0168850, 0.0043936, -0.0193803, 0.0164718
9: -0.0064773, 0.0031064, -0.0063595, 0.0075252, -0.0140026, 0.0094659

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697200, upper bound: 0.0656273
time: 1.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695387, upper bound: 0.0692920
time: 2.01 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0052499, 0.0087534, 0.0052480, 0.0087578, -0.0035078, 0.0035054
1: -0.0036094, 0.0010080, -0.0036150, 0.0010426, -0.0046520, 0.0046231
2: 0.0095092, 0.0280319, 0.0104637, 0.0280375, -0.0185282, 0.0175682
3: -0.0058035, -0.0025170, -0.0058205, -0.0025148, -0.0032888, 0.0033035
4: -0.0044967, 0.0050249, -0.0045132, 0.0046113, -0.0091080, 0.0095381
5: -0.0021481, 0.0073758, -0.0021693, 0.0073872, -0.0095353, 0.0095451
6: 0.9829183, 0.9925821, 0.9829026, 0.9924688, -0.0095505, 0.0096796
7: -0.0188663, -0.0042869, -0.0188849, -0.0050356, -0.0138307, 0.0145979
8: -0.0153793, 0.0005810, -0.0153898, 0.0006077, -0.0159870, 0.0159708
9: -0.0066212, 0.0040204, -0.0061531, 0.0040449, -0.0106661, 0.0101734

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656895, upper bound: 0.0691902
time: 1.65 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656895, upper bound: 0.0691907
time: 1.66 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0052573, 0.0087369, 0.0050628, 0.0091748, -0.0039175, 0.0036740
1: -0.0035882, 0.0009149, -0.0041512, 0.0043310, -0.0079192, 0.0050661
2: 0.0095483, 0.0280110, 0.0104624, 0.0285668, -0.0190185, 0.0175485
3: -0.0057396, -0.0025256, -0.0074351, -0.0022982, -0.0034415, 0.0049095
4: -0.0044347, 0.0050080, -0.0060816, 0.0048315, -0.0092661, 0.0110896
5: -0.0020683, 0.0073331, -0.0041859, 0.0084682, -0.0105365, 0.0115190
6: 0.9829777, 0.9925774, 0.9814012, 0.9924690, -0.0094912, 0.0111762
7: -0.0187963, -0.0043176, -0.0206533, -0.0050346, -0.0137617, 0.0163357
8: -0.0153396, 0.0004806, -0.0163931, 0.0031483, -0.0184879, 0.0168737
9: -0.0066020, 0.0039280, -0.0061536, 0.0063804, -0.0129824, 0.0100816

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0656551, upper bound: 0.0663694
time: 1.88 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0627010, upper bound: 0.0663301
time: 1.68 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0050248, 0.0092606, 0.0055103, 0.0081807, -0.0031559, 0.0037502
1: -0.0042614, 0.0050073, -0.0028554, 0.0008601, -0.0051216, 0.0078628
2: 0.0096915, 0.0286757, 0.0099901, 0.0272875, -0.0175960, 0.0186856
3: -0.0077672, -0.0022536, -0.0047990, -0.0028216, -0.0049456, 0.0025454
4: -0.0064042, 0.0051102, -0.0032308, 0.0048165, -0.0112207, 0.0083410
5: -0.0046007, 0.0086905, -0.0012388, 0.0058556, -0.0104563, 0.0099292
6: 0.9810924, 0.9925604, 0.9845367, 0.9925250, -0.0114326, 0.0080237
7: -0.0210170, -0.0044299, -0.0174291, -0.0046641, -0.0163529, 0.0129992
8: -0.0165995, 0.0036707, -0.0139683, -0.0004729, -0.0161266, 0.0176391
9: -0.0065318, 0.0068607, -0.0063853, 0.0024155, -0.0089472, 0.0132461

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695924, upper bound: 0.0661440
time: 1.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695040, upper bound: 0.0661448
time: 1.73 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0050248, 0.0092606, 0.0052724, 0.0087028, -0.0036780, 0.0039881
1: -0.0042614, 0.0050073, -0.0035443, 0.0008382, -0.0050996, 0.0085516
2: 0.0096915, 0.0286757, 0.0101671, 0.0279676, -0.0182761, 0.0185086
3: -0.0077672, -0.0022536, -0.0056075, -0.0025433, -0.0052239, 0.0033539
4: -0.0064042, 0.0051102, -0.0043063, 0.0047398, -0.0111440, 0.0094165
5: -0.0046007, 0.0086905, -0.0019033, 0.0072446, -0.0118453, 0.0105938
6: 0.9810924, 0.9925604, 0.9831008, 0.9925041, -0.0114117, 0.0094596
7: -0.0210170, -0.0044299, -0.0186516, -0.0048029, -0.0162141, 0.0142217
8: -0.0165995, 0.0036707, -0.0152575, 0.0002727, -0.0168722, 0.0189282
9: -0.0065318, 0.0068607, -0.0062985, 0.0037369, -0.0102687, 0.0131593

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695924, upper bound: 0.0661437
time: 2.04 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0695040, upper bound: 0.0694313
time: 1.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0052499, 0.0087534, 0.0049400, 0.0094513, -0.0042014, 0.0038133
1: -0.0036094, 0.0010080, -0.0045067, 0.0065117, -0.0101211, 0.0055147
2: 0.0095092, 0.0280319, 0.0096972, 0.0289178, -0.0194086, 0.0183348
3: -0.0058035, -0.0025170, -0.0085059, -0.0021545, -0.0036490, 0.0059888
4: -0.0044967, 0.0050249, -0.0071217, 0.0057302, -0.0102269, 0.0121466
5: -0.0021481, 0.0073758, -0.0055233, 0.0091850, -0.0113331, 0.0128991
6: 0.9829183, 0.9925821, 0.9804056, 0.9924793, -0.0095610, 0.0121766
7: -0.0188663, -0.0042869, -0.0218260, -0.0049665, -0.0138998, 0.0175391
8: -0.0153793, 0.0005810, -0.0170585, 0.0048330, -0.0202123, 0.0176395
9: -0.0066212, 0.0040204, -0.0061962, 0.0079292, -0.0145504, 0.0102166

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656887, upper bound: 0.0690248
time: 1.86 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0656887, upper bound: 0.0690254
time: 1.53 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0052573, 0.0087369, 0.0047398, 0.0099023, -0.0046450, 0.0039971
1: -0.0035882, 0.0009149, -0.0050865, 0.0100678, -0.0136559, 0.0060014
2: 0.0095483, 0.0280110, 0.0063664, 0.0294902, -0.0199419, 0.0216446
3: -0.0057396, -0.0025256, -0.0102519, -0.0019203, -0.0038193, 0.0077263
4: -0.0044347, 0.0050080, -0.0088178, 0.0071958, -0.0116305, 0.0138257
5: -0.0020683, 0.0073331, -0.0077040, 0.0103540, -0.0124223, 0.0150371
6: 0.9829777, 0.9925774, 0.9787821, 0.9924834, -0.0095057, 0.0137953
7: -0.0187963, -0.0043176, -0.0237383, -0.0049388, -0.0138575, 0.0194208
8: -0.0153396, 0.0004806, -0.0181435, 0.0075803, -0.0229199, 0.0186240
9: -0.0066020, 0.0039280, -0.0062136, 0.0104547, -0.0170568, 0.0101416

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0656541, upper bound: 0.0662318
time: 1.53 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0627002, upper bound: 0.0661949
time: 1.60 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0050248, 0.0092606, 0.0052009, 0.0088638, -0.0038390, 0.0040596
1: -0.0042614, 0.0050073, -0.0037513, 0.0018787, -0.0061401, 0.0087587
2: 0.0096915, 0.0286757, 0.0098702, 0.0281720, -0.0184805, 0.0188055
3: -0.0077672, -0.0022536, -0.0062310, -0.0024597, -0.0053075, 0.0039774
4: -0.0064042, 0.0051102, -0.0049120, 0.0048685, -0.0112727, 0.0100221
5: -0.0046007, 0.0086905, -0.0026821, 0.0076620, -0.0122627, 0.0113725
6: 0.9810924, 0.9925604, 0.9825209, 0.9925392, -0.0114468, 0.0100395
7: -0.0210170, -0.0044299, -0.0193345, -0.0045700, -0.0164470, 0.0149046
8: -0.0165995, 0.0036707, -0.0156449, 0.0012537, -0.0178532, 0.0193157
9: -0.0065318, 0.0068607, -0.0064441, 0.0046388, -0.0111705, 0.0133049

Time for backsubstitution: 1.55 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.34 + 596.80 = 601.13 seconds
