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
execution time: IAR + RelationalAnalysis = 1.76 + 2.71 = 4.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0731673, upper bound: 0.0731670

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0715572, upper bound: 0.0707618
time: 1.63 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0707569, upper bound: 0.0707569
time: 1.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.44 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.44
Output dim: 6, lower bound: -0.0715572, upper bound: 0.0707618
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.44
Output dim: 6, lower bound: -0.0707569, upper bound: 0.0707569

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0048154, 0.0097321, 0.0045828, 0.0243426, -0.0195273, 0.0051492
1: -0.0048677, 0.0087257, -0.0056720, 0.0211549, -0.0260225, 0.0143977
2: 0.0076234, 0.0292742, 0.0024190, 0.0394507, -0.0318273, 0.0268552
3: -0.0095929, -0.0020087, -0.0105444, -0.0010774, -0.0085156, 0.0085357
4: -0.0081777, 0.0066427, -0.0098809, 0.0091565, -0.0173342, 0.0165236
5: -0.0068810, 0.0099128, -0.0112748, 0.0139717, -0.0208527, 0.0211876
6: 0.9793949, 0.9925498, 0.9184468, 0.9928610, -0.0134662, 0.0741031
7: -0.0230166, -0.0044999, -0.0241657, 0.0032827, -0.0262993, 0.0196658
8: -0.0177340, 0.0065434, -0.0217363, 0.0139040, -0.0316380, 0.0282797
9: -0.0064880, 0.0095016, -0.0066065, 0.0159613, -0.0224493, 0.0161081

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0688950, upper bound: 0.0681348
time: 1.38 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0688962, upper bound: 0.0680342
time: 1.29 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0043790, 0.0107150, 0.0047709, 0.0191038, -0.0147249, 0.0059441
1: -0.0061314, 0.0164768, -0.0053907, 0.0164531, -0.0225845, 0.0218676
2: 0.0003634, 0.0305219, 0.0042446, 0.0357179, -0.0353545, 0.0262773
3: -0.0133988, -0.0014982, -0.0101016, -0.0013864, -0.0120124, 0.0086034
4: -0.0118746, 0.0098373, -0.0091040, 0.0082363, -0.0201109, 0.0189413
5: -0.0116344, 0.0124608, -0.0096075, 0.0123564, -0.0239908, 0.0220683
6: 0.9758558, 0.9929727, 0.9403360, 0.9925823, -0.0167266, 0.0526367
7: -0.0271850, -0.0045998, -0.0236044, -0.0004229, -0.0267621, 0.0190046
8: -0.0200989, 0.0125317, -0.0203854, 0.0111763, -0.0312752, 0.0329171
9: -0.0064255, 0.0150066, -0.0065462, 0.0133714, -0.0197969, 0.0215528

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0680291, upper bound: 0.0681290
time: 1.52 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0680290, upper bound: 0.0680290
time: 1.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.78 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.78
Output dim: 6, lower bound: -0.0688950, upper bound: 0.0681348
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.78
Output dim: 6, lower bound: -0.0688962, upper bound: 0.0680342
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.78
Output dim: 6, lower bound: -0.0680291, upper bound: 0.0681290
NS_A2_B2, status: Status.VERIFIED, split count: 2, time: 4.78
Output dim: 6, lower bound: -0.0680290, upper bound: 0.0680290

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0048154, 0.0097321, 0.0050161, 0.0092801, -0.0044647, 0.0047160
1: -0.0048677, 0.0087257, -0.0042865, 0.0051613, -0.0100290, 0.0130122
2: 0.0076234, 0.0292742, 0.0096787, 0.0287005, -0.0210770, 0.0195955
3: -0.0095929, -0.0020087, -0.0078428, -0.0022435, -0.0073494, 0.0058341
4: -0.0081777, 0.0066427, -0.0064776, 0.0051736, -0.0133513, 0.0131203
5: -0.0068810, 0.0099128, -0.0046951, 0.0087411, -0.0156221, 0.0146079
6: 0.9793949, 0.9925498, 0.9810222, 0.9925619, -0.0131671, 0.0115277
7: -0.0230166, -0.0044999, -0.0210998, -0.0044199, -0.0185968, 0.0165999
8: -0.0177340, 0.0065434, -0.0166465, 0.0037897, -0.0215237, 0.0231899
9: -0.0064880, 0.0095016, -0.0065381, 0.0069701, -0.0134581, 0.0160397

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0685663, upper bound: 0.0677791
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0685663, upper bound: 0.0677790
time: 1.57 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.0048752, 0.0095974, 0.0050722, 0.0091536, -0.0042784, 0.0045251
1: -0.0046945, 0.0076634, -0.0041239, 0.0041639, -0.0088584, 0.0117873
2: 0.0086185, 0.0291032, 0.0093247, 0.0285399, -0.0199214, 0.0197785
3: -0.0090713, -0.0020787, -0.0073531, -0.0023092, -0.0067621, 0.0052744
4: -0.0076710, 0.0062049, -0.0060019, 0.0051049, -0.0127758, 0.0122068
5: -0.0062295, 0.0095636, -0.0040834, 0.0084132, -0.0146427, 0.0136470
6: 0.9798797, 0.9925494, 0.9814776, 0.9926040, -0.0127243, 0.0110719
7: -0.0224453, -0.0045025, -0.0205634, -0.0041422, -0.0183031, 0.0160609
8: -0.0174099, 0.0057227, -0.0163421, 0.0030192, -0.0204290, 0.0220649
9: -0.0064864, 0.0087471, -0.0067117, 0.0062617, -0.0127481, 0.0154588

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0687005, upper bound: 0.0680316
time: 1.48 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0687001, upper bound: 0.0678062
time: 1.90 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0043790, 0.0107150, 0.0050450, 0.0092151, -0.0048361, 0.0056701
1: -0.0061314, 0.0164768, -0.0042030, 0.0046488, -0.0107802, 0.0206798
2: 0.0003634, 0.0305219, 0.0097539, 0.0286179, -0.0282546, 0.0207680
3: -0.0133988, -0.0014982, -0.0075912, -0.0022772, -0.0111215, 0.0060930
4: -0.0118746, 0.0098373, -0.0062332, 0.0049624, -0.0168370, 0.0160705
5: -0.0116344, 0.0124608, -0.0043808, 0.0085726, -0.0202070, 0.0168417
6: 0.9758558, 0.9929727, 0.9812561, 0.9925531, -0.0166973, 0.0117165
7: -0.0271850, -0.0045998, -0.0208242, -0.0044788, -0.0227062, 0.0162244
8: -0.0200989, 0.0125317, -0.0164901, 0.0033937, -0.0234926, 0.0290218
9: -0.0064255, 0.0150066, -0.0065012, 0.0066061, -0.0130316, 0.0215078

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0677735, upper bound: 0.0677736
time: 1.59 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0677735, upper bound: 0.0677736
time: 1.55 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.03 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 6, lower bound: -0.0685663, upper bound: 0.0677791
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 6, lower bound: -0.0685663, upper bound: 0.0677790
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 6, lower bound: -0.0687005, upper bound: 0.0680316
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 6, lower bound: -0.0687001, upper bound: 0.0678062
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 5.03
Output dim: 6, lower bound: -0.0677735, upper bound: 0.0677736
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 5.03
Output dim: 6, lower bound: -0.0677735, upper bound: 0.0677736

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0050896, 0.0091146, 0.0050161, 0.0092801, -0.0041905, 0.0040985
1: -0.0040737, 0.0038561, -0.0042865, 0.0051613, -0.0092350, 0.0081426
2: 0.0097990, 0.0284903, 0.0096787, 0.0287005, -0.0189015, 0.0188116
3: -0.0072019, -0.0023294, -0.0078428, -0.0022435, -0.0049584, 0.0055134
4: -0.0058551, 0.0048993, -0.0064776, 0.0051736, -0.0110287, 0.0113770
5: -0.0038947, 0.0083120, -0.0046951, 0.0087411, -0.0126358, 0.0130071
6: 0.9816181, 0.9925477, 0.9810222, 0.9925619, -0.0109438, 0.0115255
7: -0.0203978, -0.0045142, -0.0210998, -0.0044199, -0.0159780, 0.0165856
8: -0.0162482, 0.0027813, -0.0166465, 0.0037897, -0.0200379, 0.0194278
9: -0.0064791, 0.0060431, -0.0065381, 0.0069701, -0.0134492, 0.0125812

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682501, upper bound: 0.0681306
time: 1.67 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682500, upper bound: 0.0676675
time: 1.50 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0051463, 0.0089867, 0.0050161, 0.0092801, -0.0041337, 0.0039706
1: -0.0039094, 0.0028480, -0.0042865, 0.0051613, -0.0090707, 0.0071345
2: 0.0094580, 0.0283281, 0.0096787, 0.0287005, -0.0192425, 0.0186494
3: -0.0067069, -0.0023958, -0.0078428, -0.0022435, -0.0044635, 0.0054470
4: -0.0053743, 0.0050471, -0.0064776, 0.0051736, -0.0105479, 0.0115247
5: -0.0032764, 0.0079807, -0.0046951, 0.0087411, -0.0120176, 0.0126758
6: 0.9820783, 0.9925881, 0.9810222, 0.9925619, -0.0104836, 0.0115659
7: -0.0198558, -0.0042467, -0.0210998, -0.0044199, -0.0154359, 0.0168531
8: -0.0159407, 0.0020025, -0.0166465, 0.0037897, -0.0197304, 0.0186490
9: -0.0066463, 0.0053272, -0.0065381, 0.0069701, -0.0136164, 0.0118652

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0685636, upper bound: 0.0676679
time: 1.39 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682500, upper bound: 0.0676675
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0048762, 0.0095952, 0.0051588, 0.0089585, -0.0040824, 0.0044364
1: -0.0046917, 0.0076465, -0.0038732, 0.0026258, -0.0073175, 0.0115196
2: 0.0086343, 0.0291005, 0.0097897, 0.0282923, -0.0196580, 0.0193108
3: -0.0090630, -0.0020798, -0.0065978, -0.0024105, -0.0066525, 0.0045181
4: -0.0076629, 0.0061979, -0.0052683, 0.0049034, -0.0125663, 0.0114662
5: -0.0062191, 0.0095580, -0.0031402, 0.0079076, -0.0141267, 0.0126983
6: 0.9798874, 0.9925488, 0.9821798, 0.9925488, -0.0126614, 0.0103690
7: -0.0224362, -0.0045066, -0.0197363, -0.0045069, -0.0179294, 0.0152297
8: -0.0174047, 0.0057097, -0.0158729, 0.0018309, -0.0192356, 0.0215825
9: -0.0064838, 0.0087351, -0.0064836, 0.0051694, -0.0116532, 0.0152187

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0680315
time: 1.46 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0680307
time: 1.48 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0048803, 0.0095858, 0.0048550, 0.0096429, -0.0047626, 0.0047308
1: -0.0046796, 0.0075724, -0.0047530, 0.0080223, -0.0127019, 0.0123253
2: 0.0087037, 0.0290886, 0.0082823, 0.0291610, -0.0204572, 0.0208062
3: -0.0090266, -0.0020847, -0.0092475, -0.0020551, -0.0069716, 0.0071629
4: -0.0076276, 0.0061674, -0.0078421, 0.0063528, -0.0139803, 0.0140095
5: -0.0061737, 0.0095337, -0.0064496, 0.0096816, -0.0158553, 0.0159833
6: 0.9799212, 0.9925457, 0.9797159, 0.9925621, -0.0126409, 0.0128298
7: -0.0223964, -0.0045276, -0.0226383, -0.0044196, -0.0179768, 0.0181107
8: -0.0173821, 0.0056524, -0.0175194, 0.0060000, -0.0233821, 0.0231718
9: -0.0064707, 0.0086825, -0.0065382, 0.0090020, -0.0154727, 0.0152207

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0678059
time: 1.68 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0678051
time: 1.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.03 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.03
Output dim: 6, lower bound: -0.0682501, upper bound: 0.0681306
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.03
Output dim: 6, lower bound: -0.0682500, upper bound: 0.0676675
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.03
Output dim: 6, lower bound: -0.0685636, upper bound: 0.0676679
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.03
Output dim: 6, lower bound: -0.0682500, upper bound: 0.0676675
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.03
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0680315
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.03
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0680307
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.03
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0678059
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.03
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0678051

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0050905, 0.0091124, 0.0051091, 0.0090706, -0.0039801, 0.0040033
1: -0.0040710, 0.0038392, -0.0040172, 0.0035094, -0.0075804, 0.0078564
2: 0.0098042, 0.0284876, 0.0101538, 0.0284345, -0.0186304, 0.0183338
3: -0.0071937, -0.0023306, -0.0070317, -0.0023523, -0.0048414, 0.0047011
4: -0.0058470, 0.0048971, -0.0056897, 0.0047456, -0.0105926, 0.0105868
5: -0.0038843, 0.0083065, -0.0036821, 0.0081981, -0.0120824, 0.0119886
6: 0.9816257, 0.9925471, 0.9817764, 0.9925056, -0.0108798, 0.0107707
7: -0.0203888, -0.0045183, -0.0202115, -0.0047925, -0.0155963, 0.0156932
8: -0.0162431, 0.0027683, -0.0161425, 0.0025135, -0.0187566, 0.0189108
9: -0.0064765, 0.0060311, -0.0063050, 0.0057969, -0.0122734, 0.0123362

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0684215, upper bound: 0.0676710
time: 1.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0684215, upper bound: 0.0676710
time: 1.59 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0050947, 0.0091031, 0.0048045, 0.0097567, -0.0046620, 0.0042986
1: -0.0040590, 0.0037658, -0.0048993, 0.0089195, -0.0129785, 0.0086650
2: 0.0098309, 0.0284758, 0.0074419, 0.0293054, -0.0194745, 0.0210339
3: -0.0071576, -0.0023354, -0.0096881, -0.0019959, -0.0051616, 0.0073527
4: -0.0058120, 0.0048855, -0.0082701, 0.0067226, -0.0125346, 0.0131556
5: -0.0038393, 0.0082823, -0.0069998, 0.0099765, -0.0138158, 0.0152822
6: 0.9816592, 0.9925438, 0.9793062, 0.9925203, -0.0108610, 0.0132376
7: -0.0203493, -0.0045392, -0.0231209, -0.0046950, -0.0156543, 0.0185816
8: -0.0162207, 0.0027116, -0.0177931, 0.0066932, -0.0229139, 0.0205047
9: -0.0064634, 0.0059790, -0.0063660, 0.0096392, -0.0161027, 0.0123450

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0664969, upper bound: 0.0621254
time: 1.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0677512, upper bound: 0.0670235
time: 1.49 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.0052305, 0.0087971, 0.0050171, 0.0092779, -0.0040474, 0.0037801
1: -0.0036656, 0.0013529, -0.0042837, 0.0051441, -0.0088097, 0.0056366
2: 0.0099281, 0.0280874, 0.0096839, 0.0286977, -0.0187695, 0.0184035
3: -0.0059728, -0.0024943, -0.0078343, -0.0022446, -0.0037282, 0.0053400
4: -0.0046612, 0.0048434, -0.0064694, 0.0051665, -0.0098277, 0.0113127
5: -0.0023596, 0.0074892, -0.0046845, 0.0087354, -0.0110950, 0.0121737
6: 0.9827610, 0.9925323, 0.9810300, 0.9925614, -0.0098004, 0.0115023
7: -0.0190517, -0.0046155, -0.0210905, -0.0044239, -0.0146278, 0.0164750
8: -0.0154845, 0.0008474, -0.0166412, 0.0037764, -0.0192608, 0.0174886
9: -0.0064157, 0.0042653, -0.0065355, 0.0069578, -0.0133736, 0.0108008

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0686978, upper bound: 0.0676677
time: 1.68 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0686978, upper bound: 0.0676677
time: 1.52 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: 0.0049261, 0.0094828, 0.0050214, 0.0092682, -0.0043421, 0.0044614
1: -0.0045472, 0.0067599, -0.0042713, 0.0050679, -0.0096150, 0.0110311
2: 0.0094647, 0.0289578, 0.0097108, 0.0286854, -0.0192207, 0.0192470
3: -0.0086277, -0.0021382, -0.0077969, -0.0022496, -0.0063781, 0.0056587
4: -0.0072400, 0.0058325, -0.0064330, 0.0051351, -0.0123752, 0.0122655
5: -0.0056754, 0.0092666, -0.0046378, 0.0087104, -0.0143858, 0.0139044
6: 0.9802923, 0.9925458, 0.9810647, 0.9925581, -0.0122657, 0.0114811
7: -0.0219594, -0.0045266, -0.0210496, -0.0044450, -0.0175144, 0.0165229
8: -0.0171342, 0.0050247, -0.0166180, 0.0037175, -0.0208517, 0.0216426
9: -0.0064713, 0.0081054, -0.0065223, 0.0069037, -0.0133750, 0.0146277

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0641587, upper bound: 0.0666528
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0680333, upper bound: 0.0670208
time: 1.53 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0050905, 0.0091124, 0.0051588, 0.0089585, -0.0038680, 0.0039536
1: -0.0040710, 0.0038392, -0.0038732, 0.0026258, -0.0066968, 0.0077124
2: 0.0098042, 0.0284876, 0.0097897, 0.0282923, -0.0184881, 0.0186980
3: -0.0071937, -0.0023306, -0.0065978, -0.0024105, -0.0047832, 0.0042673
4: -0.0058470, 0.0048971, -0.0052683, 0.0049034, -0.0107504, 0.0101654
5: -0.0038843, 0.0083065, -0.0031402, 0.0079076, -0.0117919, 0.0114467
6: 0.9816257, 0.9925471, 0.9821798, 0.9925488, -0.0109231, 0.0103673
7: -0.0203888, -0.0045183, -0.0197363, -0.0045069, -0.0158820, 0.0152180
8: -0.0162431, 0.0027683, -0.0158729, 0.0018309, -0.0180740, 0.0186412
9: -0.0064765, 0.0060311, -0.0064836, 0.0051694, -0.0116459, 0.0125148

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0680315
time: 1.47 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0680314
time: 1.52 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0051473, 0.0089847, 0.0051588, 0.0089585, -0.0038113, 0.0038259
1: -0.0039068, 0.0028320, -0.0038732, 0.0026258, -0.0065326, 0.0067052
2: 0.0094632, 0.0283255, 0.0097897, 0.0282923, -0.0188291, 0.0185359
3: -0.0066991, -0.0023969, -0.0065978, -0.0024105, -0.0042886, 0.0042009
4: -0.0053666, 0.0050449, -0.0052683, 0.0049034, -0.0102700, 0.0103132
5: -0.0032667, 0.0079754, -0.0031402, 0.0079076, -0.0111743, 0.0111156
6: 0.9820856, 0.9925875, 0.9821798, 0.9925488, -0.0104632, 0.0104077
7: -0.0198472, -0.0042508, -0.0197363, -0.0045069, -0.0153403, 0.0154855
8: -0.0159358, 0.0019902, -0.0158729, 0.0018309, -0.0177667, 0.0178630
9: -0.0066438, 0.0053158, -0.0064836, 0.0051694, -0.0118131, 0.0117994

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0680305
time: 1.77 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0680307
time: 1.68 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0050947, 0.0091031, 0.0048550, 0.0096429, -0.0045482, 0.0042481
1: -0.0040590, 0.0037658, -0.0047530, 0.0080223, -0.0120812, 0.0085188
2: 0.0098309, 0.0284758, 0.0082823, 0.0291610, -0.0193300, 0.0201935
3: -0.0071576, -0.0023354, -0.0092475, -0.0020551, -0.0051025, 0.0069121
4: -0.0058120, 0.0048855, -0.0078421, 0.0063528, -0.0121648, 0.0127276
5: -0.0038393, 0.0082823, -0.0064496, 0.0096816, -0.0135209, 0.0147319
6: 0.9816592, 0.9925438, 0.9797159, 0.9925621, -0.0109029, 0.0128279
7: -0.0203493, -0.0045392, -0.0226383, -0.0044196, -0.0159298, 0.0180991
8: -0.0162207, 0.0027116, -0.0175194, 0.0060000, -0.0222206, 0.0202309
9: -0.0064634, 0.0059790, -0.0065382, 0.0090020, -0.0154654, 0.0125172

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0660367, upper bound: 0.0612369
time: 1.57 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0675988, upper bound: 0.0671796
time: 1.51 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0051515, 0.0089751, 0.0048550, 0.0096429, -0.0044914, 0.0041201
1: -0.0038944, 0.0027565, -0.0047530, 0.0080223, -0.0119167, 0.0075095
2: 0.0094902, 0.0283133, 0.0082823, 0.0291610, -0.0196707, 0.0200310
3: -0.0066620, -0.0024019, -0.0092475, -0.0020551, -0.0046070, 0.0068457
4: -0.0053306, 0.0050331, -0.0078421, 0.0063528, -0.0116834, 0.0128753
5: -0.0032204, 0.0079506, -0.0064496, 0.0096816, -0.0129019, 0.0144001
6: 0.9821201, 0.9925843, 0.9797159, 0.9925621, -0.0104420, 0.0128684
7: -0.0198065, -0.0042720, -0.0226383, -0.0044196, -0.0153870, 0.0183663
8: -0.0159127, 0.0019318, -0.0175194, 0.0060000, -0.0219127, 0.0194512
9: -0.0066305, 0.0052622, -0.0065382, 0.0090020, -0.0156325, 0.0118004

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0660367, upper bound: 0.0612371
time: 1.68 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0675988, upper bound: 0.0671787
time: 1.67 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.11 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0684215, upper bound: 0.0676710
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0684215, upper bound: 0.0676710
NS_A1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0664969, upper bound: 0.0621254
NS_A1_B1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0677512, upper bound: 0.0670235
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0686978, upper bound: 0.0676677
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0686978, upper bound: 0.0676677
NS_A1_B1_A2_A2_A1, status: Status.VERIFIED, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0641587, upper bound: 0.0666528
NS_A1_B1_A2_A2_A2, status: Status.VERIFIED, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0680333, upper bound: 0.0670208
NS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0680315
NS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0680314
NS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0680305
NS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0682491, upper bound: 0.0680307
NS_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0660367, upper bound: 0.0612369
NS_A1_B2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0675988, upper bound: 0.0671796
NS_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0660367, upper bound: 0.0612371
NS_A1_B2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 5.11
Output dim: 6, lower bound: -0.0675988, upper bound: 0.0671787

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0051799, 0.0089113, 0.0051091, 0.0090706, -0.0038907, 0.0038022
1: -0.0038124, 0.0022530, -0.0040172, 0.0035094, -0.0073218, 0.0062702
2: 0.0102759, 0.0282323, 0.0101538, 0.0284345, -0.0181587, 0.0180785
3: -0.0064148, -0.0024350, -0.0070317, -0.0023523, -0.0040625, 0.0045967
4: -0.0050905, 0.0046927, -0.0056897, 0.0047456, -0.0098360, 0.0103824
5: -0.0029115, 0.0077850, -0.0036821, 0.0081981, -0.0111096, 0.0114671
6: 0.9823500, 0.9924911, 0.9817764, 0.9925056, -0.0101556, 0.0107147
7: -0.0195357, -0.0048883, -0.0202115, -0.0047925, -0.0147432, 0.0153232
8: -0.0157591, 0.0015428, -0.0161425, 0.0025135, -0.0182726, 0.0176853
9: -0.0062452, 0.0049045, -0.0063050, 0.0057969, -0.0120421, 0.0112096

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0665199, upper bound: 0.0628984
time: 1.57 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0677513, upper bound: 0.0675039
time: 1.63 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0048745, 0.0095990, 0.0051091, 0.0090706, -0.0041961, 0.0044899
1: -0.0046966, 0.0076762, -0.0040172, 0.0035094, -0.0082060, 0.0116934
2: 0.0086065, 0.0291053, 0.0101538, 0.0284345, -0.0198281, 0.0189515
3: -0.0090776, -0.0020778, -0.0070317, -0.0023523, -0.0067253, 0.0049539
4: -0.0076771, 0.0062102, -0.0056897, 0.0047456, -0.0124227, 0.0118999
5: -0.0062374, 0.0095678, -0.0036821, 0.0081981, -0.0144354, 0.0132499
6: 0.9798739, 0.9925063, 0.9817764, 0.9925056, -0.0126317, 0.0107299
7: -0.0224522, -0.0047878, -0.0202115, -0.0047925, -0.0176597, 0.0154236
8: -0.0174138, 0.0057326, -0.0161425, 0.0025135, -0.0199273, 0.0218751
9: -0.0063080, 0.0087562, -0.0063050, 0.0057969, -0.0121049, 0.0150612

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0643816, upper bound: 0.0672325
time: 1.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0677513, upper bound: 0.0675039
time: 1.86 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0052305, 0.0087971, 0.0051091, 0.0090706, -0.0038401, 0.0036880
1: -0.0036656, 0.0013529, -0.0040172, 0.0035094, -0.0071750, 0.0053701
2: 0.0099281, 0.0280874, 0.0101538, 0.0284345, -0.0185064, 0.0179336
3: -0.0059728, -0.0024943, -0.0070317, -0.0023523, -0.0036206, 0.0045374
4: -0.0046612, 0.0048434, -0.0056897, 0.0047456, -0.0094067, 0.0105331
5: -0.0023596, 0.0074892, -0.0036821, 0.0081981, -0.0105577, 0.0111712
6: 0.9827610, 0.9925323, 0.9817764, 0.9925056, -0.0097446, 0.0107560
7: -0.0190517, -0.0046155, -0.0202115, -0.0047925, -0.0142592, 0.0155960
8: -0.0154845, 0.0008474, -0.0161425, 0.0025135, -0.0179980, 0.0169899
9: -0.0064157, 0.0042653, -0.0063050, 0.0057969, -0.0122126, 0.0105703

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0665585, upper bound: 0.0621199
time: 1.58 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682090, upper bound: 0.0670212
time: 1.59 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0052305, 0.0087971, 0.0048045, 0.0097567, -0.0045261, 0.0039927
1: -0.0036656, 0.0013529, -0.0048993, 0.0089195, -0.0125851, 0.0062522
2: 0.0099281, 0.0280874, 0.0074419, 0.0293054, -0.0193773, 0.0206455
3: -0.0059728, -0.0024943, -0.0096881, -0.0019959, -0.0039769, 0.0071938
4: -0.0046612, 0.0048434, -0.0082701, 0.0067226, -0.0113838, 0.0131135
5: -0.0023596, 0.0074892, -0.0069998, 0.0099765, -0.0123361, 0.0144890
6: 0.9827610, 0.9925323, 0.9793062, 0.9925203, -0.0097593, 0.0132261
7: -0.0190517, -0.0046155, -0.0231209, -0.0046950, -0.0143568, 0.0185054
8: -0.0154845, 0.0008474, -0.0177931, 0.0066932, -0.0221777, 0.0186406
9: -0.0064157, 0.0042653, -0.0063660, 0.0096392, -0.0160550, 0.0106313

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0665585, upper bound: 0.0621198
time: 1.67 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682090, upper bound: 0.0670212
time: 1.64 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.0051799, 0.0089113, 0.0051588, 0.0089585, -0.0037787, 0.0037524
1: -0.0038124, 0.0022530, -0.0038732, 0.0026258, -0.0064382, 0.0061261
2: 0.0102759, 0.0282323, 0.0097897, 0.0282923, -0.0180164, 0.0184426
3: -0.0064148, -0.0024350, -0.0065978, -0.0024105, -0.0040043, 0.0041628
4: -0.0050905, 0.0046927, -0.0052683, 0.0049034, -0.0099938, 0.0099610
5: -0.0029115, 0.0077850, -0.0031402, 0.0079076, -0.0108191, 0.0109252
6: 0.9823500, 0.9924911, 0.9821798, 0.9925488, -0.0101988, 0.0103112
7: -0.0195357, -0.0048883, -0.0197363, -0.0045069, -0.0150289, 0.0148480
8: -0.0157591, 0.0015428, -0.0158729, 0.0018309, -0.0175900, 0.0174157
9: -0.0062452, 0.0049045, -0.0064836, 0.0051694, -0.0114145, 0.0113882

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B1_A1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0643045, upper bound: 0.0670580
time: 1.51 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0677449, upper bound: 0.0673920
time: 1.65 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.0048745, 0.0095990, 0.0051588, 0.0089585, -0.0040840, 0.0044401
1: -0.0046966, 0.0076762, -0.0038732, 0.0026258, -0.0073224, 0.0115493
2: 0.0086065, 0.0291053, 0.0097897, 0.0282923, -0.0196859, 0.0193156
3: -0.0090776, -0.0020778, -0.0065978, -0.0024105, -0.0066671, 0.0045200
4: -0.0076771, 0.0062102, -0.0052683, 0.0049034, -0.0125805, 0.0114785
5: -0.0062374, 0.0095678, -0.0031402, 0.0079076, -0.0141450, 0.0127080
6: 0.9798739, 0.9925063, 0.9821798, 0.9925488, -0.0126749, 0.0103264
7: -0.0224522, -0.0047878, -0.0197363, -0.0045069, -0.0179454, 0.0149484
8: -0.0174138, 0.0057326, -0.0158729, 0.0018309, -0.0192446, 0.0216055
9: -0.0063080, 0.0087562, -0.0064836, 0.0051694, -0.0114773, 0.0152398

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B1_A1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0643045, upper bound: 0.0670580
time: 1.46 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0677449, upper bound: 0.0673921
time: 1.52 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.0052305, 0.0087971, 0.0051588, 0.0089585, -0.0037280, 0.0036383
1: -0.0036656, 0.0013529, -0.0038732, 0.0026258, -0.0062914, 0.0052260
2: 0.0099281, 0.0280874, 0.0097897, 0.0282923, -0.0183642, 0.0182978
3: -0.0059728, -0.0024943, -0.0065978, -0.0024105, -0.0035624, 0.0041035
4: -0.0046612, 0.0048434, -0.0052683, 0.0049034, -0.0095646, 0.0101117
5: -0.0023596, 0.0074892, -0.0031402, 0.0079076, -0.0102672, 0.0106294
6: 0.9827610, 0.9925323, 0.9821798, 0.9925488, -0.0097879, 0.0103525
7: -0.0190517, -0.0046155, -0.0197363, -0.0045069, -0.0145449, 0.0151208
8: -0.0154845, 0.0008474, -0.0158729, 0.0018309, -0.0173153, 0.0167203
9: -0.0064157, 0.0042653, -0.0064836, 0.0051694, -0.0115851, 0.0107489

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0664284, upper bound: 0.0621080
time: 2.10 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0680354, upper bound: 0.0673914
time: 1.51 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: 0.0049261, 0.0094828, 0.0051588, 0.0089585, -0.0040324, 0.0043240
1: -0.0045472, 0.0067599, -0.0038732, 0.0026258, -0.0071730, 0.0106330
2: 0.0094647, 0.0289578, 0.0097897, 0.0282923, -0.0188276, 0.0191681
3: -0.0086277, -0.0021382, -0.0065978, -0.0024105, -0.0062172, 0.0044597
4: -0.0072400, 0.0058325, -0.0052683, 0.0049034, -0.0121434, 0.0111008
5: -0.0056754, 0.0092666, -0.0031402, 0.0079076, -0.0135830, 0.0124068
6: 0.9802923, 0.9925458, 0.9821798, 0.9925488, -0.0122565, 0.0103660
7: -0.0219594, -0.0045266, -0.0197363, -0.0045069, -0.0174526, 0.0152097
8: -0.0171342, 0.0050247, -0.0158729, 0.0018309, -0.0189650, 0.0208975
9: -0.0064713, 0.0081054, -0.0064836, 0.0051694, -0.0116407, 0.0145891

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B1_A2_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0641597, upper bound: 0.0669159
time: 1.35 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0680354, upper bound: 0.0673915
time: 1.51 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.72 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0665199, upper bound: 0.0628984
NS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0677513, upper bound: 0.0675039
NS_A1_B1_A1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0643816, upper bound: 0.0672325
NS_A1_B1_A1_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0677513, upper bound: 0.0675039
NS_A1_B1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0665585, upper bound: 0.0621199
NS_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0682090, upper bound: 0.0670212
NS_A1_B1_A2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0665585, upper bound: 0.0621198
NS_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0682090, upper bound: 0.0670212
NS_A1_B2_B1_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0643045, upper bound: 0.0670580
NS_A1_B2_B1_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0677449, upper bound: 0.0673920
NS_A1_B2_B1_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0643045, upper bound: 0.0670580
NS_A1_B2_B1_A1_A2_A2, status: Status.VERIFIED, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0677449, upper bound: 0.0673921
NS_A1_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0664284, upper bound: 0.0621080
NS_A1_B2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0680354, upper bound: 0.0673914
NS_A1_B2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0641597, upper bound: 0.0669159
NS_A1_B2_B1_A2_A2_A2, status: Status.VERIFIED, split count: 6, time: 4.72
Output dim: 6, lower bound: -0.0680354, upper bound: 0.0673915

## BFS NS instance: NS_A1_B1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0052305, 0.0087971, 0.0052724, 0.0087028, -0.0034722, 0.0035247
1: -0.0036656, 0.0013529, -0.0035443, 0.0008382, -0.0045038, 0.0048972
2: 0.0099281, 0.0280874, 0.0101671, 0.0279676, -0.0180395, 0.0179203
3: -0.0059728, -0.0024943, -0.0056075, -0.0025433, -0.0034295, 0.0031132
4: -0.0046612, 0.0048434, -0.0043063, 0.0047398, -0.0094010, 0.0091497
5: -0.0023596, 0.0074892, -0.0019033, 0.0072446, -0.0096042, 0.0093925
6: 0.9827610, 0.9925323, 0.9831008, 0.9925041, -0.0097431, 0.0094315
7: -0.0190517, -0.0046155, -0.0186516, -0.0048029, -0.0142488, 0.0140361
8: -0.0154845, 0.0008474, -0.0152575, 0.0002727, -0.0157572, 0.0161049
9: -0.0064157, 0.0042653, -0.0062985, 0.0037369, -0.0101526, 0.0105638

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A2_A1_B1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0679936, upper bound: 0.0674988
time: 1.81 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0679857, upper bound: 0.0668873
time: 1.49 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0052305, 0.0087971, 0.0049721, 0.0093792, -0.0041487, 0.0038251
1: -0.0036656, 0.0013529, -0.0044140, 0.0059430, -0.0096086, 0.0057669
2: 0.0099281, 0.0280874, 0.0100427, 0.0288263, -0.0188981, 0.0180447
3: -0.0059728, -0.0024943, -0.0082266, -0.0021920, -0.0037808, 0.0057323
4: -0.0046612, 0.0048434, -0.0068504, 0.0054958, -0.0101570, 0.0116938
5: -0.0023596, 0.0074892, -0.0051745, 0.0089981, -0.0113576, 0.0126636
6: 0.9827610, 0.9925323, 0.9806653, 0.9925188, -0.0097578, 0.0118670
7: -0.0190517, -0.0046155, -0.0215201, -0.0047054, -0.0143463, 0.0169047
8: -0.0154845, 0.0008474, -0.0168850, 0.0043936, -0.0198781, 0.0177324
9: -0.0064157, 0.0042653, -0.0063595, 0.0075252, -0.0139410, 0.0106248

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681936, upper bound: 0.0666833
time: 1.73 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0679834, upper bound: 0.0666833
time: 1.52 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.12 seconds
NS_A1_B1_A2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.12
Output dim: 6, lower bound: -0.0679936, upper bound: 0.0674988
NS_A1_B1_A2_A1_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 5.12
Output dim: 6, lower bound: -0.0679857, upper bound: 0.0668873
NS_A1_B1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.12
Output dim: 6, lower bound: -0.0681936, upper bound: 0.0666833
NS_A1_B1_A2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.12
Output dim: 6, lower bound: -0.0679834, upper bound: 0.0666833

## BFS NS instance: NS_A1_B1_A2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0053030, 0.0086338, 0.0049721, 0.0093792, -0.0040762, 0.0036617
1: -0.0034556, 0.0008302, -0.0044140, 0.0059430, -0.0093986, 0.0052441
2: 0.0102318, 0.0278801, 0.0100427, 0.0288263, -0.0185945, 0.0178374
3: -0.0053405, -0.0025791, -0.0082266, -0.0021920, -0.0031485, 0.0056475
4: -0.0040470, 0.0047118, -0.0068504, 0.0054958, -0.0095428, 0.0115622
5: -0.0015699, 0.0070658, -0.0051745, 0.0089981, -0.0105679, 0.0122403
6: 0.9833489, 0.9924963, 0.9806653, 0.9925188, -0.0091699, 0.0118310
7: -0.0183592, -0.0048537, -0.0215201, -0.0047054, -0.0136538, 0.0166664
8: -0.0150916, -0.0001474, -0.0168850, 0.0043936, -0.0194852, 0.0167375
9: -0.0062668, 0.0033507, -0.0063595, 0.0075252, -0.0137920, 0.0097102

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of NS_A1_B1_A2_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0673425, upper bound: 0.0637727
time: 1.58 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0650660, upper bound: 0.0637730
time: 1.53 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 4.94 seconds
NS_A1_B1_A2_A1_B2_B2_A1_A1, status: Status.VERIFIED, split count: 8, time: 4.94
Output dim: 6, lower bound: -0.0673425, upper bound: 0.0637727
NS_A1_B1_A2_A1_B2_B2_A1_A2, status: Status.VERIFIED, split count: 8, time: 4.94
Output dim: 6, lower bound: -0.0650660, upper bound: 0.0637730

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.47 + 143.91 = 148.38 seconds
