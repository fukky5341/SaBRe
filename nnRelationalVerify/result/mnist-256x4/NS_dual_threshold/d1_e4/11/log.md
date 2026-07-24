## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.014896192999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959)
1: (0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687)
2: (-0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327574, 0.0327574)
3: (-0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814)
4: (-0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126)
5: (-0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610)
6: (-0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640)
7: (-0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709)
8: (-0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398314, 0.0398314)
9: (-0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.71 + 3.50 = 5.21 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0153569, upper bound: 0.0153569

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149782, upper bound: 0.0149906
time: 2.22 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153279, upper bound: 0.0153279
time: 2.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.91 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.91
Output dim: 1, lower bound: -0.0149782, upper bound: 0.0149906
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.91
Output dim: 1, lower bound: -0.0153279, upper bound: 0.0153279

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0039219, 0.0043159, -0.0042816, 0.0048514, -0.0087734, 0.0085975
1: 0.9920992, 1.0048027, 0.9921645, 1.0055166, -0.0134174, 0.0126382
2: -0.0225655, 0.0090017, -0.0218784, 0.0095092, -0.0311888, 0.0300190
3: -0.0014010, 0.0050735, -0.0011522, 0.0050280, -0.0064291, 0.0062257
4: -0.0102479, 0.0162516, -0.0108754, 0.0157085, -0.0259564, 0.0271270
5: -0.0043830, 0.0139313, -0.0039116, 0.0147508, -0.0191338, 0.0178429
6: -0.0069538, 0.0078800, -0.0067152, 0.0074441, -0.0143979, 0.0145952
7: -0.0118847, -0.0019086, -0.0121325, -0.0016477, -0.0102370, 0.0102239
8: -0.0116479, 0.0278999, -0.0120331, 0.0269971, -0.0384000, 0.0396782
9: -0.0093898, 0.0059528, -0.0102590, 0.0061667, -0.0155565, 0.0162118

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149367, upper bound: 0.0149719
time: 2.16 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149557, upper bound: 0.0149719
time: 2.50 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0045032, 0.0051814, -0.0076237, 0.0055722, -0.0100754, 0.0128051
1: 0.9921629, 1.0059568, 0.9907092, 1.0064780, -0.0143150, 0.0152475
2: -0.0218946, 0.0098218, -0.0219780, 0.0113110, -0.0326251, 0.0309725
3: -0.0011580, 0.0050291, -0.0027511, 0.0050303, -0.0061883, 0.0077802
4: -0.0112621, 0.0157213, -0.0121325, 0.0157801, -0.0270422, 0.0278538
5: -0.0039227, 0.0152558, -0.0040185, 0.0179425, -0.0218651, 0.0192743
6: -0.0067209, 0.0074544, -0.0067780, 0.0077860, -0.0145069, 0.0142323
7: -0.0122852, -0.0014870, -0.0124661, 0.0050048, -0.0172900, 0.0109791
8: -0.0122705, 0.0270184, -0.0125516, 0.0275235, -0.0395434, 0.0393292
9: -0.0107945, 0.0062985, -0.0122965, 0.0064546, -0.0172492, 0.0185950

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152916, upper bound: 0.0153099
time: 2.70 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153099, upper bound: 0.0153099
time: 2.38 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 6.68 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 6.68
Output dim: 1, lower bound: -0.0149367, upper bound: 0.0149719
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 6.68
Output dim: 1, lower bound: -0.0149557, upper bound: 0.0149719
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 6.68
Output dim: 1, lower bound: -0.0152916, upper bound: 0.0153099
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 6.68
Output dim: 1, lower bound: -0.0153099, upper bound: 0.0153099

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -0.0033057, 0.0033985, -0.0040323, 0.0044803, -0.0077861, 0.0074309
1: 0.9922322, 1.0035791, 0.9922205, 1.0050219, -0.0127897, 0.0113587
2: -0.0211652, 0.0081323, -0.0212887, 0.0091575, -0.0294581, 0.0285580
3: -0.0008939, 0.0049808, -0.0009386, 0.0049890, -0.0058829, 0.0059194
4: -0.0091729, 0.0151449, -0.0104405, 0.0152424, -0.0244153, 0.0255854
5: -0.0034223, 0.0125273, -0.0035070, 0.0141829, -0.0176052, 0.0160343
6: -0.0064677, 0.0069917, -0.0065106, 0.0070699, -0.0135377, 0.0135022
7: -0.0114602, -0.0023556, -0.0119608, -0.0018285, -0.0096316, 0.0096052
8: -0.0109879, 0.0260601, -0.0117661, 0.0262223, -0.0369665, 0.0375804
9: -0.0079007, 0.0055863, -0.0096566, 0.0060185, -0.0139191, 0.0152429

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149062, upper bound: 0.0149537
time: 2.44 seconds

## Relational analysis of NS_A1_A1_A2

### Relational analysis result of NS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149119, upper bound: 0.0149495
time: 2.61 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -0.0036940, 0.0039766, -0.0042070, 0.0047404, -0.0084344, 0.0081836
1: 0.9921462, 1.0043498, 0.9921793, 1.0053686, -0.0132224, 0.0121706
2: -0.0220708, 0.0086802, -0.0217233, 0.0094039, -0.0304939, 0.0295665
3: -0.0012219, 0.0050408, -0.0010960, 0.0050178, -0.0062396, 0.0061367
4: -0.0098503, 0.0158606, -0.0107453, 0.0155859, -0.0254362, 0.0266059
5: -0.0040436, 0.0134120, -0.0038051, 0.0145809, -0.0186244, 0.0172171
6: -0.0067820, 0.0075662, -0.0066614, 0.0073457, -0.0141277, 0.0142276
7: -0.0117277, -0.0020739, -0.0120811, -0.0017018, -0.0100259, 0.0100072
8: -0.0114038, 0.0272499, -0.0119532, 0.0267933, -0.0379603, 0.0389236
9: -0.0088390, 0.0058172, -0.0100787, 0.0061223, -0.0149614, 0.0158960

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149175, upper bound: 0.0149041
time: 2.21 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148999, upper bound: 0.0149041
time: 2.16 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.0038920, 0.0042714, -0.0071427, 0.0052024, -0.0090944, 0.0114141
1: 0.9922981, 1.0047433, 0.9909374, 1.0059847, -0.0136865, 0.0138059
2: -0.0204718, 0.0089596, -0.0213816, 0.0108700, -0.0307670, 0.0295110
3: -0.0006427, 0.0049349, -0.0024212, 0.0049912, -0.0056339, 0.0073561
4: -0.0101958, 0.0145968, -0.0116639, 0.0153103, -0.0255061, 0.0262607
5: -0.0029466, 0.0138632, -0.0036087, 0.0172191, -0.0201657, 0.0174719
6: -0.0062270, 0.0065518, -0.0065675, 0.0073883, -0.0136153, 0.0131192
7: -0.0118641, -0.0019303, -0.0122949, 0.0043070, -0.0161711, 0.0103646
8: -0.0116159, 0.0251491, -0.0122855, 0.0267127, -0.0380788, 0.0372027
9: -0.0093176, 0.0059350, -0.0116355, 0.0063069, -0.0156245, 0.0175705

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152425, upper bound: 0.0152923
time: 2.79 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152413, upper bound: 0.0152581
time: 2.65 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.0042770, 0.0048446, -0.0074827, 0.0054604, -0.0097374, 0.0123273
1: 0.9922099, 1.0055078, 0.9907764, 1.0063289, -0.0141191, 0.0147314
2: -0.0214006, 0.0095027, -0.0218208, 0.0111790, -0.0319001, 0.0305210
3: -0.0009791, 0.0049964, -0.0026558, 0.0050200, -0.0059992, 0.0076522
4: -0.0108674, 0.0153309, -0.0119916, 0.0156563, -0.0265238, 0.0273225
5: -0.0035838, 0.0147404, -0.0039103, 0.0177265, -0.0213103, 0.0186507
6: -0.0065494, 0.0071410, -0.0067223, 0.0076800, -0.0142294, 0.0138633
7: -0.0121294, -0.0016511, -0.0124144, 0.0048024, -0.0169318, 0.0107633
8: -0.0120282, 0.0263694, -0.0124712, 0.0273101, -0.0390968, 0.0385767
9: -0.0102479, 0.0061640, -0.0120979, 0.0064100, -0.0166579, 0.0182619

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152654, upper bound: 0.0152943
time: 2.38 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152654, upper bound: 0.0152654
time: 2.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.68 seconds
NS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 6.68
Output dim: 1, lower bound: -0.0149062, upper bound: 0.0149537
NS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 6.68
Output dim: 1, lower bound: -0.0149119, upper bound: 0.0149495
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 6.68
Output dim: 1, lower bound: -0.0149175, upper bound: 0.0149041
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 6.68
Output dim: 1, lower bound: -0.0148999, upper bound: 0.0149041
NS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 6.68
Output dim: 1, lower bound: -0.0152425, upper bound: 0.0152923
NS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 6.68
Output dim: 1, lower bound: -0.0152413, upper bound: 0.0152581
NS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 6.68
Output dim: 1, lower bound: -0.0152654, upper bound: 0.0152943
NS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 6.68
Output dim: 1, lower bound: -0.0152654, upper bound: 0.0152654

## BFS NS instance: NS_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0026420, 0.0024104, -0.0039275, 0.0043242, -0.0069662, 0.0063379
1: 0.9923469, 1.0022612, 0.9922634, 1.0048138, -0.0124669, 0.0099978
2: -0.0199581, 0.0071959, -0.0208371, 0.0090096, -0.0281226, 0.0271761
3: -0.0004566, 0.0049009, -0.0007750, 0.0049591, -0.0054157, 0.0056760
4: -0.0080149, 0.0141908, -0.0102576, 0.0148855, -0.0229004, 0.0244485
5: -0.0025942, 0.0110151, -0.0031972, 0.0139440, -0.0165382, 0.0142123
6: -0.0060487, 0.0062259, -0.0063538, 0.0067835, -0.0128322, 0.0125797
7: -0.0110029, -0.0028371, -0.0118886, -0.0019046, -0.0090983, 0.0090515
8: -0.0102771, 0.0244741, -0.0116539, 0.0256290, -0.0356657, 0.0358849
9: -0.0062968, 0.0051916, -0.0094032, 0.0059561, -0.0122529, 0.0145948

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_A1_A1_B1

### Relational analysis result of NS_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148567, upper bound: 0.0148830
time: 2.62 seconds

## Relational analysis of NS_A1_A1_A1_B2

### Relational analysis result of NS_A1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148495, upper bound: 0.0148836
time: 2.56 seconds

## BFS NS instance: NS_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0031159, 0.0031159, -0.0039686, 0.0043854, -0.0075013, 0.0070845
1: 0.9922701, 1.0032022, 0.9922358, 1.0048952, -0.0126252, 0.0109664
2: -0.0207673, 0.0078645, -0.0211280, 0.0090676, -0.0288827, 0.0281292
3: -0.0007497, 0.0049545, -0.0008804, 0.0049784, -0.0057281, 0.0058348
4: -0.0088417, 0.0148303, -0.0103294, 0.0151154, -0.0239571, 0.0251597
5: -0.0031493, 0.0120947, -0.0033967, 0.0140377, -0.0171870, 0.0154915
6: -0.0063296, 0.0067392, -0.0064548, 0.0069680, -0.0132976, 0.0131940
7: -0.0113294, -0.0024933, -0.0119169, -0.0018748, -0.0094546, 0.0094236
8: -0.0107846, 0.0255373, -0.0116979, 0.0260112, -0.0365538, 0.0369648
9: -0.0074419, 0.0054734, -0.0095026, 0.0059805, -0.0134225, 0.0149760

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_A1_A2_B1

### Relational analysis result of NS_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148677, upper bound: 0.0148826
time: 2.25 seconds

## Relational analysis of NS_A1_A1_A2_B2

### Relational analysis result of NS_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148495, upper bound: 0.0148832
time: 2.81 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0035346, 0.0037394, -0.0037733, 0.0040947, -0.0076293, 0.0075127
1: 0.9921730, 1.0040337, 0.9922529, 1.0045074, -0.0123344, 0.0117807
2: -0.0217882, 0.0084553, -0.0209470, 0.0087920, -0.0296128, 0.0285661
3: -0.0011195, 0.0050221, -0.0008148, 0.0049664, -0.0060858, 0.0058369
4: -0.0095722, 0.0156372, -0.0099886, 0.0149723, -0.0245446, 0.0256258
5: -0.0038497, 0.0130489, -0.0032726, 0.0135927, -0.0174424, 0.0163215
6: -0.0066839, 0.0073869, -0.0063920, 0.0068532, -0.0135371, 0.0137788
7: -0.0116179, -0.0021895, -0.0117823, -0.0020164, -0.0096015, 0.0095928
8: -0.0112331, 0.0268786, -0.0114887, 0.0257733, -0.0367715, 0.0380937
9: -0.0084539, 0.0057225, -0.0090306, 0.0058644, -0.0143183, 0.0147531

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148909, upper bound: 0.0148894
time: 2.56 seconds

## Relational analysis of NS_A1_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148995, upper bound: 0.0148888
time: 2.40 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0035575, 0.0037734, -0.0050355, 0.0059739, -0.0095314, 0.0088090
1: 0.9921656, 1.0040791, 0.9922268, 1.0070139, -0.0148483, 0.0118523
2: -0.0218673, 0.0084876, -0.0212224, 0.0105729, -0.0314760, 0.0288734
3: -0.0011481, 0.0050273, -0.0009146, 0.0049846, -0.0061327, 0.0059419
4: -0.0096122, 0.0156997, -0.0121908, 0.0151900, -0.0248022, 0.0278905
5: -0.0039039, 0.0131010, -0.0034615, 0.0164686, -0.0203726, 0.0165626
6: -0.0067114, 0.0074370, -0.0064875, 0.0070279, -0.0137393, 0.0139246
7: -0.0116337, -0.0021729, -0.0126520, -0.0011008, -0.0105328, 0.0104790
8: -0.0112576, 0.0269825, -0.0128406, 0.0261352, -0.0371560, 0.0395513
9: -0.0085092, 0.0057361, -0.0120810, 0.0066151, -0.0151243, 0.0178170

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148791, upper bound: 0.0148896
time: 2.51 seconds

## Relational analysis of NS_A1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148843, upper bound: 0.0148889
time: 2.40 seconds

## BFS NS instance: NS_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0034359, 0.0035923, -0.0067057, 0.0049651, -0.0084010, 0.0102980
1: 0.9923679, 1.0038375, 0.9911687, 1.0056683, -0.0133004, 0.0126688
2: -0.0197378, 0.0083160, -0.0211076, 0.0105368, -0.0296771, 0.0285922
3: -0.0003768, 0.0048863, -0.0021264, 0.0049735, -0.0053504, 0.0070128
4: -0.0094000, 0.0140167, -0.0113459, 0.0150949, -0.0244949, 0.0253626
5: -0.0024430, 0.0128239, -0.0034184, 0.0166572, -0.0191003, 0.0162423
6: -0.0059723, 0.0060861, -0.0064691, 0.0071887, -0.0131610, 0.0125553
7: -0.0115499, -0.0022612, -0.0121852, 0.0035759, -0.0151257, 0.0099240
8: -0.0111274, 0.0241847, -0.0121149, 0.0263204, -0.0371989, 0.0360676
9: -0.0082152, 0.0056637, -0.0111706, 0.0062121, -0.0144274, 0.0168343

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A2_A1_A1_A1

### Relational analysis result of NS_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152189, upper bound: 0.0152783
time: 2.66 seconds

## Relational analysis of NS_A2_A1_A1_A2

### Relational analysis result of NS_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152286, upper bound: 0.0152760
time: 2.73 seconds

## BFS NS instance: NS_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0045597, 0.0052655, -0.0067190, 0.0049858, -0.0095455, 0.0119845
1: 0.9923490, 1.0060689, 0.9911696, 1.0056959, -0.0133469, 0.0148993
2: -0.0199356, 0.0099016, -0.0211480, 0.0105564, -0.0298895, 0.0302246
3: -0.0004485, 0.0048994, -0.0021343, 0.0049762, -0.0054247, 0.0070337
4: -0.0113607, 0.0141731, -0.0113702, 0.0151268, -0.0264875, 0.0255433
5: -0.0025788, 0.0153845, -0.0034455, 0.0166878, -0.0192666, 0.0188300
6: -0.0060409, 0.0062116, -0.0064830, 0.0072121, -0.0132531, 0.0126946
7: -0.0123242, -0.0014460, -0.0121947, 0.0035854, -0.0159096, 0.0107488
8: -0.0123310, 0.0244446, -0.0121297, 0.0263705, -0.0384525, 0.0363366
9: -0.0109311, 0.0063321, -0.0112034, 0.0062204, -0.0171515, 0.0175355

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_A1_A2_B1

### Relational analysis result of NS_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148785, upper bound: 0.0148939
time: 2.50 seconds

## Relational analysis of NS_A2_A1_A2_B2

### Relational analysis result of NS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148785, upper bound: 0.0152560
time: 3.00 seconds

## BFS NS instance: NS_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0038372, 0.0041898, -0.0070442, 0.0052213, -0.0090584, 0.0112339
1: 0.9922845, 1.0046344, 0.9910119, 1.0060101, -0.0137256, 0.0136225
2: -0.0206156, 0.0088822, -0.0215429, 0.0108438, -0.0307697, 0.0296213
3: -0.0006948, 0.0049444, -0.0023608, 0.0050021, -0.0056969, 0.0073053
4: -0.0101001, 0.0147105, -0.0116712, 0.0154380, -0.0255380, 0.0263817
5: -0.0030453, 0.0137382, -0.0037178, 0.0171623, -0.0202075, 0.0174560
6: -0.0062770, 0.0066430, -0.0066226, 0.0074787, -0.0137556, 0.0132656
7: -0.0118263, -0.0019701, -0.0123037, 0.0040697, -0.0158960, 0.0103336
8: -0.0115571, 0.0253380, -0.0122992, 0.0269122, -0.0382292, 0.0373776
9: -0.0091850, 0.0059024, -0.0116306, 0.0063145, -0.0154994, 0.0175330

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A2_A2_A1_A1

### Relational analysis result of NS_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152458, upper bound: 0.0152800
time: 2.74 seconds

## Relational analysis of NS_A2_A2_A1_A2

### Relational analysis result of NS_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152522, upper bound: 0.0152778
time: 2.43 seconds

## BFS NS instance: NS_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0051005, 0.0060706, -0.0070671, 0.0052478, -0.0103482, 0.0131377
1: 0.9922565, 1.0071428, 0.9910083, 1.0060455, -0.0137889, 0.0161345
2: -0.0209093, 0.0106645, -0.0215933, 0.0108711, -0.0310877, 0.0314620
3: -0.0008012, 0.0049639, -0.0023754, 0.0050054, -0.0058066, 0.0073393
4: -0.0123041, 0.0149426, -0.0117032, 0.0154778, -0.0277818, 0.0266458
5: -0.0032468, 0.0166166, -0.0037521, 0.0172058, -0.0204525, 0.0203687
6: -0.0063789, 0.0068294, -0.0066401, 0.0075089, -0.0138878, 0.0134695
7: -0.0126967, -0.0010537, -0.0123160, 0.0040946, -0.0167913, 0.0112622
8: -0.0129101, 0.0257239, -0.0123182, 0.0269775, -0.0396494, 0.0377820
9: -0.0122379, 0.0066537, -0.0116745, 0.0063250, -0.0185629, 0.0183283

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_A2_A2_B1

### Relational analysis result of NS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149041, upper bound: 0.0148999
time: 2.57 seconds

## Relational analysis of NS_A2_A2_A2_B2

### Relational analysis result of NS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149041, upper bound: 0.0152631
time: 2.53 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 7.13 seconds
NS_A1_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0148567, upper bound: 0.0148830
NS_A1_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0148495, upper bound: 0.0148836
NS_A1_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0148677, upper bound: 0.0148826
NS_A1_A1_A2_B2, status: Status.VERIFIED, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0148495, upper bound: 0.0148832
NS_A1_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0148909, upper bound: 0.0148894
NS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0148995, upper bound: 0.0148888
NS_A1_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0148791, upper bound: 0.0148896
NS_A1_A2_B2_A2, status: Status.VERIFIED, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0148843, upper bound: 0.0148889
NS_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0152189, upper bound: 0.0152783
NS_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0152286, upper bound: 0.0152760
NS_A2_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0148785, upper bound: 0.0148939
NS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0148785, upper bound: 0.0152560
NS_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0152458, upper bound: 0.0152800
NS_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0152522, upper bound: 0.0152778
NS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0149041, upper bound: 0.0148999
NS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.13
Output dim: 1, lower bound: -0.0149041, upper bound: 0.0152631

## BFS NS instance: NS_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0033548, 0.0034716, -0.0037092, 0.0039993, -0.0073541, 0.0071808
1: 0.9922141, 1.0036763, 0.9922686, 1.0043802, -0.0121661, 0.0114077
2: -0.0213559, 0.0082016, -0.0207825, 0.0087016, -0.0290111, 0.0281479
3: -0.0009629, 0.0049934, -0.0007552, 0.0049555, -0.0059184, 0.0057487
4: -0.0092585, 0.0152955, -0.0098768, 0.0148423, -0.0241008, 0.0251723
5: -0.0035531, 0.0126391, -0.0031597, 0.0134467, -0.0169998, 0.0157988
6: -0.0065339, 0.0071126, -0.0063349, 0.0067488, -0.0132828, 0.0134475
7: -0.0114940, -0.0023200, -0.0117382, -0.0020629, -0.0094311, 0.0094182
8: -0.0110405, 0.0263106, -0.0114201, 0.0255572, -0.0363610, 0.0374396
9: -0.0080193, 0.0056155, -0.0088758, 0.0058263, -0.0138456, 0.0144913

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_A2_B1_A2_A1

### Relational analysis result of NS_A1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148281, upper bound: 0.0148175
time: 2.45 seconds

## Relational analysis of NS_A1_A2_B1_A2_A2

### Relational analysis result of NS_A1_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148464, upper bound: 0.0148367
time: 2.21 seconds

## BFS NS instance: NS_A2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0028011, 0.0026472, -0.0063440, 0.0048021, -0.0076031, 0.0089912
1: 0.9924785, 1.0025769, 0.9913778, 1.0054511, -0.0129725, 0.0111991
2: -0.0185721, 0.0074203, -0.0206514, 0.0102881, -0.0282548, 0.0272319
3: 0.0000454, 0.0048092, -0.0018378, 0.0049437, -0.0048983, 0.0066470
4: -0.0082924, 0.0130954, -0.0111181, 0.0147352, -0.0230276, 0.0242135
5: -0.0016433, 0.0113775, -0.0031029, 0.0162332, -0.0178765, 0.0144804
6: -0.0055676, 0.0053466, -0.0063082, 0.0068749, -0.0124426, 0.0116547
7: -0.0111125, -0.0027217, -0.0121097, 0.0029530, -0.0140655, 0.0093880
8: -0.0104474, 0.0226531, -0.0119976, 0.0256865, -0.0358879, 0.0344234
9: -0.0066811, 0.0052862, -0.0108341, 0.0061470, -0.0128281, 0.0161203

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_A1_A1_A1_A1

### Relational analysis result of NS_A2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151358, upper bound: 0.0151886
time: 2.81 seconds

## Relational analysis of NS_A2_A1_A1_A1_A2

### Relational analysis result of NS_A2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151751, upper bound: 0.0152327
time: 2.56 seconds

## BFS NS instance: NS_A2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0032421, 0.0033038, -0.0064832, 0.0048649, -0.0081070, 0.0097870
1: 0.9924071, 1.0034529, 0.9912946, 1.0055349, -0.0131278, 0.0121583
2: -0.0193251, 0.0080426, -0.0209456, 0.0103829, -0.0290089, 0.0281534
3: -0.0002274, 0.0048590, -0.0019644, 0.0049630, -0.0051904, 0.0068234
4: -0.0090619, 0.0136905, -0.0112065, 0.0149674, -0.0240293, 0.0248970
5: -0.0021599, 0.0123824, -0.0033059, 0.0163939, -0.0185538, 0.0156883
6: -0.0058290, 0.0058243, -0.0064114, 0.0070719, -0.0129008, 0.0122357
7: -0.0114163, -0.0024017, -0.0121388, 0.0031877, -0.0146041, 0.0097370
8: -0.0109198, 0.0236424, -0.0120428, 0.0260895, -0.0367615, 0.0354306
9: -0.0077470, 0.0055485, -0.0109629, 0.0061721, -0.0139191, 0.0165114

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_A1_A1_A2_A1

### Relational analysis result of NS_A2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151413, upper bound: 0.0151860
time: 2.68 seconds

## Relational analysis of NS_A2_A1_A1_A2_A2

### Relational analysis result of NS_A2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151839, upper bound: 0.0152308
time: 2.94 seconds

## BFS NS instance: NS_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0045597, 0.0052655, -0.0041066, 0.0045909, -0.0091506, 0.0093721
1: 0.9923490, 1.0060689, 0.9922404, 1.0051693, -0.0128202, 0.0138285
2: -0.0199356, 0.0099016, -0.0210789, 0.0092623, -0.0283447, 0.0301105
3: -0.0004485, 0.0048994, -0.0008626, 0.0049751, -0.0054236, 0.0057620
4: -0.0113607, 0.0141731, -0.0105701, 0.0150766, -0.0264373, 0.0247432
5: -0.0025788, 0.0153845, -0.0033631, 0.0143520, -0.0169308, 0.0187476
6: -0.0060409, 0.0062116, -0.0064378, 0.0069369, -0.0129779, 0.0126494
7: -0.0123242, -0.0014460, -0.0120120, -0.0017747, -0.0105495, 0.0105660
8: -0.0123310, 0.0244446, -0.0118457, 0.0259467, -0.0380310, 0.0360451
9: -0.0109311, 0.0063321, -0.0098361, 0.0060626, -0.0169937, 0.0161682

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_A1_A2_B2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148648, upper bound: 0.0152359
time: 2.97 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148645, upper bound: 0.0152428
time: 2.92 seconds

## BFS NS instance: NS_A2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0032796, 0.0033597, -0.0066869, 0.0050613, -0.0083409, 0.0100466
1: 0.9923902, 1.0035273, 0.9912241, 1.0057966, -0.0134064, 0.0123032
2: -0.0195029, 0.0080956, -0.0210898, 0.0105985, -0.0293917, 0.0283725
3: -0.0002918, 0.0048708, -0.0020753, 0.0049724, -0.0052642, 0.0069461
4: -0.0091274, 0.0138311, -0.0114468, 0.0150807, -0.0242081, 0.0252779
5: -0.0022819, 0.0124679, -0.0034047, 0.0167451, -0.0190270, 0.0158726
6: -0.0058907, 0.0059371, -0.0064628, 0.0071682, -0.0130589, 0.0123999
7: -0.0114422, -0.0023745, -0.0122296, 0.0034538, -0.0148960, 0.0098551
8: -0.0109600, 0.0238760, -0.0121840, 0.0262864, -0.0370063, 0.0358014
9: -0.0078377, 0.0055708, -0.0113003, 0.0062505, -0.0140882, 0.0168711

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_A2_A1_A1_A1

### Relational analysis result of NS_A2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151552, upper bound: 0.0151893
time: 2.62 seconds

## Relational analysis of NS_A2_A2_A1_A1_A2

### Relational analysis result of NS_A2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152028, upper bound: 0.0152353
time: 2.56 seconds

## BFS NS instance: NS_A2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0036550, 0.0039185, -0.0068222, 0.0051221, -0.0087770, 0.0107407
1: 0.9923275, 1.0042725, 0.9911417, 1.0058776, -0.0135501, 0.0131308
2: -0.0201623, 0.0086251, -0.0213750, 0.0106908, -0.0300736, 0.0291944
3: -0.0005306, 0.0049144, -0.0021984, 0.0049912, -0.0055218, 0.0071128
4: -0.0097822, 0.0143522, -0.0115326, 0.0153058, -0.0250880, 0.0258848
5: -0.0027343, 0.0133230, -0.0036012, 0.0169009, -0.0196352, 0.0169242
6: -0.0061196, 0.0063554, -0.0065627, 0.0073581, -0.0134777, 0.0129181
7: -0.0117008, -0.0021023, -0.0122578, 0.0036821, -0.0153828, 0.0101555
8: -0.0113620, 0.0247424, -0.0122278, 0.0266738, -0.0377945, 0.0366906
9: -0.0087446, 0.0057940, -0.0114250, 0.0062748, -0.0150195, 0.0172190

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_A2_A1_A2_A1

### Relational analysis result of NS_A2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151607, upper bound: 0.0151868
time: 2.58 seconds

## Relational analysis of NS_A2_A2_A1_A2_A2

### Relational analysis result of NS_A2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152085, upper bound: 0.0152328
time: 2.63 seconds

## BFS NS instance: NS_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0051005, 0.0060706, -0.0037048, 0.0039927, -0.0090932, 0.0097754
1: 0.9922565, 1.0071428, 0.9921343, 1.0043714, -0.0121149, 0.0150085
2: -0.0209093, 0.0106645, -0.0221968, 0.0086954, -0.0286598, 0.0320213
3: -0.0008012, 0.0049639, -0.0012675, 0.0050491, -0.0058503, 0.0062314
4: -0.0123041, 0.0149426, -0.0098692, 0.0159602, -0.0282642, 0.0248118
5: -0.0032468, 0.0166166, -0.0041300, 0.0134366, -0.0166834, 0.0207466
6: -0.0063789, 0.0068294, -0.0068258, 0.0076461, -0.0140250, 0.0136552
7: -0.0126967, -0.0010537, -0.0117351, -0.0020661, -0.0106306, 0.0106814
8: -0.0129101, 0.0257239, -0.0114154, 0.0274155, -0.0400867, 0.0368748
9: -0.0122379, 0.0066537, -0.0088651, 0.0058237, -0.0180615, 0.0155189

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 131

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148896, upper bound: 0.0148791
time: 2.97 seconds

## Relational analysis of NS_A2_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148889, upper bound: 0.0148843
time: 2.38 seconds

## BFS NS instance: NS_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0051005, 0.0060706, -0.0042844, 0.0048556, -0.0099561, 0.0103550
1: 0.9922565, 1.0071428, 0.9921986, 1.0055224, -0.0132658, 0.0149441
2: -0.0209093, 0.0106645, -0.0215189, 0.0095132, -0.0294636, 0.0313404
3: -0.0008012, 0.0049639, -0.0010220, 0.0050042, -0.0058054, 0.0059858
4: -0.0123041, 0.0149426, -0.0108804, 0.0154244, -0.0277284, 0.0258230
5: -0.0032468, 0.0166166, -0.0036649, 0.0147573, -0.0180040, 0.0202815
6: -0.0063789, 0.0068294, -0.0065905, 0.0072160, -0.0135950, 0.0134198
7: -0.0126967, -0.0010537, -0.0121345, -0.0016457, -0.0110511, 0.0110808
8: -0.0129101, 0.0257239, -0.0120361, 0.0265248, -0.0391986, 0.0374928
9: -0.0122379, 0.0066537, -0.0102658, 0.0061684, -0.0184063, 0.0169196

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148896, upper bound: 0.0152428
time: 3.03 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148889, upper bound: 0.0152495
time: 2.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.28 seconds
NS_A1_A2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0148281, upper bound: 0.0148175
NS_A1_A2_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0148464, upper bound: 0.0148367
NS_A2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0151358, upper bound: 0.0151886
NS_A2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0151751, upper bound: 0.0152327
NS_A2_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0151413, upper bound: 0.0151860
NS_A2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0151839, upper bound: 0.0152308
NS_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0148648, upper bound: 0.0152359
NS_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0148645, upper bound: 0.0152428
NS_A2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0151552, upper bound: 0.0151893
NS_A2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0152028, upper bound: 0.0152353
NS_A2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0151607, upper bound: 0.0151868
NS_A2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0152085, upper bound: 0.0152328
NS_A2_A2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0148896, upper bound: 0.0148791
NS_A2_A2_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0148889, upper bound: 0.0148843
NS_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0148896, upper bound: 0.0152428
NS_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 7.28
Output dim: 1, lower bound: -0.0148889, upper bound: 0.0152495

## BFS NS instance: NS_A2_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0021146, 0.0012017, -0.0038070, 0.0041448, -0.0062594, 0.0050087
1: 0.9924779, 1.0006491, 0.9922918, 1.0045744, -0.0120965, 0.0083573
2: -0.0185799, 0.0060505, -0.0205380, 0.0088396, -0.0265917, 0.0256743
3: 0.0000426, 0.0048097, -0.0006667, 0.0049393, -0.0048967, 0.0054764
4: -0.0065985, 0.0131015, -0.0100474, 0.0146491, -0.0212476, 0.0231489
5: -0.0016486, 0.0091653, -0.0029920, 0.0136694, -0.0153181, 0.0121572
6: -0.0055703, 0.0053516, -0.0062500, 0.0065937, -0.0121641, 0.0116015
7: -0.0104435, -0.0034259, -0.0118055, -0.0019920, -0.0084515, 0.0083796
8: -0.0094076, 0.0226633, -0.0115248, 0.0252360, -0.0343891, 0.0339542
9: -0.0043349, 0.0047087, -0.0091121, 0.0058844, -0.0102193, 0.0138208

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A1_A1_A1_A1_B1

### Relational analysis result of NS_A2_A1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151358, upper bound: 0.0151578
time: 3.02 seconds

## Relational analysis of NS_A2_A1_A1_A1_A1_B2

### Relational analysis result of NS_A2_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151358, upper bound: 0.0151886
time: 2.91 seconds

## BFS NS instance: NS_A2_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0021463, 0.0016724, -0.0042287, 0.0044287, -0.0065750, 0.0059011
1: 0.9924859, 1.0012771, 0.9922225, 1.0049529, -0.0124670, 0.0090545
2: -0.0184948, 0.0064965, -0.0205800, 0.0092117, -0.0269259, 0.0261720
3: 0.0000734, 0.0048040, -0.0007418, 0.0049417, -0.0048683, 0.0055459
4: -0.0071501, 0.0130343, -0.0104179, 0.0146820, -0.0218321, 0.0234522
5: -0.0015902, 0.0098856, -0.0030251, 0.0142958, -0.0158860, 0.0129107
6: -0.0055408, 0.0052975, -0.0062667, 0.0066438, -0.0121846, 0.0115643
7: -0.0106613, -0.0031966, -0.0119369, -0.0013473, -0.0093140, 0.0087403
8: -0.0097462, 0.0225514, -0.0117290, 0.0253290, -0.0348227, 0.0340534
9: -0.0050988, 0.0048967, -0.0096550, 0.0059979, -0.0110967, 0.0145518

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A1_A1_A1_A2_B1

### Relational analysis result of NS_A2_A1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151751, upper bound: 0.0152027
time: 3.06 seconds

## Relational analysis of NS_A2_A1_A1_A1_A2_B2

### Relational analysis result of NS_A2_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151751, upper bound: 0.0152327
time: 2.78 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0023380, 0.0019060, -0.0038498, 0.0042085, -0.0065465, 0.0057557
1: 0.9924045, 1.0015885, 0.9922639, 1.0046593, -0.0122548, 0.0093246
2: -0.0193514, 0.0067179, -0.0208318, 0.0089000, -0.0273309, 0.0266411
3: -0.0002369, 0.0048607, -0.0007731, 0.0049588, -0.0051957, 0.0056338
4: -0.0074238, 0.0137113, -0.0101221, 0.0148813, -0.0223051, 0.0238334
5: -0.0021779, 0.0102431, -0.0031936, 0.0137669, -0.0159449, 0.0134366
6: -0.0058381, 0.0058410, -0.0063520, 0.0067802, -0.0126183, 0.0121930
7: -0.0107694, -0.0030828, -0.0118350, -0.0019609, -0.0088085, 0.0087522
8: -0.0099142, 0.0236770, -0.0115706, 0.0256221, -0.0352808, 0.0349886
9: -0.0054780, 0.0049900, -0.0092155, 0.0059099, -0.0113878, 0.0142055

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A1_A1_A2_A1_B1

### Relational analysis result of NS_A2_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151413, upper bound: 0.0151560
time: 2.76 seconds

## Relational analysis of NS_A2_A1_A1_A2_A1_B2

### Relational analysis result of NS_A2_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151413, upper bound: 0.0151860
time: 2.89 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0025809, 0.0023195, -0.0043517, 0.0044920, -0.0070730, 0.0066712
1: 0.9924141, 1.0021399, 0.9921703, 1.0050374, -0.0126234, 0.0099697
2: -0.0192511, 0.0071097, -0.0208748, 0.0093051, -0.0276811, 0.0270841
3: -0.0002005, 0.0048541, -0.0008710, 0.0049611, -0.0051616, 0.0057251
4: -0.0079084, 0.0136320, -0.0105046, 0.0149149, -0.0228232, 0.0241366
5: -0.0021091, 0.0108759, -0.0032287, 0.0144544, -0.0165635, 0.0141047
6: -0.0058033, 0.0057774, -0.0063698, 0.0068388, -0.0126421, 0.0121471
7: -0.0109608, -0.0028813, -0.0119662, -0.0011395, -0.0098213, 0.0090849
8: -0.0102117, 0.0235452, -0.0117746, 0.0257291, -0.0356848, 0.0350645
9: -0.0061492, 0.0051552, -0.0097840, 0.0060231, -0.0121723, 0.0149393

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A1_A1_A2_A2_B1

### Relational analysis result of NS_A2_A1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151838, upper bound: 0.0152014
time: 2.72 seconds

## Relational analysis of NS_A2_A1_A1_A2_A2_B2

### Relational analysis result of NS_A2_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151838, upper bound: 0.0152309
time: 2.56 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0044468, 0.0050973, -0.0035628, 0.0037813, -0.0082281, 0.0086601
1: 0.9923927, 1.0058447, 0.9923487, 1.0040896, -0.0116969, 0.0134960
2: -0.0194766, 0.0097422, -0.0199393, 0.0084951, -0.0271161, 0.0288104
3: -0.0002822, 0.0048690, -0.0004498, 0.0048997, -0.0051819, 0.0053188
4: -0.0111636, 0.0138103, -0.0096214, 0.0141759, -0.0253395, 0.0234316
5: -0.0022638, 0.0151271, -0.0025812, 0.0131131, -0.0153769, 0.0177084
6: -0.0058816, 0.0059204, -0.0060422, 0.0062139, -0.0120955, 0.0119626
7: -0.0122463, -0.0015279, -0.0116373, -0.0021691, -0.0100772, 0.0101094
8: -0.0122100, 0.0238415, -0.0112633, 0.0244493, -0.0364138, 0.0348599
9: -0.0106581, 0.0062649, -0.0085220, 0.0057392, -0.0163974, 0.0147869

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_A1_A2_B2_B1_B1

### Relational analysis result of NS_A2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151381, upper bound: 0.0151493
time: 2.95 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_B2

### Relational analysis result of NS_A2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151829, upper bound: 0.0151910
time: 3.13 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0044944, 0.0051682, -0.0039308, 0.0043292, -0.0088236, 0.0090991
1: 0.9923640, 1.0059392, 0.9922817, 1.0048205, -0.0124565, 0.0136576
2: -0.0197777, 0.0098094, -0.0206447, 0.0090143, -0.0279415, 0.0295003
3: -0.0003913, 0.0048890, -0.0007053, 0.0049464, -0.0053377, 0.0055943
4: -0.0112467, 0.0140482, -0.0102635, 0.0147335, -0.0259802, 0.0243117
5: -0.0024704, 0.0152357, -0.0030652, 0.0139516, -0.0164220, 0.0183009
6: -0.0059861, 0.0061114, -0.0062871, 0.0066615, -0.0126476, 0.0123985
7: -0.0122791, -0.0014934, -0.0118909, -0.0019021, -0.0103770, 0.0103975
8: -0.0122610, 0.0242370, -0.0116575, 0.0253763, -0.0373690, 0.0356502
9: -0.0107732, 0.0062933, -0.0094114, 0.0059581, -0.0167313, 0.0157047

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151379, upper bound: 0.0151550
time: 2.52 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151828, upper bound: 0.0151975
time: 2.80 seconds

## BFS NS instance: NS_A2_A2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0023874, 0.0019771, -0.0039822, 0.0044057, -0.0067930, 0.0059593
1: 0.9923882, 1.0016835, 0.9922507, 1.0049223, -0.0125340, 0.0094327
2: -0.0195221, 0.0067853, -0.0209705, 0.0090868, -0.0276602, 0.0268718
3: -0.0002987, 0.0048721, -0.0008233, 0.0049679, -0.0052666, 0.0056954
4: -0.0075072, 0.0138462, -0.0103531, 0.0149910, -0.0224982, 0.0241993
5: -0.0022950, 0.0103520, -0.0032887, 0.0140686, -0.0163637, 0.0136407
6: -0.0058974, 0.0059493, -0.0064001, 0.0068681, -0.0127655, 0.0123494
7: -0.0108024, -0.0030481, -0.0119263, -0.0018649, -0.0089375, 0.0088781
8: -0.0099654, 0.0239012, -0.0117124, 0.0258043, -0.0355229, 0.0353492
9: -0.0055935, 0.0050185, -0.0095354, 0.0059886, -0.0115821, 0.0145539

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A2_A1_A1_A1_B1

### Relational analysis result of NS_A2_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151510, upper bound: 0.0151578
time: 2.81 seconds

## Relational analysis of NS_A2_A2_A1_A1_A1_B2

### Relational analysis result of NS_A2_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151510, upper bound: 0.0151647
time: 2.45 seconds

## BFS NS instance: NS_A2_A2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0026166, 0.0023726, -0.0045559, 0.0046897, -0.0073064, 0.0069286
1: 0.9923977, 1.0022109, 0.9921351, 1.0053011, -0.0129035, 0.0100757
2: -0.0194237, 0.0071601, -0.0210170, 0.0095220, -0.0280616, 0.0273064
3: -0.0002631, 0.0048655, -0.0009420, 0.0049704, -0.0052335, 0.0058075
4: -0.0079706, 0.0137684, -0.0107471, 0.0150271, -0.0229977, 0.0245155
5: -0.0022275, 0.0109573, -0.0033272, 0.0148115, -0.0170390, 0.0142845
6: -0.0058632, 0.0058868, -0.0064199, 0.0069355, -0.0127987, 0.0123067
7: -0.0109854, -0.0028554, -0.0120577, -0.0008854, -0.0101000, 0.0092023
8: -0.0102499, 0.0237719, -0.0119168, 0.0259257, -0.0359309, 0.0354313
9: -0.0062355, 0.0051765, -0.0101278, 0.0061021, -0.0123376, 0.0153042

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A2_A1_A1_A2_B1

### Relational analysis result of NS_A2_A2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151935, upper bound: 0.0152027
time: 7.50 seconds

## Relational analysis of NS_A2_A2_A1_A1_A2_B2

### Relational analysis result of NS_A2_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151935, upper bound: 0.0152133
time: 2.81 seconds

## BFS NS instance: NS_A2_A2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0027488, 0.0025694, -0.0040234, 0.0044671, -0.0072159, 0.0065929
1: 0.9923220, 1.0024735, 0.9922236, 1.0050040, -0.0126820, 0.0102499
2: -0.0202207, 0.0073466, -0.0212553, 0.0091449, -0.0283387, 0.0277176
3: -0.0005518, 0.0049183, -0.0009264, 0.0049868, -0.0055386, 0.0058447
4: -0.0082013, 0.0143983, -0.0104250, 0.0152160, -0.0234173, 0.0248234
5: -0.0027743, 0.0112584, -0.0034841, 0.0141626, -0.0169369, 0.0147425
6: -0.0061399, 0.0063924, -0.0064990, 0.0070488, -0.0131887, 0.0128914
7: -0.0110765, -0.0027595, -0.0119547, -0.0018350, -0.0092415, 0.0091951
8: -0.0103915, 0.0248191, -0.0117566, 0.0261784, -0.0363228, 0.0362896
9: -0.0065549, 0.0052551, -0.0096351, 0.0060132, -0.0125681, 0.0148902

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A2_A1_A2_A1_B1

### Relational analysis result of NS_A2_A2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151564, upper bound: 0.0151560
time: 2.54 seconds

## Relational analysis of NS_A2_A2_A1_A2_A1_B2

### Relational analysis result of NS_A2_A2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151564, upper bound: 0.0151621
time: 2.72 seconds

## BFS NS instance: NS_A2_A2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0029870, 0.0029240, -0.0046779, 0.0047510, -0.0077379, 0.0076019
1: 0.9923347, 1.0029461, 0.9920833, 1.0053827, -0.0130479, 0.0108629
2: -0.0200861, 0.0076826, -0.0213029, 0.0096128, -0.0287452, 0.0281238
3: -0.0005030, 0.0049094, -0.0010680, 0.0049892, -0.0054922, 0.0059774
4: -0.0086168, 0.0142919, -0.0108313, 0.0152529, -0.0238696, 0.0251233
5: -0.0026819, 0.0118011, -0.0035246, 0.0149655, -0.0176475, 0.0153257
6: -0.0060931, 0.0063071, -0.0065199, 0.0071245, -0.0132176, 0.0128270
7: -0.0112406, -0.0025868, -0.0120860, -0.0006782, -0.0105624, 0.0094992
8: -0.0106466, 0.0246422, -0.0119608, 0.0263139, -0.0367167, 0.0363234
9: -0.0071304, 0.0053967, -0.0102527, 0.0061266, -0.0132570, 0.0156494

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A2_A1_A2_A2_B1

### Relational analysis result of NS_A2_A2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151999, upper bound: 0.0152014
time: 2.72 seconds

## Relational analysis of NS_A2_A2_A1_A2_A2_B2

### Relational analysis result of NS_A2_A2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151999, upper bound: 0.0152108
time: 2.84 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0049914, 0.0059082, -0.0037497, 0.0040595, -0.0090509, 0.0096579
1: 0.9922986, 1.0069262, 0.9923056, 1.0044606, -0.0121620, 0.0146206
2: -0.0204675, 0.0105107, -0.0203932, 0.0087587, -0.0282518, 0.0300564
3: -0.0006412, 0.0049346, -0.0006142, 0.0049297, -0.0055709, 0.0055489
4: -0.0121138, 0.0145935, -0.0099474, 0.0145347, -0.0266485, 0.0245408
5: -0.0029437, 0.0163681, -0.0028927, 0.0135388, -0.0164824, 0.0192608
6: -0.0062255, 0.0065490, -0.0061998, 0.0065019, -0.0127275, 0.0127488
7: -0.0126216, -0.0011328, -0.0117660, -0.0020336, -0.0105880, 0.0106332
8: -0.0127933, 0.0251435, -0.0114634, 0.0250458, -0.0376006, 0.0363346
9: -0.0119743, 0.0065889, -0.0089735, 0.0058504, -0.0178247, 0.0155624

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_A2_A2_B2_B1_B1

### Relational analysis result of NS_A2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151580, upper bound: 0.0151534
time: 2.79 seconds

## Relational analysis of NS_A2_A2_A2_B2_B1_B2

### Relational analysis result of NS_A2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152092, upper bound: 0.0151999
time: 2.56 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0050344, 0.0059722, -0.0041102, 0.0045963, -0.0096306, 0.0100824
1: 0.9922726, 1.0070114, 0.9922405, 1.0051764, -0.0129039, 0.0147709
2: -0.0207406, 0.0105713, -0.0210774, 0.0092674, -0.0290541, 0.0307281
3: -0.0007401, 0.0049527, -0.0008621, 0.0049750, -0.0057151, 0.0058148
4: -0.0121887, 0.0148093, -0.0105764, 0.0150754, -0.0272641, 0.0253857
5: -0.0031310, 0.0164660, -0.0033620, 0.0143603, -0.0174913, 0.0198280
6: -0.0063203, 0.0067223, -0.0064372, 0.0069359, -0.0132562, 0.0131595
7: -0.0126512, -0.0011017, -0.0120144, -0.0017720, -0.0108792, 0.0109128
8: -0.0128393, 0.0255023, -0.0118496, 0.0259446, -0.0385242, 0.0370872
9: -0.0120781, 0.0066144, -0.0098448, 0.0060648, -0.0181429, 0.0164592

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_A2_A2_B2_B2_B1

### Relational analysis result of NS_A2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151577, upper bound: 0.0151593
time: 2.27 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_B2

### Relational analysis result of NS_A2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152085, upper bound: 0.0152060
time: 2.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.49 seconds
NS_A2_A1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151358, upper bound: 0.0151578
NS_A2_A1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151358, upper bound: 0.0151886
NS_A2_A1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151751, upper bound: 0.0152027
NS_A2_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151751, upper bound: 0.0152327
NS_A2_A1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151413, upper bound: 0.0151560
NS_A2_A1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151413, upper bound: 0.0151860
NS_A2_A1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151838, upper bound: 0.0152014
NS_A2_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151838, upper bound: 0.0152309
NS_A2_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151381, upper bound: 0.0151493
NS_A2_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151829, upper bound: 0.0151910
NS_A2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151379, upper bound: 0.0151550
NS_A2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151828, upper bound: 0.0151975
NS_A2_A2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151510, upper bound: 0.0151578
NS_A2_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151510, upper bound: 0.0151647
NS_A2_A2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151935, upper bound: 0.0152027
NS_A2_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151935, upper bound: 0.0152133
NS_A2_A2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151564, upper bound: 0.0151560
NS_A2_A2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151564, upper bound: 0.0151621
NS_A2_A2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151999, upper bound: 0.0152014
NS_A2_A2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151999, upper bound: 0.0152108
NS_A2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151580, upper bound: 0.0151534
NS_A2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0152092, upper bound: 0.0151999
NS_A2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0151577, upper bound: 0.0151593
NS_A2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 1, lower bound: -0.0152085, upper bound: 0.0152060

## BFS NS instance: NS_A2_A1_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0021146, 0.0012017, -0.0034047, 0.0035459, -0.0056605, 0.0046064
1: 0.9924779, 1.0006491, 0.9923716, 1.0037755, -0.0112976, 0.0082775
2: -0.0185799, 0.0060505, -0.0196990, 0.0082720, -0.0260013, 0.0248489
3: 0.0000426, 0.0048097, -0.0003628, 0.0048838, -0.0048412, 0.0051725
4: -0.0065985, 0.0131015, -0.0093455, 0.0139860, -0.0205845, 0.0224471
5: -0.0016486, 0.0091653, -0.0024164, 0.0127529, -0.0144015, 0.0115817
6: -0.0055703, 0.0053516, -0.0059588, 0.0060615, -0.0116318, 0.0113103
7: -0.0104435, -0.0034259, -0.0115284, -0.0022838, -0.0081597, 0.0081024
8: -0.0094076, 0.0226633, -0.0110939, 0.0241336, -0.0332868, 0.0335175
9: -0.0043349, 0.0047087, -0.0081399, 0.0056452, -0.0099800, 0.0128486

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_A1_A1_A1_A1_B1_B1

### Relational analysis result of NS_A2_A1_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149659, upper bound: 0.0149567
time: 2.69 seconds

## Relational analysis of NS_A2_A1_A1_A1_A1_B1_B2

### Relational analysis result of NS_A2_A1_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149207, upper bound: 0.0149460
time: 2.49 seconds

## BFS NS instance: NS_A2_A1_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0021146, 0.0012017, -0.0038278, 0.0041758, -0.0062904, 0.0050295
1: 0.9924779, 1.0006491, 0.9922826, 1.0046158, -0.0121379, 0.0083665
2: -0.0185799, 0.0060505, -0.0206345, 0.0088689, -0.0266207, 0.0257798
3: 0.0000426, 0.0048097, -0.0007016, 0.0049457, -0.0049031, 0.0055113
4: -0.0065985, 0.0131015, -0.0100837, 0.0147253, -0.0213239, 0.0231852
5: -0.0016486, 0.0091653, -0.0030582, 0.0137169, -0.0153655, 0.0122234
6: -0.0055703, 0.0053516, -0.0062835, 0.0066550, -0.0122253, 0.0116351
7: -0.0104435, -0.0034259, -0.0118199, -0.0019769, -0.0084667, 0.0083939
8: -0.0094076, 0.0226633, -0.0115471, 0.0253627, -0.0345175, 0.0339763
9: -0.0043349, 0.0047087, -0.0091624, 0.0058968, -0.0102317, 0.0138711

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_A1_A1_A1_A1_B2_A1

### Relational analysis result of NS_A2_A1_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149349, upper bound: 0.0150294
time: 2.66 seconds

## Relational analysis of NS_A2_A1_A1_A1_A1_B2_A2

### Relational analysis result of NS_A2_A1_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149207, upper bound: 0.0149752
time: 2.45 seconds

## BFS NS instance: NS_A2_A1_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0021463, 0.0016724, -0.0044652, 0.0038750, -0.0060214, 0.0061376
1: 0.9924859, 1.0012771, 0.9921008, 1.0042145, -0.0117286, 0.0091763
2: -0.0184948, 0.0064965, -0.0197535, 0.0089253, -0.0266813, 0.0253691
3: 0.0000734, 0.0048040, -0.0006660, 0.0048862, -0.0048128, 0.0054700
4: -0.0071501, 0.0130343, -0.0098556, 0.0140287, -0.0211788, 0.0228899
5: -0.0015902, 0.0098856, -0.0024698, 0.0139057, -0.0154959, 0.0123554
6: -0.0055408, 0.0052975, -0.0059841, 0.0061795, -0.0117203, 0.0112816
7: -0.0106613, -0.0031966, -0.0116807, -0.0002905, -0.0103708, 0.0084841
8: -0.0097462, 0.0225514, -0.0113307, 0.0243359, -0.0338292, 0.0336490
9: -0.0050988, 0.0048967, -0.0089541, 0.0057767, -0.0108755, 0.0138508

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_A1_A1_A1_A2_B1_B1

### Relational analysis result of NS_A2_A1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149882, upper bound: 0.0149781
time: 2.40 seconds

## Relational analysis of NS_A2_A1_A1_A1_A2_B1_B2

### Relational analysis result of NS_A2_A1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149406, upper bound: 0.0149665
time: 2.35 seconds

## BFS NS instance: NS_A2_A1_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0021463, 0.0016724, -0.0042656, 0.0044608, -0.0066071, 0.0059380
1: 0.9924859, 1.0012771, 0.9922075, 1.0049958, -0.0125099, 0.0090696
2: -0.0184948, 0.0064965, -0.0206775, 0.0092486, -0.0269636, 0.0262842
3: 0.0000734, 0.0048040, -0.0007819, 0.0049481, -0.0048748, 0.0055859
4: -0.0071501, 0.0130343, -0.0104579, 0.0147590, -0.0219091, 0.0234922
5: -0.0015902, 0.0098856, -0.0030923, 0.0143567, -0.0159469, 0.0129779
6: -0.0055408, 0.0052975, -0.0063007, 0.0067076, -0.0122484, 0.0115983
7: -0.0106613, -0.0031966, -0.0119518, -0.0012973, -0.0093640, 0.0087551
8: -0.0097462, 0.0225514, -0.0117521, 0.0254601, -0.0349574, 0.0340763
9: -0.0050988, 0.0048967, -0.0097121, 0.0060107, -0.0111095, 0.0146088

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_A1_A1_A1_A2_B2_A1

### Relational analysis result of NS_A2_A1_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149589, upper bound: 0.0150535
time: 2.61 seconds

## Relational analysis of NS_A2_A1_A1_A1_A2_B2_A2

### Relational analysis result of NS_A2_A1_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149406, upper bound: 0.0149957
time: 2.61 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0023380, 0.0019060, -0.0034515, 0.0036156, -0.0059535, 0.0053574
1: 0.9924045, 1.0015885, 0.9923416, 1.0038686, -0.0114641, 0.0092468
2: -0.0193514, 0.0067179, -0.0200130, 0.0083380, -0.0267438, 0.0258315
3: -0.0002369, 0.0048607, -0.0004766, 0.0049046, -0.0051415, 0.0053373
4: -0.0074238, 0.0137113, -0.0094272, 0.0142342, -0.0216580, 0.0231385
5: -0.0021779, 0.0102431, -0.0026319, 0.0128594, -0.0150373, 0.0128749
6: -0.0058381, 0.0058410, -0.0060678, 0.0062607, -0.0120989, 0.0119088
7: -0.0107694, -0.0030828, -0.0115606, -0.0022498, -0.0085196, 0.0084778
8: -0.0099142, 0.0236770, -0.0111440, 0.0245463, -0.0342066, 0.0345555
9: -0.0054780, 0.0049900, -0.0082530, 0.0056730, -0.0111510, 0.0132430

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_A1_A1_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149747, upper bound: 0.0149566
time: 3.01 seconds

## Relational analysis of NS_A2_A1_A1_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149290, upper bound: 0.0149457
time: 2.61 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0023380, 0.0019060, -0.0038702, 0.0042389, -0.0065769, 0.0057761
1: 0.9924045, 1.0015885, 0.9922565, 1.0046999, -0.0122954, 0.0093319
2: -0.0193514, 0.0067179, -0.0209087, 0.0089287, -0.0273591, 0.0267336
3: -0.0002369, 0.0048607, -0.0008009, 0.0049638, -0.0052007, 0.0056617
4: -0.0074238, 0.0137113, -0.0101576, 0.0149421, -0.0223659, 0.0238689
5: -0.0021779, 0.0102431, -0.0032463, 0.0138134, -0.0159913, 0.0134894
6: -0.0058381, 0.0058410, -0.0063787, 0.0068289, -0.0126670, 0.0122196
7: -0.0107694, -0.0030828, -0.0118491, -0.0019462, -0.0088233, 0.0087663
8: -0.0099142, 0.0236770, -0.0115925, 0.0257230, -0.0353871, 0.0350103
9: -0.0054780, 0.0049900, -0.0092648, 0.0059220, -0.0114000, 0.0142548

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_A1_A1_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149403, upper bound: 0.0149914
time: 2.84 seconds

## Relational analysis of NS_A2_A1_A1_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149290, upper bound: 0.0149746
time: 2.52 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0025809, 0.0023195, -0.0046032, 0.0039442, -0.0065251, 0.0069227
1: 0.9924141, 1.0021399, 0.9920408, 1.0043070, -0.0118930, 0.0100991
2: -0.0192511, 0.0071097, -0.0200692, 0.0090251, -0.0274406, 0.0262996
3: -0.0002005, 0.0048541, -0.0008080, 0.0049069, -0.0051075, 0.0056621
4: -0.0079084, 0.0136320, -0.0099500, 0.0142781, -0.0221864, 0.0235820
5: -0.0021091, 0.0108759, -0.0026877, 0.0140748, -0.0161839, 0.0135637
6: -0.0058033, 0.0057774, -0.0060944, 0.0063886, -0.0121919, 0.0118718
7: -0.0109608, -0.0028813, -0.0117127, -0.0000608, -0.0109000, 0.0088314
8: -0.0102117, 0.0235452, -0.0113805, 0.0247644, -0.0347212, 0.0346633
9: -0.0061492, 0.0051552, -0.0090934, 0.0058043, -0.0119535, 0.0142486

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_A1_A1_A2_A2_B1_B1

### Relational analysis result of NS_A2_A1_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149958, upper bound: 0.0149762
time: 2.39 seconds

## Relational analysis of NS_A2_A1_A1_A2_A2_B1_B2

### Relational analysis result of NS_A2_A1_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149472, upper bound: 0.0149648
time: 2.61 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0025809, 0.0023195, -0.0043879, 0.0045233, -0.0071043, 0.0067074
1: 0.9924141, 1.0021399, 0.9921575, 1.0050792, -0.0126651, 0.0099825
2: -0.0192511, 0.0071097, -0.0209533, 0.0093409, -0.0277179, 0.0271849
3: -0.0002005, 0.0048541, -0.0009036, 0.0049663, -0.0051668, 0.0057577
4: -0.0079084, 0.0136320, -0.0105436, 0.0149769, -0.0228852, 0.0241756
5: -0.0021091, 0.0108759, -0.0032827, 0.0145136, -0.0166227, 0.0141587
6: -0.0058033, 0.0057774, -0.0063971, 0.0068901, -0.0126934, 0.0121745
7: -0.0109608, -0.0028813, -0.0119807, -0.0010914, -0.0098694, 0.0090994
8: -0.0102117, 0.0235452, -0.0117971, 0.0258342, -0.0357972, 0.0350870
9: -0.0061492, 0.0051552, -0.0098396, 0.0060356, -0.0121848, 0.0149948

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_A1_A1_A2_A2_B2_A1

### Relational analysis result of NS_A2_A1_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149635, upper bound: 0.0150493
time: 2.46 seconds

## Relational analysis of NS_A2_A1_A1_A2_A2_B2_A2

### Relational analysis result of NS_A2_A1_A1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149472, upper bound: 0.0149931
time: 2.50 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0040121, 0.0044501, -0.0026055, 0.0023560, -0.0063680, 0.0070556
1: 0.9923987, 1.0049816, 0.9923474, 1.0021887, -0.0097900, 0.0126342
2: -0.0194131, 0.0091289, -0.0199529, 0.0071444, -0.0256739, 0.0281963
3: -0.0002592, 0.0048648, -0.0004547, 0.0049006, -0.0051598, 0.0053196
4: -0.0104052, 0.0137601, -0.0079512, 0.0141866, -0.0245918, 0.0217112
5: -0.0022203, 0.0141366, -0.0025906, 0.0109318, -0.0131521, 0.0167272
6: -0.0058596, 0.0058802, -0.0060469, 0.0062225, -0.0120821, 0.0119270
7: -0.0119468, -0.0018432, -0.0109777, -0.0028635, -0.0090833, 0.0091345
8: -0.0117444, 0.0237581, -0.0102379, 0.0244672, -0.0359635, 0.0337416
9: -0.0096076, 0.0060064, -0.0062084, 0.0051698, -0.0147774, 0.0122148

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A1_A2_B2_B1_B1_B1

### Relational analysis result of NS_A2_A1_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151381, upper bound: 0.0151318
time: 2.72 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_B1_B2

### Relational analysis result of NS_A2_A1_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151381, upper bound: 0.0151493
time: 2.60 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0042063, 0.0047394, -0.0028761, 0.0027589, -0.0069652, 0.0076154
1: 0.9923952, 1.0053673, 0.9923562, 1.0027261, -0.0103309, 0.0130111
2: -0.0194490, 0.0094030, -0.0198603, 0.0075262, -0.0260896, 0.0283920
3: -0.0002723, 0.0048672, -0.0004212, 0.0048944, -0.0051667, 0.0052884
4: -0.0107441, 0.0137885, -0.0084233, 0.0141135, -0.0248576, 0.0222118
5: -0.0022449, 0.0145793, -0.0025271, 0.0115484, -0.0137933, 0.0171064
6: -0.0058720, 0.0059029, -0.0060148, 0.0061638, -0.0120358, 0.0119177
7: -0.0120807, -0.0017023, -0.0111641, -0.0026673, -0.0094134, 0.0094618
8: -0.0119525, 0.0238052, -0.0105278, 0.0243456, -0.0360536, 0.0340802
9: -0.0100771, 0.0061220, -0.0068624, 0.0053308, -0.0154079, 0.0129844

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A1_A2_B2_B1_B2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151829, upper bound: 0.0151714
time: 2.83 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_B2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151829, upper bound: 0.0151910
time: 2.72 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0040601, 0.0045217, -0.0030047, 0.0029504, -0.0070106, 0.0075264
1: 0.9923701, 1.0050771, 0.9922760, 1.0029814, -0.0106114, 0.0128012
2: -0.0197150, 0.0091967, -0.0207049, 0.0077077, -0.0265401, 0.0289162
3: -0.0003686, 0.0048848, -0.0007271, 0.0049504, -0.0053190, 0.0056119
4: -0.0104890, 0.0139987, -0.0086477, 0.0147810, -0.0252700, 0.0226464
5: -0.0024274, 0.0142462, -0.0031065, 0.0118415, -0.0142689, 0.0173526
6: -0.0059643, 0.0060716, -0.0063079, 0.0066996, -0.0126639, 0.0123796
7: -0.0119799, -0.0018084, -0.0112528, -0.0025739, -0.0094060, 0.0094444
8: -0.0117959, 0.0241547, -0.0106656, 0.0254553, -0.0369771, 0.0345676
9: -0.0097237, 0.0060350, -0.0071733, 0.0054073, -0.0151310, 0.0132083

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A1_A2_B2_B2_B1_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151379, upper bound: 0.0151384
time: 2.75 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_B1_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151379, upper bound: 0.0151550
time: 2.73 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0042539, 0.0048101, -0.0032482, 0.0033129, -0.0075668, 0.0080583
1: 0.9923666, 1.0054616, 0.9922888, 1.0034648, -0.0110982, 0.0131728
2: -0.0197504, 0.0094700, -0.0205696, 0.0080512, -0.0269233, 0.0290848
3: -0.0003814, 0.0048872, -0.0006781, 0.0049414, -0.0053228, 0.0055653
4: -0.0108270, 0.0140267, -0.0090725, 0.0146741, -0.0255011, 0.0230992
5: -0.0024516, 0.0146876, -0.0030137, 0.0123963, -0.0148479, 0.0177013
6: -0.0059766, 0.0060941, -0.0062610, 0.0066138, -0.0125905, 0.0123551
7: -0.0121134, -0.0016678, -0.0114205, -0.0023973, -0.0097161, 0.0097527
8: -0.0120034, 0.0242012, -0.0109263, 0.0252776, -0.0370124, 0.0348773
9: -0.0101920, 0.0061502, -0.0077617, 0.0055521, -0.0157441, 0.0139119

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A1_A2_B2_B2_B2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151828, upper bound: 0.0151799
time: 2.79 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_B2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151828, upper bound: 0.0151975
time: 2.81 seconds

## BFS NS instance: NS_A2_A2_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0023874, 0.0019771, -0.0034047, 0.0035459, -0.0059333, 0.0053818
1: 0.9923882, 1.0016835, 0.9923716, 1.0037755, -0.0113872, 0.0093119
2: -0.0195221, 0.0067853, -0.0196990, 0.0082720, -0.0269403, 0.0256117
3: -0.0002987, 0.0048721, -0.0003628, 0.0048838, -0.0051825, 0.0052348
4: -0.0075072, 0.0138462, -0.0093455, 0.0139860, -0.0214932, 0.0231917
5: -0.0022950, 0.0103520, -0.0024164, 0.0127529, -0.0150479, 0.0127683
6: -0.0058974, 0.0059493, -0.0059588, 0.0060615, -0.0119588, 0.0119080
7: -0.0108024, -0.0030481, -0.0115284, -0.0022838, -0.0085186, 0.0084802
8: -0.0099654, 0.0239012, -0.0110939, 0.0241336, -0.0338511, 0.0347549
9: -0.0055935, 0.0050185, -0.0081399, 0.0056452, -0.0112387, 0.0131584

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_A2_A1_A1_A1_B1_B1

### Relational analysis result of NS_A2_A2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149876, upper bound: 0.0149567
time: 2.83 seconds

## Relational analysis of NS_A2_A2_A1_A1_A1_B1_B2

### Relational analysis result of NS_A2_A2_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149406, upper bound: 0.0149460
time: 2.54 seconds

## BFS NS instance: NS_A2_A2_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0023874, 0.0019771, -0.0038278, 0.0041758, -0.0065632, 0.0058049
1: 0.9923882, 1.0016835, 0.9922826, 1.0046158, -0.0122275, 0.0094008
2: -0.0195221, 0.0067853, -0.0206345, 0.0088689, -0.0274427, 0.0264119
3: -0.0002987, 0.0048721, -0.0007016, 0.0049457, -0.0052444, 0.0055737
4: -0.0075072, 0.0138462, -0.0100837, 0.0147253, -0.0222325, 0.0239299
5: -0.0022950, 0.0103520, -0.0030582, 0.0137169, -0.0160119, 0.0134101
6: -0.0058974, 0.0059493, -0.0062835, 0.0066550, -0.0125523, 0.0122328
7: -0.0108024, -0.0030481, -0.0118199, -0.0019769, -0.0088255, 0.0087717
8: -0.0099654, 0.0239012, -0.0115471, 0.0253627, -0.0350475, 0.0351842
9: -0.0055935, 0.0050185, -0.0091624, 0.0058968, -0.0114903, 0.0141808

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 212

## Relational analysis of NS_A2_A2_A1_A1_A1_B2_B1

### Relational analysis result of NS_A2_A2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150553, upper bound: 0.0150529
time: 2.86 seconds

## Relational analysis of NS_A2_A2_A1_A1_A1_B2_B2

### Relational analysis result of NS_A2_A2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151030, upper bound: 0.0151180
time: 2.87 seconds

## BFS NS instance: NS_A2_A2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0026166, 0.0023726, -0.0044652, 0.0038750, -0.0064917, 0.0068378
1: 0.9923977, 1.0022109, 0.9921008, 1.0042145, -0.0118169, 0.0101100
2: -0.0194237, 0.0071601, -0.0197535, 0.0089253, -0.0276092, 0.0260575
3: -0.0002631, 0.0048655, -0.0006660, 0.0048862, -0.0051493, 0.0055315
4: -0.0079706, 0.0137684, -0.0098556, 0.0140287, -0.0219993, 0.0236240
5: -0.0022275, 0.0109573, -0.0024698, 0.0139057, -0.0161332, 0.0134270
6: -0.0058632, 0.0058868, -0.0059841, 0.0061795, -0.0120427, 0.0118709
7: -0.0109854, -0.0028554, -0.0116807, -0.0002905, -0.0106949, 0.0088252
8: -0.0102499, 0.0237719, -0.0113307, 0.0243359, -0.0343377, 0.0348695
9: -0.0062355, 0.0051765, -0.0089541, 0.0057767, -0.0120121, 0.0141305

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_A2_A1_A1_A2_B1_B1

### Relational analysis result of NS_A2_A2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150095, upper bound: 0.0149781
time: 2.49 seconds

## Relational analysis of NS_A2_A2_A1_A1_A2_B1_B2

### Relational analysis result of NS_A2_A2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149597, upper bound: 0.0149665
time: 2.58 seconds

## BFS NS instance: NS_A2_A2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0026166, 0.0023726, -0.0042656, 0.0044608, -0.0070774, 0.0066383
1: 0.9923977, 1.0022109, 0.9922075, 1.0049958, -0.0125982, 0.0100033
2: -0.0194237, 0.0071601, -0.0206775, 0.0092486, -0.0277753, 0.0268450
3: -0.0002631, 0.0048655, -0.0007819, 0.0049481, -0.0052112, 0.0056474
4: -0.0079706, 0.0137684, -0.0104579, 0.0147590, -0.0227296, 0.0242263
5: -0.0022275, 0.0109573, -0.0030923, 0.0143567, -0.0165842, 0.0140496
6: -0.0058632, 0.0058868, -0.0063007, 0.0067076, -0.0125708, 0.0121876
7: -0.0109854, -0.0028554, -0.0119518, -0.0012973, -0.0096881, 0.0090963
8: -0.0102499, 0.0237719, -0.0117521, 0.0254601, -0.0354335, 0.0352669
9: -0.0062355, 0.0051765, -0.0097121, 0.0060107, -0.0122461, 0.0148885

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_A2_A1_A1_A2_B2_B1

### Relational analysis result of NS_A2_A2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150095, upper bound: 0.0149934
time: 2.55 seconds

## Relational analysis of NS_A2_A2_A1_A1_A2_B2_B2

### Relational analysis result of NS_A2_A2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149597, upper bound: 0.0149818
time: 2.06 seconds

## BFS NS instance: NS_A2_A2_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0027488, 0.0025694, -0.0034515, 0.0036156, -0.0063644, 0.0060209
1: 0.9923220, 1.0024735, 0.9923416, 1.0038686, -0.0115466, 0.0101318
2: -0.0202207, 0.0073466, -0.0200130, 0.0083380, -0.0276229, 0.0264809
3: -0.0005518, 0.0049183, -0.0004766, 0.0049046, -0.0054563, 0.0053949
4: -0.0082013, 0.0143983, -0.0094272, 0.0142342, -0.0224355, 0.0238255
5: -0.0027743, 0.0112584, -0.0026319, 0.0128594, -0.0156337, 0.0138903
6: -0.0061399, 0.0063924, -0.0060678, 0.0062607, -0.0124006, 0.0124602
7: -0.0110765, -0.0027595, -0.0115606, -0.0022498, -0.0088266, 0.0088010
8: -0.0103915, 0.0248191, -0.0111440, 0.0245463, -0.0346898, 0.0356995
9: -0.0065549, 0.0052551, -0.0082530, 0.0056730, -0.0122279, 0.0135080

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_A2_A1_A2_A1_B1_B1

### Relational analysis result of NS_A2_A2_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149976, upper bound: 0.0149566
time: 2.54 seconds

## Relational analysis of NS_A2_A2_A1_A2_A1_B1_B2

### Relational analysis result of NS_A2_A2_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149483, upper bound: 0.0149457
time: 2.43 seconds

## BFS NS instance: NS_A2_A2_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0027488, 0.0025694, -0.0038702, 0.0042389, -0.0069877, 0.0064396
1: 0.9923220, 1.0024735, 0.9922565, 1.0046999, -0.0123779, 0.0102170
2: -0.0202207, 0.0073466, -0.0209087, 0.0089287, -0.0281219, 0.0272635
3: -0.0005518, 0.0049183, -0.0008009, 0.0049638, -0.0055156, 0.0057192
4: -0.0082013, 0.0143983, -0.0101576, 0.0149421, -0.0231434, 0.0245560
5: -0.0027743, 0.0112584, -0.0032463, 0.0138134, -0.0165877, 0.0145047
6: -0.0061399, 0.0063924, -0.0063787, 0.0068289, -0.0129688, 0.0127711
7: -0.0110765, -0.0027595, -0.0118491, -0.0019462, -0.0091303, 0.0090895
8: -0.0103915, 0.0248191, -0.0115925, 0.0257230, -0.0358403, 0.0361257
9: -0.0065549, 0.0052551, -0.0092648, 0.0059220, -0.0124769, 0.0145199

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 212

## Relational analysis of NS_A2_A2_A1_A2_A1_B2_B1

### Relational analysis result of NS_A2_A2_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150608, upper bound: 0.0150478
time: 3.32 seconds

## Relational analysis of NS_A2_A2_A1_A2_A1_B2_B2

### Relational analysis result of NS_A2_A2_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151093, upper bound: 0.0151164
time: 2.34 seconds

## BFS NS instance: NS_A2_A2_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0029870, 0.0029240, -0.0046032, 0.0039442, -0.0069312, 0.0075272
1: 0.9923347, 1.0029461, 0.9920408, 1.0043070, -0.0119723, 0.0109053
2: -0.0200861, 0.0076826, -0.0200692, 0.0090251, -0.0282973, 0.0268985
3: -0.0005030, 0.0049094, -0.0008080, 0.0049069, -0.0054099, 0.0057174
4: -0.0086168, 0.0142919, -0.0099500, 0.0142781, -0.0228948, 0.0242420
5: -0.0026819, 0.0118011, -0.0026877, 0.0140748, -0.0167568, 0.0144888
6: -0.0060931, 0.0063071, -0.0060944, 0.0063886, -0.0124817, 0.0124015
7: -0.0112406, -0.0025868, -0.0117127, -0.0000608, -0.0111798, 0.0091259
8: -0.0106466, 0.0246422, -0.0113805, 0.0247644, -0.0351649, 0.0357650
9: -0.0071304, 0.0053967, -0.0090934, 0.0058043, -0.0129347, 0.0144901

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_A2_A1_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150177, upper bound: 0.0149762
time: 2.80 seconds

## Relational analysis of NS_A2_A2_A1_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149647, upper bound: 0.0149648
time: 2.50 seconds

## BFS NS instance: NS_A2_A2_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0029870, 0.0029240, -0.0043879, 0.0045233, -0.0075103, 0.0073119
1: 0.9923347, 1.0029461, 0.9921575, 1.0050792, -0.0127444, 0.0107887
2: -0.0200861, 0.0076826, -0.0209533, 0.0093409, -0.0284607, 0.0276651
3: -0.0005030, 0.0049094, -0.0009036, 0.0049663, -0.0054693, 0.0058130
4: -0.0086168, 0.0142919, -0.0105436, 0.0149769, -0.0235936, 0.0248356
5: -0.0026819, 0.0118011, -0.0032827, 0.0145136, -0.0171955, 0.0150838
6: -0.0060931, 0.0063071, -0.0063971, 0.0068901, -0.0129832, 0.0127042
7: -0.0112406, -0.0025868, -0.0119807, -0.0010914, -0.0101492, 0.0093939
8: -0.0106466, 0.0246422, -0.0117971, 0.0258342, -0.0362104, 0.0361598
9: -0.0071304, 0.0053967, -0.0098396, 0.0060356, -0.0131661, 0.0152363

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_A2_A1_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150177, upper bound: 0.0149890
time: 2.64 seconds

## Relational analysis of NS_A2_A2_A1_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149648, upper bound: 0.0149784
time: 2.55 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0045676, 0.0052773, -0.0027961, 0.0026398, -0.0072074, 0.0080734
1: 0.9923048, 1.0060849, 0.9923037, 1.0025672, -0.0102624, 0.0137812
2: -0.0204019, 0.0099128, -0.0204138, 0.0074133, -0.0268025, 0.0294690
3: -0.0006174, 0.0049303, -0.0006217, 0.0049311, -0.0055485, 0.0055520
4: -0.0113745, 0.0145416, -0.0082837, 0.0145510, -0.0259255, 0.0228253
5: -0.0028986, 0.0154026, -0.0029068, 0.0113661, -0.0142647, 0.0183094
6: -0.0062028, 0.0065074, -0.0062069, 0.0065150, -0.0127177, 0.0127143
7: -0.0123296, -0.0014402, -0.0111090, -0.0027253, -0.0096044, 0.0096688
8: -0.0123395, 0.0250572, -0.0104421, 0.0250729, -0.0371688, 0.0352157
9: -0.0109503, 0.0063368, -0.0066691, 0.0052832, -0.0162334, 0.0130059

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A2_A2_B2_B1_B1_B1

### Relational analysis result of NS_A2_A2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151539, upper bound: 0.0151318
time: 2.77 seconds

## Relational analysis of NS_A2_A2_A2_B2_B1_B1_B2

### Relational analysis result of NS_A2_A2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151539, upper bound: 0.0151416
time: 3.12 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0047477, 0.0055454, -0.0030630, 0.0030372, -0.0077850, 0.0086085
1: 0.9923012, 1.0064422, 0.9923131, 1.0030972, -0.0107960, 0.0141291
2: -0.0204389, 0.0101669, -0.0203139, 0.0077899, -0.0272310, 0.0296328
3: -0.0006308, 0.0049327, -0.0005855, 0.0049245, -0.0055553, 0.0055183
4: -0.0116887, 0.0145708, -0.0087495, 0.0144720, -0.0261607, 0.0233203
5: -0.0029240, 0.0158129, -0.0028382, 0.0119743, -0.0148984, 0.0186512
6: -0.0062156, 0.0065309, -0.0061722, 0.0064516, -0.0126672, 0.0127031
7: -0.0124537, -0.0013096, -0.0112930, -0.0025316, -0.0099221, 0.0099834
8: -0.0125324, 0.0251058, -0.0107280, 0.0249416, -0.0372355, 0.0355559
9: -0.0113855, 0.0064440, -0.0073142, 0.0054420, -0.0168275, 0.0137582

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A2_A2_B2_B1_B2_B1

### Relational analysis result of NS_A2_A2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152000, upper bound: 0.0151716
time: 2.61 seconds

## Relational analysis of NS_A2_A2_A2_B2_B1_B2_B2

### Relational analysis result of NS_A2_A2_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152000, upper bound: 0.0151881
time: 2.55 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0046107, 0.0053414, -0.0031862, 0.0032206, -0.0078313, 0.0085276
1: 0.9922788, 1.0061702, 0.9922348, 1.0033419, -0.0110632, 0.0139354
2: -0.0206761, 0.0099735, -0.0211386, 0.0079637, -0.0276495, 0.0301743
3: -0.0007167, 0.0049484, -0.0008842, 0.0049791, -0.0056957, 0.0058327
4: -0.0114496, 0.0147582, -0.0089644, 0.0151238, -0.0265734, 0.0237227
5: -0.0030867, 0.0155006, -0.0034040, 0.0122550, -0.0153418, 0.0189047
6: -0.0062979, 0.0066813, -0.0064584, 0.0069748, -0.0132727, 0.0131398
7: -0.0123593, -0.0014090, -0.0113778, -0.0024423, -0.0099170, 0.0099688
8: -0.0123856, 0.0254174, -0.0108600, 0.0260251, -0.0381440, 0.0360040
9: -0.0110543, 0.0063625, -0.0076119, 0.0055152, -0.0165695, 0.0139744

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A2_A2_B2_B2_B1_B1

### Relational analysis result of NS_A2_A2_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151537, upper bound: 0.0151384
time: 2.77 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_B1_B2

### Relational analysis result of NS_A2_A2_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151537, upper bound: 0.0151485
time: 2.71 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0047911, 0.0056100, -0.0034304, 0.0035841, -0.0083753, 0.0090404
1: 0.9922752, 1.0065285, 0.9922479, 1.0038265, -0.0115513, 0.0142806
2: -0.0207127, 0.0102281, -0.0210005, 0.0083082, -0.0280431, 0.0303068
3: -0.0007300, 0.0049509, -0.0008342, 0.0049699, -0.0056999, 0.0057851
4: -0.0117644, 0.0147872, -0.0093903, 0.0150146, -0.0267790, 0.0241775
5: -0.0031119, 0.0159118, -0.0033092, 0.0128113, -0.0159231, 0.0192210
6: -0.0063107, 0.0067046, -0.0064105, 0.0068871, -0.0131978, 0.0131151
7: -0.0124836, -0.0012781, -0.0115460, -0.0022652, -0.0102184, 0.0102679
8: -0.0125788, 0.0254655, -0.0111214, 0.0258436, -0.0381614, 0.0363164
9: -0.0114903, 0.0064698, -0.0082019, 0.0056604, -0.0171508, 0.0146717

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_A2_A2_B2_B2_B2_B1

### Relational analysis result of NS_A2_A2_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151999, upper bound: 0.0151799
time: 2.77 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_B2_B2

### Relational analysis result of NS_A2_A2_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151999, upper bound: 0.0151954
time: 2.89 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 7.25 seconds
NS_A2_A1_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149659, upper bound: 0.0149567
NS_A2_A1_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149207, upper bound: 0.0149460
NS_A2_A1_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149349, upper bound: 0.0150294
NS_A2_A1_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149207, upper bound: 0.0149752
NS_A2_A1_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149882, upper bound: 0.0149781
NS_A2_A1_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149406, upper bound: 0.0149665
NS_A2_A1_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149589, upper bound: 0.0150535
NS_A2_A1_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149406, upper bound: 0.0149957
NS_A2_A1_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149747, upper bound: 0.0149566
NS_A2_A1_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149290, upper bound: 0.0149457
NS_A2_A1_A1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149403, upper bound: 0.0149914
NS_A2_A1_A1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149290, upper bound: 0.0149746
NS_A2_A1_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149958, upper bound: 0.0149762
NS_A2_A1_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149472, upper bound: 0.0149648
NS_A2_A1_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149635, upper bound: 0.0150493
NS_A2_A1_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149472, upper bound: 0.0149931
NS_A2_A1_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151381, upper bound: 0.0151318
NS_A2_A1_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151381, upper bound: 0.0151493
NS_A2_A1_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151829, upper bound: 0.0151714
NS_A2_A1_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151829, upper bound: 0.0151910
NS_A2_A1_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151379, upper bound: 0.0151384
NS_A2_A1_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151379, upper bound: 0.0151550
NS_A2_A1_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151828, upper bound: 0.0151799
NS_A2_A1_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151828, upper bound: 0.0151975
NS_A2_A2_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149876, upper bound: 0.0149567
NS_A2_A2_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149406, upper bound: 0.0149460
NS_A2_A2_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0150553, upper bound: 0.0150529
NS_A2_A2_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151030, upper bound: 0.0151180
NS_A2_A2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0150095, upper bound: 0.0149781
NS_A2_A2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149597, upper bound: 0.0149665
NS_A2_A2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0150095, upper bound: 0.0149934
NS_A2_A2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149597, upper bound: 0.0149818
NS_A2_A2_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149976, upper bound: 0.0149566
NS_A2_A2_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149483, upper bound: 0.0149457
NS_A2_A2_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0150608, upper bound: 0.0150478
NS_A2_A2_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151093, upper bound: 0.0151164
NS_A2_A2_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0150177, upper bound: 0.0149762
NS_A2_A2_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149647, upper bound: 0.0149648
NS_A2_A2_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0150177, upper bound: 0.0149890
NS_A2_A2_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0149648, upper bound: 0.0149784
NS_A2_A2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151539, upper bound: 0.0151318
NS_A2_A2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151539, upper bound: 0.0151416
NS_A2_A2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0152000, upper bound: 0.0151716
NS_A2_A2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0152000, upper bound: 0.0151881
NS_A2_A2_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151537, upper bound: 0.0151384
NS_A2_A2_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151537, upper bound: 0.0151485
NS_A2_A2_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151999, upper bound: 0.0151799
NS_A2_A2_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.25
Output dim: 1, lower bound: -0.0151999, upper bound: 0.0151954

## BFS NS instance: NS_A2_A1_A1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0021141, 0.0011912, -0.0030147, 0.0029652, -0.0050793, 0.0042059
1: 0.9924780, 1.0005962, 0.9923741, 1.0030013, -0.0105233, 0.0082221
2: -0.0185781, 0.0060128, -0.0196718, 0.0077217, -0.0254519, 0.0247814
3: 0.0000432, 0.0048096, -0.0003529, 0.0048820, -0.0048388, 0.0051625
4: -0.0065519, 0.0131001, -0.0086651, 0.0139645, -0.0205164, 0.0217652
5: -0.0016474, 0.0091044, -0.0023977, 0.0118642, -0.0135115, 0.0115021
6: -0.0055697, 0.0053504, -0.0059493, 0.0060442, -0.0116139, 0.0112997
7: -0.0104251, -0.0034453, -0.0112596, -0.0025667, -0.0078584, 0.0078143
8: -0.0093790, 0.0226609, -0.0106762, 0.0240979, -0.0332226, 0.0330991
9: -0.0042703, 0.0046928, -0.0071974, 0.0054132, -0.0096835, 0.0118902

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 212

## Relational analysis of NS_A2_A1_A1_A1_A1_B1_B1_B1

### Relational analysis result of NS_A2_A1_A1_A1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148434, upper bound: 0.0148048
time: 2.62 seconds

## Relational analysis of NS_A2_A1_A1_A1_A1_B1_B1_B2

### Relational analysis result of NS_A2_A1_A1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149213, upper bound: 0.0149128
time: 2.87 seconds

## BFS NS instance: NS_A2_A1_A1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0021057, 0.0011910, -0.0025782, 0.0022385, -0.0043442, 0.0037692
1: 0.9924808, 1.0003386, 0.9923257, 1.0020320, -0.0095513, 0.0080129
2: -0.0185493, 0.0054491, -0.0201812, 0.0070330, -0.0247947, 0.0247178
3: 0.0000536, 0.0048077, -0.0005375, 0.0049157, -0.0048621, 0.0053451
4: -0.0058897, 0.0130774, -0.0078135, 0.0143671, -0.0202568, 0.0208909
5: -0.0016277, 0.0087740, -0.0027472, 0.0107520, -0.0123797, 0.0115212
6: -0.0055597, 0.0053322, -0.0061261, 0.0063674, -0.0119271, 0.0114583
7: -0.0103018, -0.0037264, -0.0109233, -0.0029208, -0.0073810, 0.0071969
8: -0.0089078, 0.0226231, -0.0101535, 0.0247672, -0.0334180, 0.0325537
9: -0.0037949, 0.0044312, -0.0060178, 0.0051229, -0.0089178, 0.0104490

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 212

## Relational analysis of NS_A2_A1_A1_A1_A1_B1_B2_B1

### Relational analysis result of NS_A2_A1_A1_A1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147772, upper bound: 0.0147841
time: 2.93 seconds

## Relational analysis of NS_A2_A1_A1_A1_A1_B1_B2_B2

### Relational analysis result of NS_A2_A1_A1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148762, upper bound: 0.0149017
time: 2.46 seconds

## BFS NS instance: NS_A2_A1_A1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0021069, 0.0011910, -0.0038009, 0.0041358, -0.0062426, 0.0049919
1: 0.9924804, 1.0003409, 0.9922828, 1.0045621, -0.0120817, 0.0080581
2: -0.0185531, 0.0055143, -0.0206326, 0.0088310, -0.0265548, 0.0252427
3: 0.0000522, 0.0048079, -0.0007010, 0.0049456, -0.0048933, 0.0055089
4: -0.0059412, 0.0130804, -0.0100368, 0.0147239, -0.0206651, 0.0231172
5: -0.0016303, 0.0087881, -0.0030569, 0.0136555, -0.0152858, 0.0118450
6: -0.0055611, 0.0053346, -0.0062828, 0.0066538, -0.0122148, 0.0116174
7: -0.0103141, -0.0037002, -0.0118013, -0.0019964, -0.0083177, 0.0081012
8: -0.0089935, 0.0226282, -0.0115183, 0.0253603, -0.0341020, 0.0339119
9: -0.0037955, 0.0044787, -0.0090973, 0.0058808, -0.0096763, 0.0135761

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A2_A1_A1_A1_A1_B2_A1_B1

### Relational analysis result of NS_A2_A1_A1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149352, upper bound: 0.0150294
time: 2.49 seconds

## Relational analysis of NS_A2_A1_A1_A1_A1_B2_A1_B2

### Relational analysis result of NS_A2_A1_A1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149352, upper bound: 0.0150294
time: 2.57 seconds

## BFS NS instance: NS_A2_A1_A1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0022609, 0.0011950, -0.0033689, 0.0034926, -0.0057535, 0.0045639
1: 0.9924299, 1.0006671, 0.9922856, 1.0037043, -0.0112745, 0.0083815
2: -0.0190852, 0.0051006, -0.0206040, 0.0082215, -0.0264668, 0.0248653
3: -0.0001405, 0.0048431, -0.0006906, 0.0049437, -0.0050841, 0.0055337
4: -0.0056142, 0.0135009, -0.0092831, 0.0147013, -0.0203155, 0.0227840
5: -0.0019953, 0.0086986, -0.0030373, 0.0126712, -0.0146665, 0.0117359
6: -0.0057457, 0.0056721, -0.0062729, 0.0066356, -0.0123813, 0.0119450
7: -0.0102359, -0.0038666, -0.0115037, -0.0023098, -0.0079261, 0.0076371
8: -0.0084500, 0.0233272, -0.0110556, 0.0253227, -0.0335383, 0.0341464
9: -0.0037913, 0.0041769, -0.0080534, 0.0056239, -0.0094152, 0.0122303

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 212

## Relational analysis of NS_A2_A1_A1_A1_A1_B2_A2_B1

### Relational analysis result of NS_A2_A1_A1_A1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147773, upper bound: 0.0148216
time: 2.50 seconds

## Relational analysis of NS_A2_A1_A1_A1_A1_B2_A2_B2

### Relational analysis result of NS_A2_A1_A1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148765, upper bound: 0.0149314
time: 2.54 seconds

## BFS NS instance: NS_A2_A1_A1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0021198, 0.0016329, -0.0038053, 0.0033084, -0.0054282, 0.0054382
1: 0.9924861, 1.0012243, 0.9922059, 1.0034590, -0.0109729, 0.0090184
2: -0.0184929, 0.0064591, -0.0197188, 0.0082767, -0.0260014, 0.0252898
3: 0.0000741, 0.0048039, -0.0005256, 0.0048843, -0.0048103, 0.0053295
4: -0.0071038, 0.0130328, -0.0091523, 0.0140019, -0.0211057, 0.0221851
5: -0.0015889, 0.0098252, -0.0024411, 0.0128267, -0.0144157, 0.0122663
6: -0.0055401, 0.0052964, -0.0059690, 0.0061288, -0.0116690, 0.0112654
7: -0.0106431, -0.0032159, -0.0114185, -0.0012007, -0.0094423, 0.0082026
8: -0.0097178, 0.0225490, -0.0109231, 0.0242469, -0.0337122, 0.0332394
9: -0.0050347, 0.0048810, -0.0079454, 0.0055503, -0.0105851, 0.0128264

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 212

## Relational analysis of NS_A2_A1_A1_A1_A2_B1_B1_A1

### Relational analysis result of NS_A2_A1_A1_A1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148322, upper bound: 0.0148396
time: 2.65 seconds

## Relational analysis of NS_A2_A1_A1_A1_A2_B1_B1_A2

### Relational analysis result of NS_A2_A1_A1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149419, upper bound: 0.0149320
time: 2.62 seconds

## BFS NS instance: NS_A2_A1_A1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0020810, 0.0011903, -0.0047564, 0.0026211, -0.0047021, 0.0059468
1: 0.9924889, 1.0003659, 0.9915804, 1.0025423, -0.0100534, 0.0087855
2: -0.0184637, 0.0058333, -0.0202491, 0.0081216, -0.0260009, 0.0252220
3: 0.0000846, 0.0048020, -0.0013627, 0.0049182, -0.0048336, 0.0061647
4: -0.0063299, 0.0130097, -0.0085396, 0.0144233, -0.0207532, 0.0215493
5: -0.0015689, 0.0088603, -0.0028312, 0.0127586, -0.0143275, 0.0116915
6: -0.0055300, 0.0052778, -0.0061573, 0.0066030, -0.0121330, 0.0114352
7: -0.0103500, -0.0035376, -0.0111004, 0.0013646, -0.0117145, 0.0075628
8: -0.0092427, 0.0225107, -0.0104287, 0.0251482, -0.0341354, 0.0327242
9: -0.0039976, 0.0046171, -0.0072741, 0.0052757, -0.0092733, 0.0118912

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A1_A1_A2_B1_B2_B1

### Relational analysis result of NS_A2_A1_A1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148792, upper bound: 0.0149233
time: 2.90 seconds

## Relational analysis of NS_A2_A1_A1_A1_A2_B1_B2_B2

### Relational analysis result of NS_A2_A1_A1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149258, upper bound: 0.0149515
time: 2.58 seconds

## BFS NS instance: NS_A2_A1_A1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0020820, 0.0011904, -0.0042199, 0.0044205, -0.0065025, 0.0054103
1: 0.9924885, 1.0005062, 0.9922134, 1.0049421, -0.0124536, 0.0082928
2: -0.0184674, 0.0059487, -0.0206752, 0.0092025, -0.0268864, 0.0257356
3: 0.0000833, 0.0048022, -0.0007757, 0.0049480, -0.0048647, 0.0055780
4: -0.0064726, 0.0130126, -0.0104079, 0.0147572, -0.0212298, 0.0234205
5: -0.0015715, 0.0090009, -0.0030904, 0.0142804, -0.0158519, 0.0120913
6: -0.0055313, 0.0052802, -0.0062997, 0.0067043, -0.0122356, 0.0115799
7: -0.0103938, -0.0034783, -0.0119331, -0.0013586, -0.0090352, 0.0084548
8: -0.0093303, 0.0225155, -0.0117231, 0.0254541, -0.0345352, 0.0340114
9: -0.0041605, 0.0046658, -0.0096405, 0.0059946, -0.0101550, 0.0143063

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 212

## Relational analysis of NS_A2_A1_A1_A1_A2_B2_A1_A1

### Relational analysis result of NS_A2_A1_A1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148138, upper bound: 0.0149325
time: 2.56 seconds

## Relational analysis of NS_A2_A1_A1_A1_A2_B2_A1_A2

### Relational analysis result of NS_A2_A1_A1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149125, upper bound: 0.0150079
time: 2.36 seconds

## BFS NS instance: NS_A2_A1_A1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0022289, 0.0011942, -0.0035717, 0.0037757, -0.0060045, 0.0047659
1: 0.9924403, 1.0005993, 0.9922785, 1.0040821, -0.0116418, 0.0083208
2: -0.0189747, 0.0053750, -0.0206408, 0.0084955, -0.0266538, 0.0251835
3: -0.0001005, 0.0048358, -0.0007070, 0.0049461, -0.0050465, 0.0055428
4: -0.0058311, 0.0134136, -0.0096170, 0.0147303, -0.0205614, 0.0230305
5: -0.0019195, 0.0087579, -0.0030628, 0.0131154, -0.0150348, 0.0118207
6: -0.0057074, 0.0056020, -0.0062858, 0.0066603, -0.0123677, 0.0118878
7: -0.0102877, -0.0037562, -0.0116347, -0.0021446, -0.0081431, 0.0078785
8: -0.0088104, 0.0231820, -0.0112592, 0.0253733, -0.0339515, 0.0342129
9: -0.0037941, 0.0043771, -0.0085176, 0.0057370, -0.0095311, 0.0128947

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_A1_A1_A1_A2_B2_A2_B1

### Relational analysis result of NS_A2_A1_A1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149185, upper bound: 0.0149781
time: 2.34 seconds

## Relational analysis of NS_A2_A1_A1_A1_A2_B2_A2_B2

### Relational analysis result of NS_A2_A1_A1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149185, upper bound: 0.0149951
time: 2.46 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0023375, 0.0018652, -0.0030622, 0.0030359, -0.0053734, 0.0049274
1: 0.9924047, 1.0015342, 0.9923443, 1.0030955, -0.0106908, 0.0091899
2: -0.0193497, 0.0066793, -0.0199857, 0.0077887, -0.0261961, 0.0257637
3: -0.0002362, 0.0048606, -0.0004666, 0.0049028, -0.0051390, 0.0053273
4: -0.0073761, 0.0137099, -0.0087480, 0.0142126, -0.0215887, 0.0224579
5: -0.0021767, 0.0101807, -0.0026131, 0.0119724, -0.0141491, 0.0127938
6: -0.0058375, 0.0058399, -0.0060583, 0.0062434, -0.0120809, 0.0118982
7: -0.0107506, -0.0031026, -0.0112924, -0.0025322, -0.0082183, 0.0081897
8: -0.0098849, 0.0236746, -0.0107271, 0.0245104, -0.0341412, 0.0341378
9: -0.0054119, 0.0049738, -0.0073121, 0.0054414, -0.0108533, 0.0122859

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 212

## Relational analysis of NS_A2_A1_A1_A2_A1_B1_B1_B1

### Relational analysis result of NS_A2_A1_A1_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148533, upper bound: 0.0148036
time: 2.57 seconds

## Relational analysis of NS_A2_A1_A1_A2_A1_B1_B1_B2

### Relational analysis result of NS_A2_A1_A1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149298, upper bound: 0.0149126
time: 2.60 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0023293, 0.0012247, -0.0026678, 0.0023056, -0.0046349, 0.0038924
1: 0.9924074, 1.0008119, 0.9922963, 1.0021214, -0.0097141, 0.0085156
2: -0.0193214, 0.0060723, -0.0204906, 0.0070966, -0.0255365, 0.0256471
3: -0.0002260, 0.0048588, -0.0006495, 0.0049362, -0.0051622, 0.0055082
4: -0.0066255, 0.0136876, -0.0078921, 0.0146116, -0.0212371, 0.0215797
5: -0.0021574, 0.0092004, -0.0029594, 0.0108547, -0.0130121, 0.0121599
6: -0.0058277, 0.0058220, -0.0062335, 0.0065636, -0.0123914, 0.0120555
7: -0.0104541, -0.0034147, -0.0109544, -0.0028881, -0.0075660, 0.0075396
8: -0.0094241, 0.0236376, -0.0102017, 0.0251736, -0.0343393, 0.0335915
9: -0.0043721, 0.0047179, -0.0061267, 0.0051497, -0.0095218, 0.0108446

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 212

## Relational analysis of NS_A2_A1_A1_A2_A1_B1_B2_A1

### Relational analysis result of NS_A2_A1_A1_A2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147705, upper bound: 0.0147988
time: 2.53 seconds

## Relational analysis of NS_A2_A1_A1_A2_A1_B1_B2_A2

### Relational analysis result of NS_A2_A1_A1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148849, upper bound: 0.0149016
time: 2.66 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0023304, 0.0013113, -0.0038433, 0.0041990, -0.0065294, 0.0051546
1: 0.9924070, 1.0008142, 0.9922568, 1.0046467, -0.0122396, 0.0085574
2: -0.0193253, 0.0061543, -0.0209068, 0.0088909, -0.0272936, 0.0261686
3: -0.0002274, 0.0048590, -0.0008003, 0.0049637, -0.0051912, 0.0056593
4: -0.0067269, 0.0136906, -0.0101108, 0.0149406, -0.0216675, 0.0238015
5: -0.0021600, 0.0093329, -0.0032450, 0.0137523, -0.0159123, 0.0125780
6: -0.0058291, 0.0058244, -0.0063780, 0.0068277, -0.0126568, 0.0122025
7: -0.0104942, -0.0033725, -0.0118306, -0.0019656, -0.0085286, 0.0084581
8: -0.0094864, 0.0236426, -0.0115637, 0.0257206, -0.0349567, 0.0349471
9: -0.0045127, 0.0047525, -0.0091999, 0.0059061, -0.0104188, 0.0139524

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 212

## Relational analysis of NS_A2_A1_A1_A2_A1_B2_A1_B1

### Relational analysis result of NS_A2_A1_A1_A2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147980, upper bound: 0.0148691
time: 2.78 seconds

## Relational analysis of NS_A2_A1_A1_A2_A1_B2_A1_B2

### Relational analysis result of NS_A2_A1_A1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148965, upper bound: 0.0149819
time: 2.70 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0024789, 0.0012007, -0.0034116, 0.0035562, -0.0060351, 0.0046124
1: 0.9923584, 1.0011286, 0.9922595, 1.0037893, -0.0114309, 0.0088691
2: -0.0198381, 0.0054792, -0.0208782, 0.0082818, -0.0271828, 0.0255318
3: -0.0004132, 0.0048930, -0.0007899, 0.0049618, -0.0053750, 0.0056829
4: -0.0059135, 0.0140959, -0.0093576, 0.0149180, -0.0208315, 0.0234536
5: -0.0025118, 0.0087805, -0.0032254, 0.0127686, -0.0152804, 0.0120059
6: -0.0060071, 0.0061497, -0.0063681, 0.0068096, -0.0128166, 0.0125178
7: -0.0103075, -0.0037143, -0.0115331, -0.0022788, -0.0080287, 0.0078188
8: -0.0089474, 0.0243164, -0.0111013, 0.0256830, -0.0343990, 0.0351545
9: -0.0037952, 0.0044532, -0.0081566, 0.0056493, -0.0094445, 0.0126098

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A2_A1_A1_A2_A1_B2_A2_B1

### Relational analysis result of NS_A2_A1_A1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149292, upper bound: 0.0149746
time: 2.65 seconds

## Relational analysis of NS_A2_A1_A1_A2_A1_B2_A2_B2

### Relational analysis result of NS_A2_A1_A1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149292, upper bound: 0.0149746
time: 2.83 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0025545, 0.0022800, -0.0039434, 0.0033781, -0.0059325, 0.0062234
1: 0.9924143, 1.0020874, 0.9921479, 1.0035518, -0.0111375, 0.0099394
2: -0.0192492, 0.0070724, -0.0200341, 0.0083767, -0.0267604, 0.0262202
3: -0.0001999, 0.0048540, -0.0006648, 0.0049051, -0.0051050, 0.0055188
4: -0.0078622, 0.0136306, -0.0092471, 0.0142510, -0.0221132, 0.0228777
5: -0.0021078, 0.0108155, -0.0026589, 0.0129973, -0.0151051, 0.0134744
6: -0.0058027, 0.0057762, -0.0060791, 0.0063373, -0.0121400, 0.0118553
7: -0.0109426, -0.0029005, -0.0114507, -0.0009740, -0.0099686, 0.0085501
8: -0.0101833, 0.0235428, -0.0109732, 0.0246748, -0.0346034, 0.0342545
9: -0.0060852, 0.0051395, -0.0080862, 0.0055781, -0.0116633, 0.0132257

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 212

## Relational analysis of NS_A2_A1_A1_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A1_A1_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148385, upper bound: 0.0148391
time: 2.83 seconds

## Relational analysis of NS_A2_A1_A1_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A1_A1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149495, upper bound: 0.0149300
time: 2.98 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0023001, 0.0016295, -0.0049037, 0.0026877, -0.0049878, 0.0065331
1: 0.9924170, 1.0012197, 0.9915001, 1.0026311, -0.0102141, 0.0097196
2: -0.0192207, 0.0064558, -0.0205594, 0.0082204, -0.0267621, 0.0261558
3: -0.0001895, 0.0048521, -0.0014989, 0.0049386, -0.0051281, 0.0063510
4: -0.0070998, 0.0136080, -0.0086329, 0.0146685, -0.0217683, 0.0222409
5: -0.0020882, 0.0098199, -0.0030456, 0.0129270, -0.0150153, 0.0128655
6: -0.0057928, 0.0057581, -0.0062657, 0.0068093, -0.0126021, 0.0120238
7: -0.0106415, -0.0032175, -0.0111312, 0.0016025, -0.0122439, 0.0079137
8: -0.0097153, 0.0235052, -0.0104766, 0.0255716, -0.0350275, 0.0337389
9: -0.0050292, 0.0048796, -0.0074110, 0.0053023, -0.0103315, 0.0122906

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 212

## Relational analysis of NS_A2_A1_A1_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A1_A1_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147926, upper bound: 0.0148249
time: 5.06 seconds

## Relational analysis of NS_A2_A1_A1_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A1_A1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149016, upper bound: 0.0149187
time: 2.49 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0023011, 0.0017485, -0.0043420, 0.0044832, -0.0067843, 0.0060906
1: 0.9924167, 1.0013784, 0.9921635, 1.0050256, -0.0126089, 0.0092149
2: -0.0192240, 0.0065687, -0.0209509, 0.0092950, -0.0276412, 0.0266418
3: -0.0001907, 0.0048523, -0.0008972, 0.0049662, -0.0051569, 0.0057495
4: -0.0072393, 0.0136106, -0.0104937, 0.0149750, -0.0222144, 0.0241044
5: -0.0020905, 0.0100021, -0.0032808, 0.0144376, -0.0165281, 0.0132830
6: -0.0057939, 0.0057602, -0.0063961, 0.0068867, -0.0126807, 0.0121563
7: -0.0106966, -0.0031595, -0.0119621, -0.0011526, -0.0095440, 0.0088026
8: -0.0098010, 0.0235096, -0.0117682, 0.0258282, -0.0353814, 0.0350225
9: -0.0052224, 0.0049272, -0.0097683, 0.0060196, -0.0112420, 0.0146955

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 212

## Relational analysis of NS_A2_A1_A1_A2_A2_B2_A1_A1

### Relational analysis result of NS_A2_A1_A1_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148160, upper bound: 0.0149305
time: 2.30 seconds

## Relational analysis of NS_A2_A1_A1_A2_A2_B2_A1_A2

### Relational analysis result of NS_A2_A1_A1_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149177, upper bound: 0.0150039
time: 2.36 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0024452, 0.0011998, -0.0036864, 0.0038387, -0.0062839, 0.0048862
1: 0.9923694, 1.0010574, 0.9922321, 1.0041660, -0.0117966, 0.0088253
2: -0.0197219, 0.0058772, -0.0209163, 0.0085882, -0.0273990, 0.0259778
3: -0.0003710, 0.0048853, -0.0008245, 0.0049642, -0.0053352, 0.0057098
4: -0.0063842, 0.0140041, -0.0097032, 0.0149481, -0.0213323, 0.0237072
5: -0.0024321, 0.0088854, -0.0032532, 0.0132733, -0.0157054, 0.0121386
6: -0.0059667, 0.0060760, -0.0063820, 0.0068426, -0.0128093, 0.0124580
7: -0.0103589, -0.0035150, -0.0116638, -0.0019580, -0.0084008, 0.0081488
8: -0.0092760, 0.0241637, -0.0113045, 0.0257474, -0.0347927, 0.0352109
9: -0.0040380, 0.0046356, -0.0086465, 0.0057621, -0.0098001, 0.0132821

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 212

## Relational analysis of NS_A2_A1_A1_A2_A2_B2_A2_A1

### Relational analysis result of NS_A2_A1_A1_A2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147926, upper bound: 0.0148559
time: 2.25 seconds

## Relational analysis of NS_A2_A1_A1_A2_A2_B2_A2_A2

### Relational analysis result of NS_A2_A1_A1_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149016, upper bound: 0.0149479
time: 2.43 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0040121, 0.0044501, -0.0022636, 0.0016323, -0.0056443, 0.0067137
1: 0.9923987, 1.0049816, 0.9924289, 1.0012236, -0.0088249, 0.0125527
2: -0.0194131, 0.0091289, -0.0190944, 0.0064585, -0.0249617, 0.0273408
3: -0.0002592, 0.0048648, -0.0001438, 0.0048437, -0.0051030, 0.0050086
4: -0.0104052, 0.0137601, -0.0071031, 0.0135082, -0.0239133, 0.0208632
5: -0.0022203, 0.0141366, -0.0020016, 0.0098242, -0.0120445, 0.0161382
6: -0.0058596, 0.0058802, -0.0057489, 0.0056780, -0.0115375, 0.0116291
7: -0.0119468, -0.0018432, -0.0106428, -0.0032161, -0.0087307, 0.0087996
8: -0.0117444, 0.0237581, -0.0097173, 0.0233393, -0.0348330, 0.0332152
9: -0.0096076, 0.0060064, -0.0050338, 0.0048807, -0.0144883, 0.0110402

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_A1_A2_B2_B1_B1_B1_A1

### Relational analysis result of NS_A2_A1_A2_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149391, upper bound: 0.0149613
time: 2.93 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_B1_B1_A2

### Relational analysis result of NS_A2_A1_A2_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149261, upper bound: 0.0149179
time: 2.88 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0040121, 0.0044501, -0.0026366, 0.0024024, -0.0064144, 0.0070867
1: 0.9923987, 1.0049816, 0.9923339, 1.0022504, -0.0098518, 0.0126477
2: -0.0194131, 0.0091289, -0.0200945, 0.0071883, -0.0257174, 0.0283504
3: -0.0002592, 0.0048648, -0.0005060, 0.0049099, -0.0051692, 0.0053709
4: -0.0104052, 0.0137601, -0.0080055, 0.0142986, -0.0247037, 0.0217656
5: -0.0022203, 0.0141366, -0.0026877, 0.0110027, -0.0132230, 0.0168244
6: -0.0058596, 0.0058802, -0.0060961, 0.0063124, -0.0121720, 0.0119762
7: -0.0119468, -0.0018432, -0.0109991, -0.0028410, -0.0091059, 0.0091559
8: -0.0117444, 0.0237581, -0.0102713, 0.0246533, -0.0361524, 0.0337746
9: -0.0096076, 0.0060064, -0.0062837, 0.0051883, -0.0147959, 0.0122901

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_A1_A2_B2_B1_B1_B2_A1

### Relational analysis result of NS_A2_A1_A2_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149391, upper bound: 0.0149840
time: 2.91 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_B1_B2_A2

### Relational analysis result of NS_A2_A1_A2_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149261, upper bound: 0.0149396
time: 2.58 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0042063, 0.0047394, -0.0024220, 0.0020829, -0.0062892, 0.0071614
1: 0.9923952, 1.0053673, 0.9924377, 1.0018243, -0.0094290, 0.0129296
2: -0.0194490, 0.0094030, -0.0190025, 0.0068855, -0.0254257, 0.0275449
3: -0.0002723, 0.0048672, -0.0001105, 0.0048377, -0.0051099, 0.0049777
4: -0.0107441, 0.0137885, -0.0076311, 0.0134356, -0.0241797, 0.0214196
5: -0.0022449, 0.0145793, -0.0019386, 0.0105138, -0.0127587, 0.0165179
6: -0.0058720, 0.0059029, -0.0057170, 0.0056197, -0.0114917, 0.0116199
7: -0.0120807, -0.0017023, -0.0108513, -0.0029966, -0.0090841, 0.0091490
8: -0.0119525, 0.0238052, -0.0100415, 0.0232186, -0.0349268, 0.0335892
9: -0.0100771, 0.0061220, -0.0057651, 0.0050607, -0.0151378, 0.0118871

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A2_B2_B1_B2_B1_A1

### Relational analysis result of NS_A2_A1_A2_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151534, upper bound: 0.0151341
time: 2.54 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_B2_B1_A2

### Relational analysis result of NS_A2_A1_A2_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151680, upper bound: 0.0151568
time: 2.87 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0042063, 0.0047394, -0.0029056, 0.0028028, -0.0070091, 0.0076449
1: 0.9923952, 1.0053673, 0.9923436, 1.0027846, -0.0103894, 0.0130237
2: -0.0194490, 0.0094030, -0.0199932, 0.0075678, -0.0261309, 0.0285358
3: -0.0002723, 0.0048672, -0.0004694, 0.0049032, -0.0051755, 0.0053366
4: -0.0107441, 0.0137885, -0.0084747, 0.0142186, -0.0249627, 0.0222632
5: -0.0022449, 0.0145793, -0.0026182, 0.0116156, -0.0138605, 0.0171975
6: -0.0058720, 0.0059029, -0.0060609, 0.0062482, -0.0121202, 0.0119638
7: -0.0120807, -0.0017023, -0.0111845, -0.0026459, -0.0094348, 0.0094821
8: -0.0119525, 0.0238052, -0.0105594, 0.0245202, -0.0362319, 0.0341117
9: -0.0100771, 0.0061220, -0.0069337, 0.0053483, -0.0154254, 0.0130556

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A2_B2_B1_B2_B2_A1

### Relational analysis result of NS_A2_A1_A2_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151534, upper bound: 0.0151528
time: 2.59 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_B2_B2_A2

### Relational analysis result of NS_A2_A1_A2_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151680, upper bound: 0.0151761
time: 2.93 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0040601, 0.0045217, -0.0025802, 0.0023183, -0.0063784, 0.0071018
1: 0.9923701, 1.0050771, 0.9923562, 1.0021385, -0.0097684, 0.0127209
2: -0.0197150, 0.0091967, -0.0198600, 0.0071087, -0.0259221, 0.0280891
3: -0.0003686, 0.0048848, -0.0004211, 0.0048944, -0.0052630, 0.0053059
4: -0.0104890, 0.0139987, -0.0079070, 0.0141133, -0.0246023, 0.0219057
5: -0.0024274, 0.0142462, -0.0025268, 0.0108741, -0.0133015, 0.0167730
6: -0.0059643, 0.0060716, -0.0060147, 0.0061636, -0.0121280, 0.0120863
7: -0.0119799, -0.0018084, -0.0109603, -0.0028819, -0.0090980, 0.0091519
8: -0.0117959, 0.0241547, -0.0102109, 0.0243452, -0.0358687, 0.0341072
9: -0.0097237, 0.0060350, -0.0061473, 0.0051548, -0.0148785, 0.0121823

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_A1_A2_B2_B2_B1_B1_A1

### Relational analysis result of NS_A2_A1_A2_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149391, upper bound: 0.0149714
time: 2.94 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_B1_B1_A2

### Relational analysis result of NS_A2_A1_A2_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149261, upper bound: 0.0149267
time: 2.84 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0040601, 0.0045217, -0.0030327, 0.0029920, -0.0070521, 0.0075544
1: 0.9923701, 1.0050771, 0.9922667, 1.0030369, -0.0106668, 0.0128104
2: -0.0197150, 0.0091967, -0.0208022, 0.0077471, -0.0265791, 0.0290506
3: -0.0003686, 0.0048848, -0.0007624, 0.0049568, -0.0053254, 0.0056472
4: -0.0104890, 0.0139987, -0.0086965, 0.0148580, -0.0253470, 0.0226952
5: -0.0024274, 0.0142462, -0.0031733, 0.0119052, -0.0143326, 0.0174194
6: -0.0059643, 0.0060716, -0.0063417, 0.0067614, -0.0127257, 0.0124134
7: -0.0119799, -0.0018084, -0.0112720, -0.0025536, -0.0094263, 0.0094637
8: -0.0117959, 0.0241547, -0.0106955, 0.0255832, -0.0371105, 0.0345975
9: -0.0097237, 0.0060350, -0.0072409, 0.0054239, -0.0151477, 0.0132759

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_A1_A2_B2_B2_B1_B2_A1

### Relational analysis result of NS_A2_A1_A2_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149391, upper bound: 0.0149940
time: 2.77 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_B1_B2_A2

### Relational analysis result of NS_A2_A1_A2_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149261, upper bound: 0.0149477
time: 3.00 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0042539, 0.0048101, -0.0028611, 0.0027366, -0.0069905, 0.0076713
1: 0.9923666, 1.0054616, 0.9923674, 1.0026964, -0.0103298, 0.0130941
2: -0.0197504, 0.0094700, -0.0197428, 0.0075051, -0.0263520, 0.0282610
3: -0.0003814, 0.0048872, -0.0003786, 0.0048867, -0.0052681, 0.0052658
4: -0.0108270, 0.0140267, -0.0083972, 0.0140206, -0.0248477, 0.0224239
5: -0.0024516, 0.0146876, -0.0024465, 0.0115143, -0.0139659, 0.0171341
6: -0.0059766, 0.0060941, -0.0059740, 0.0060893, -0.0120659, 0.0120681
7: -0.0121134, -0.0016678, -0.0111538, -0.0026781, -0.0094353, 0.0094860
8: -0.0120034, 0.0242012, -0.0105118, 0.0241912, -0.0359257, 0.0344548
9: -0.0101920, 0.0061502, -0.0068263, 0.0053219, -0.0155138, 0.0129765

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A2_B2_B2_B2_B1_A1

### Relational analysis result of NS_A2_A1_A2_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151533, upper bound: 0.0151424
time: 2.98 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_B2_B1_A2

### Relational analysis result of NS_A2_A1_A2_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151678, upper bound: 0.0151652
time: 2.99 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0042539, 0.0048101, -0.0032770, 0.0033557, -0.0076096, 0.0080871
1: 0.9923666, 1.0054616, 0.9922804, 1.0035218, -0.0111552, 0.0131811
2: -0.0197504, 0.0094700, -0.0206577, 0.0080918, -0.0269632, 0.0291982
3: -0.0003814, 0.0048872, -0.0007100, 0.0049472, -0.0053286, 0.0055972
4: -0.0108270, 0.0140267, -0.0091227, 0.0147437, -0.0255707, 0.0231494
5: -0.0024516, 0.0146876, -0.0030741, 0.0124618, -0.0149134, 0.0177617
6: -0.0059766, 0.0060941, -0.0062915, 0.0066697, -0.0126463, 0.0123856
7: -0.0121134, -0.0016678, -0.0114404, -0.0023765, -0.0097370, 0.0097725
8: -0.0120034, 0.0242012, -0.0109571, 0.0253933, -0.0371359, 0.0349079
9: -0.0101920, 0.0061502, -0.0078312, 0.0055692, -0.0157612, 0.0139814

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A2_B2_B2_B2_B2_A1

### Relational analysis result of NS_A2_A1_A2_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151533, upper bound: 0.0151597
time: 2.88 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_B2_B2_A2

### Relational analysis result of NS_A2_A1_A2_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151678, upper bound: 0.0151829
time: 3.00 seconds

## BFS NS instance: NS_A2_A2_A1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0023868, 0.0019363, -0.0030147, 0.0029652, -0.0053521, 0.0049509
1: 0.9923885, 1.0016289, 0.9923741, 1.0030013, -0.0106128, 0.0092548
2: -0.0195203, 0.0067466, -0.0196718, 0.0077217, -0.0263909, 0.0255432
3: -0.0002980, 0.0048719, -0.0003529, 0.0048820, -0.0051800, 0.0052248
4: -0.0074593, 0.0138447, -0.0086651, 0.0139645, -0.0214238, 0.0225098
5: -0.0022937, 0.0102894, -0.0023977, 0.0118642, -0.0141579, 0.0126871
6: -0.0058967, 0.0059481, -0.0059493, 0.0060442, -0.0119409, 0.0118974
7: -0.0107834, -0.0030681, -0.0112596, -0.0025667, -0.0082167, 0.0081916
8: -0.0099360, 0.0238988, -0.0106762, 0.0240979, -0.0337860, 0.0343364
9: -0.0055271, 0.0050021, -0.0071974, 0.0054132, -0.0109404, 0.0121995

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 212

## Relational analysis of NS_A2_A2_A1_A1_A1_B1_B1_B1

### Relational analysis result of NS_A2_A2_A1_A1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148677, upper bound: 0.0148048
time: 2.94 seconds

## Relational analysis of NS_A2_A2_A1_A1_A1_B1_B1_B2

### Relational analysis result of NS_A2_A2_A1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149431, upper bound: 0.0149128
time: 3.22 seconds

## BFS NS instance: NS_A2_A2_A1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0023784, 0.0013009, -0.0025782, 0.0022385, -0.0046169, 0.0038791
1: 0.9923913, 1.0009161, 0.9923257, 1.0020320, -0.0096408, 0.0085905
2: -0.0194911, 0.0061445, -0.0201812, 0.0070330, -0.0257329, 0.0254376
3: -0.0002875, 0.0048700, -0.0005375, 0.0049157, -0.0052031, 0.0054075
4: -0.0067147, 0.0138217, -0.0078135, 0.0143671, -0.0210818, 0.0216352
5: -0.0022738, 0.0093170, -0.0027472, 0.0107520, -0.0130258, 0.0120643
6: -0.0058866, 0.0059296, -0.0061261, 0.0063674, -0.0122540, 0.0120557
7: -0.0104894, -0.0033776, -0.0109233, -0.0029208, -0.0075686, 0.0075457
8: -0.0094789, 0.0238605, -0.0101535, 0.0247672, -0.0339945, 0.0337908
9: -0.0044958, 0.0047483, -0.0060178, 0.0051229, -0.0096187, 0.0107661

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 212

## Relational analysis of NS_A2_A2_A1_A1_A1_B1_B2_B1

### Relational analysis result of NS_A2_A2_A1_A1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147939, upper bound: 0.0147841
time: 2.29 seconds

## Relational analysis of NS_A2_A2_A1_A1_A1_B1_B2_B2

### Relational analysis result of NS_A2_A2_A1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148963, upper bound: 0.0149017
time: 2.40 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.21 + 595.85 = 601.06 seconds
