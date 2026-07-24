## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.011909105


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0046572, 0.0033948, -0.0046572, 0.0033948, -0.0080520, 0.0080520)
1: (0.9879019, 1.0062624, 0.9879019, 1.0062624, -0.0183606, 0.0183606)
2: (-0.0157628, 0.0036852, -0.0157628, 0.0036852, -0.0189225, 0.0189225)
3: (0.0004106, 0.0063618, 0.0004106, 0.0063618, -0.0059512, 0.0059512)
4: (-0.0049218, 0.0108751, -0.0049218, 0.0108751, -0.0157968, 0.0157968)
5: (-0.0013414, 0.0120105, -0.0013414, 0.0120105, -0.0133519, 0.0133519)
6: (-0.0049703, 0.0050756, -0.0049703, 0.0050756, -0.0100459, 0.0100459)
7: (-0.0122603, -0.0013753, -0.0122603, -0.0013753, -0.0108851, 0.0108851)
8: (-0.0114670, 0.0189620, -0.0114670, 0.0189620, -0.0302684, 0.0302684)
9: (-0.0110455, 0.0063901, -0.0110455, 0.0063901, -0.0174356, 0.0174356)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.81 + 2.91 = 4.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0125356, upper bound: 0.0125359

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124479, upper bound: 0.0125331
time: 2.34 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0125068, upper bound: 0.0125069
time: 1.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.33 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.33
Output dim: 1, lower bound: -0.0124479, upper bound: 0.0125331
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.33
Output dim: 1, lower bound: -0.0125068, upper bound: 0.0125069

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0034953, 0.0030272, -0.0046204, 0.0033551, -0.0068504, 0.0076476
1: 0.9888287, 1.0039555, 0.9880019, 1.0061896, -0.0173609, 0.0159535
2: -0.0141504, 0.0024819, -0.0155887, 0.0036471, -0.0172652, 0.0175164
3: 0.0007359, 0.0059773, 0.0004209, 0.0063203, -0.0055844, 0.0055564
4: -0.0033131, 0.0096007, -0.0047481, 0.0107375, -0.0140506, 0.0143488
5: -0.0008122, 0.0111097, -0.0013246, 0.0119133, -0.0127255, 0.0124343
6: -0.0040328, 0.0038113, -0.0048955, 0.0049391, -0.0089719, 0.0087069
7: -0.0116117, -0.0022180, -0.0121903, -0.0014019, -0.0102097, 0.0099723
8: -0.0102891, 0.0168435, -0.0114297, 0.0187333, -0.0288554, 0.0281132
9: -0.0098691, 0.0056991, -0.0109185, 0.0063682, -0.0162373, 0.0166176

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123591, upper bound: 0.0123929
time: 2.93 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123266, upper bound: 0.0124142
time: 2.77 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0045201, 0.0032881, -0.0046572, 0.0033948, -0.0079148, 0.0079453
1: 0.9881709, 1.0059903, 0.9879019, 1.0062624, -0.0180916, 0.0180884
2: -0.0152950, 0.0035432, -0.0157628, 0.0036852, -0.0183747, 0.0187808
3: 0.0004490, 0.0062502, 0.0004106, 0.0063618, -0.0059128, 0.0058396
4: -0.0044551, 0.0105054, -0.0049218, 0.0108751, -0.0153301, 0.0154271
5: -0.0012789, 0.0117491, -0.0013414, 0.0120105, -0.0132894, 0.0130905
6: -0.0046915, 0.0047088, -0.0049703, 0.0050756, -0.0097671, 0.0096791
7: -0.0120721, -0.0014747, -0.0122603, -0.0013753, -0.0106969, 0.0107856
8: -0.0113280, 0.0183474, -0.0114670, 0.0189620, -0.0301297, 0.0296345
9: -0.0107042, 0.0063085, -0.0110455, 0.0063901, -0.0170943, 0.0173540

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124445, upper bound: 0.0123914
time: 3.17 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124136, upper bound: 0.0124132
time: 2.56 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 7.61 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 7.61
Output dim: 1, lower bound: -0.0123591, upper bound: 0.0123929
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 7.61
Output dim: 1, lower bound: -0.0123266, upper bound: 0.0124142
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 7.61
Output dim: 1, lower bound: -0.0124445, upper bound: 0.0123914
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 7.61
Output dim: 1, lower bound: -0.0124136, upper bound: 0.0124132

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0034953, 0.0030272, -0.0043781, 0.0031330, -0.0066284, 0.0074053
1: 0.9888287, 1.0039555, 0.9885619, 1.0057083, -0.0168796, 0.0153936
2: -0.0141504, 0.0024819, -0.0146147, 0.0033962, -0.0170138, 0.0165284
3: 0.0007359, 0.0059773, 0.0004887, 0.0060880, -0.0053521, 0.0054886
4: -0.0033131, 0.0096007, -0.0039083, 0.0099677, -0.0132808, 0.0135091
5: -0.0008122, 0.0111097, -0.0012143, 0.0113691, -0.0121813, 0.0123239
6: -0.0040328, 0.0038113, -0.0044029, 0.0041754, -0.0082082, 0.0082142
7: -0.0116117, -0.0022180, -0.0117985, -0.0015777, -0.0100340, 0.0095804
8: -0.0102891, 0.0168435, -0.0111841, 0.0174536, -0.0275738, 0.0278682
9: -0.0098691, 0.0056991, -0.0102079, 0.0062241, -0.0160932, 0.0159069

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0123925
time: 2.83 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0123931
time: 3.05 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0034671, 0.0030074, -0.0058187, 0.0038886, -0.0073558, 0.0088261
1: 0.9888785, 1.0038997, 0.9884241, 1.0085688, -0.0196902, 0.0154756
2: -0.0140637, 0.0024527, -0.0148545, 0.0048881, -0.0184728, 0.0167729
3: 0.0007438, 0.0059566, 0.0000854, 0.0061452, -0.0054014, 0.0058712
4: -0.0032266, 0.0095322, -0.0058045, 0.0101572, -0.0133838, 0.0153366
5: -0.0007994, 0.0110612, -0.0018704, 0.0115031, -0.0123024, 0.0129316
6: -0.0040027, 0.0037433, -0.0073318, 0.0043634, -0.0083661, 0.0110751
7: -0.0115768, -0.0022385, -0.0118950, -0.0005327, -0.0110441, 0.0096564
8: -0.0102605, 0.0167296, -0.0126446, 0.0177686, -0.0278659, 0.0292267
9: -0.0098058, 0.0056823, -0.0103828, 0.0070809, -0.0168867, 0.0160651

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0124136
time: 2.77 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0124144
time: 3.19 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0045201, 0.0032881, -0.0044170, 0.0031728, -0.0076928, 0.0077051
1: 0.9881709, 1.0059903, 0.9884616, 1.0057858, -0.0176150, 0.0175287
2: -0.0152950, 0.0035432, -0.0147891, 0.0034365, -0.0181257, 0.0177933
3: 0.0004490, 0.0062502, 0.0004778, 0.0061296, -0.0056806, 0.0057724
4: -0.0044551, 0.0105054, -0.0039596, 0.0101055, -0.0145606, 0.0144649
5: -0.0012789, 0.0117491, -0.0012320, 0.0114665, -0.0127454, 0.0129811
6: -0.0046915, 0.0047088, -0.0044820, 0.0043121, -0.0090036, 0.0091908
7: -0.0120721, -0.0014747, -0.0118686, -0.0015495, -0.0105227, 0.0103939
8: -0.0113280, 0.0183474, -0.0112235, 0.0176827, -0.0288483, 0.0293914
9: -0.0107042, 0.0063085, -0.0103351, 0.0062472, -0.0169515, 0.0166436

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124442, upper bound: 0.0123136
time: 2.50 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124442, upper bound: 0.0123135
time: 2.74 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0044950, 0.0032682, -0.0058596, 0.0039422, -0.0084372, 0.0091278
1: 0.9882210, 1.0059406, 0.9883237, 1.0086501, -0.0204291, 0.0176168
2: -0.0152076, 0.0035172, -0.0150291, 0.0049305, -0.0195887, 0.0180417
3: 0.0004560, 0.0062294, 0.0000740, 0.0061868, -0.0057308, 0.0061554
4: -0.0043679, 0.0104363, -0.0058583, 0.0102952, -0.0146631, 0.0162946
5: -0.0012675, 0.0117003, -0.0018890, 0.0116006, -0.0128681, 0.0135893
6: -0.0046405, 0.0046403, -0.0074150, 0.0045003, -0.0091408, 0.0120553
7: -0.0120370, -0.0014929, -0.0119651, -0.0005030, -0.0115339, 0.0104722
8: -0.0113025, 0.0182326, -0.0126860, 0.0179980, -0.0291446, 0.0307510
9: -0.0106404, 0.0062936, -0.0105102, 0.0071052, -0.0177457, 0.0168038

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124133, upper bound: 0.0123263
time: 2.53 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124136, upper bound: 0.0123262
time: 2.30 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.90 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0123925
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0123931
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0124136
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0124144
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 1, lower bound: -0.0124442, upper bound: 0.0123136
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 1, lower bound: -0.0124442, upper bound: 0.0123135
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 1, lower bound: -0.0124133, upper bound: 0.0123263
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 1, lower bound: -0.0124136, upper bound: 0.0123262

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0032417, 0.0028070, -0.0043781, 0.0031330, -0.0063747, 0.0071851
1: 0.9893840, 1.0034519, 0.9885619, 1.0057083, -0.0163243, 0.0148901
2: -0.0131847, 0.0022193, -0.0146147, 0.0033962, -0.0160355, 0.0161327
3: 0.0008069, 0.0057470, 0.0004887, 0.0060880, -0.0052811, 0.0052583
4: -0.0024127, 0.0088374, -0.0039083, 0.0099677, -0.0123803, 0.0127458
5: -0.0006967, 0.0105700, -0.0012143, 0.0113691, -0.0120658, 0.0117843
6: -0.0036976, 0.0030541, -0.0044029, 0.0041754, -0.0078730, 0.0074570
7: -0.0112231, -0.0024020, -0.0117985, -0.0015777, -0.0096454, 0.0093965
8: -0.0100320, 0.0155746, -0.0111841, 0.0174536, -0.0272845, 0.0265966
9: -0.0091644, 0.0055483, -0.0102079, 0.0062241, -0.0153885, 0.0157561

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123410
time: 2.62 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123926
time: 3.56 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0046388, 0.0028599, -0.0043781, 0.0031330, -0.0077718, 0.0072380
1: 0.9892505, 1.0062261, 0.9885619, 1.0057083, -0.0164578, 0.0176642
2: -0.0134166, 0.0036661, -0.0146147, 0.0033962, -0.0163036, 0.0176399
3: 0.0004158, 0.0058023, 0.0004887, 0.0060880, -0.0056722, 0.0053136
4: -0.0042515, 0.0090208, -0.0039083, 0.0099677, -0.0142192, 0.0129291
5: -0.0013330, 0.0106996, -0.0012143, 0.0113691, -0.0127020, 0.0119139
6: -0.0049329, 0.0032360, -0.0044029, 0.0041754, -0.0091083, 0.0076389
7: -0.0113164, -0.0013886, -0.0117985, -0.0015777, -0.0097387, 0.0104099
8: -0.0114483, 0.0158794, -0.0111841, 0.0174536, -0.0287153, 0.0269069
9: -0.0093337, 0.0063792, -0.0102079, 0.0062241, -0.0155578, 0.0165870

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123409
time: 6.12 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123935
time: 2.76 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0032417, 0.0028070, -0.0058187, 0.0038886, -0.0071304, 0.0086257
1: 0.9893840, 1.0034519, 0.9884241, 1.0085688, -0.0191848, 0.0150279
2: -0.0131847, 0.0022193, -0.0148545, 0.0048881, -0.0175795, 0.0164370
3: 0.0008069, 0.0057470, 0.0000854, 0.0061452, -0.0053383, 0.0056616
4: -0.0024127, 0.0088374, -0.0058045, 0.0101572, -0.0125699, 0.0146419
5: -0.0006967, 0.0105700, -0.0018704, 0.0115031, -0.0121998, 0.0124404
6: -0.0036976, 0.0030541, -0.0073318, 0.0043634, -0.0080610, 0.0103859
7: -0.0112231, -0.0024020, -0.0118950, -0.0005327, -0.0106904, 0.0094929
8: -0.0100320, 0.0155746, -0.0126446, 0.0177686, -0.0276101, 0.0280692
9: -0.0091644, 0.0055483, -0.0103828, 0.0070809, -0.0162453, 0.0159311

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0123559
time: 2.45 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0124145
time: 2.35 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0046388, 0.0028599, -0.0058187, 0.0038886, -0.0085274, 0.0086786
1: 0.9892505, 1.0062261, 0.9884241, 1.0085688, -0.0193182, 0.0178020
2: -0.0134166, 0.0036661, -0.0148545, 0.0048881, -0.0177464, 0.0177994
3: 0.0004158, 0.0058023, 0.0000854, 0.0061452, -0.0057294, 0.0057169
4: -0.0042515, 0.0090208, -0.0058045, 0.0101572, -0.0144087, 0.0148252
5: -0.0013330, 0.0106996, -0.0018704, 0.0115031, -0.0128360, 0.0125700
6: -0.0049329, 0.0032360, -0.0073318, 0.0043634, -0.0092963, 0.0105678
7: -0.0113164, -0.0013886, -0.0118950, -0.0005327, -0.0107837, 0.0105064
8: -0.0114483, 0.0158794, -0.0126446, 0.0177686, -0.0290090, 0.0283556
9: -0.0093337, 0.0063792, -0.0103828, 0.0070809, -0.0164146, 0.0167620

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0123409
time: 2.45 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0123926
time: 2.47 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0045201, 0.0032881, -0.0032417, 0.0028070, -0.0073270, 0.0065298
1: 0.9881709, 1.0059903, 0.9893840, 1.0034519, -0.0152811, 0.0166063
2: -0.0152950, 0.0035432, -0.0131847, 0.0022193, -0.0168407, 0.0161835
3: 0.0004490, 0.0062502, 0.0008069, 0.0057470, -0.0052980, 0.0054434
4: -0.0044551, 0.0105054, -0.0024127, 0.0088374, -0.0132925, 0.0129180
5: -0.0012789, 0.0117491, -0.0006967, 0.0105700, -0.0118490, 0.0124459
6: -0.0046915, 0.0047088, -0.0036976, 0.0030541, -0.0077456, 0.0084064
7: -0.0120721, -0.0014747, -0.0112231, -0.0024020, -0.0096701, 0.0097484
8: -0.0113280, 0.0183474, -0.0100320, 0.0155746, -0.0267401, 0.0281827
9: -0.0107042, 0.0063085, -0.0091644, 0.0055483, -0.0162525, 0.0154730

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124426, upper bound: 0.0123133
time: 2.16 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124426, upper bound: 0.0123138
time: 2.27 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0045201, 0.0032881, -0.0042693, 0.0030647, -0.0075848, 0.0075574
1: 0.9881709, 1.0059903, 0.9887339, 1.0054924, -0.0173216, 0.0172563
2: -0.0152950, 0.0035432, -0.0143153, 0.0032835, -0.0179735, 0.0172435
3: 0.0004490, 0.0062502, 0.0005192, 0.0060166, -0.0055676, 0.0057310
4: -0.0044551, 0.0105054, -0.0037652, 0.0097310, -0.0141861, 0.0142705
5: -0.0012789, 0.0117491, -0.0011647, 0.0112018, -0.0124807, 0.0129139
6: -0.0046915, 0.0047088, -0.0041817, 0.0039406, -0.0086321, 0.0088906
7: -0.0120721, -0.0014747, -0.0116780, -0.0016566, -0.0104155, 0.0102032
8: -0.0113280, 0.0183474, -0.0110738, 0.0170602, -0.0282070, 0.0292422
9: -0.0107042, 0.0063085, -0.0099894, 0.0061594, -0.0168636, 0.0162979

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124426, upper bound: 0.0123136
time: 3.13 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124426, upper bound: 0.0123131
time: 2.51 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0044950, 0.0032682, -0.0046416, 0.0028598, -0.0073548, 0.0079098
1: 0.9882210, 1.0059406, 0.9892505, 1.0062318, -0.0180108, 0.0166900
2: -0.0152076, 0.0035172, -0.0134167, 0.0036691, -0.0182671, 0.0164110
3: 0.0004560, 0.0062294, 0.0004149, 0.0058023, -0.0053463, 0.0058144
4: -0.0043679, 0.0104363, -0.0042552, 0.0090208, -0.0133887, 0.0146915
5: -0.0012675, 0.0117003, -0.0013343, 0.0106997, -0.0119672, 0.0130346
6: -0.0046405, 0.0046403, -0.0049387, 0.0032360, -0.0078765, 0.0095790
7: -0.0120370, -0.0014929, -0.0113164, -0.0013865, -0.0106504, 0.0098235
8: -0.0113025, 0.0182326, -0.0114512, 0.0158794, -0.0270224, 0.0295012
9: -0.0106404, 0.0062936, -0.0093337, 0.0063809, -0.0170213, 0.0156273

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123265
time: 2.36 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123134
time: 3.10 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0044950, 0.0032682, -0.0057217, 0.0037616, -0.0082566, 0.0089899
1: 0.9882210, 1.0059406, 0.9885953, 1.0083764, -0.0201554, 0.0173452
2: -0.0152076, 0.0035172, -0.0145566, 0.0047877, -0.0194474, 0.0174966
3: 0.0004560, 0.0062294, 0.0001126, 0.0060741, -0.0056181, 0.0061168
4: -0.0043679, 0.0104363, -0.0056768, 0.0099217, -0.0142896, 0.0161130
5: -0.0012675, 0.0117003, -0.0018262, 0.0113366, -0.0126041, 0.0135265
6: -0.0046405, 0.0046403, -0.0071345, 0.0041298, -0.0087703, 0.0117748
7: -0.0120370, -0.0014929, -0.0117750, -0.0006031, -0.0114339, 0.0102821
8: -0.0113025, 0.0182326, -0.0125462, 0.0173771, -0.0285055, 0.0306115
9: -0.0106404, 0.0062936, -0.0101654, 0.0070232, -0.0176637, 0.0164590

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123263
time: 2.80 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123136
time: 2.61 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 7.43 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123410
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123926
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123409
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123935
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0123559
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0124145
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0123409
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0123926
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0124426, upper bound: 0.0123133
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0124426, upper bound: 0.0123138
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0124426, upper bound: 0.0123136
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0124426, upper bound: 0.0123131
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123265
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123134
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123263
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123136

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0032417, 0.0028070, -0.0032417, 0.0028070, -0.0060487, 0.0060487
1: 0.9893840, 1.0034519, 0.9893840, 1.0034519, -0.0140679, 0.0140679
2: -0.0131847, 0.0022193, -0.0131847, 0.0022193, -0.0146976, 0.0146976
3: 0.0008069, 0.0057470, 0.0008069, 0.0057470, -0.0049402, 0.0049402
4: -0.0024127, 0.0088374, -0.0024127, 0.0088374, -0.0112501, 0.0112501
5: -0.0006967, 0.0105700, -0.0006967, 0.0105700, -0.0112668, 0.0112668
6: -0.0036976, 0.0030541, -0.0036976, 0.0030541, -0.0067517, 0.0067517
7: -0.0112231, -0.0024020, -0.0112231, -0.0024020, -0.0088211, 0.0088211
8: -0.0100320, 0.0155746, -0.0100320, 0.0155746, -0.0254055, 0.0254055
9: -0.0091644, 0.0055483, -0.0091644, 0.0055483, -0.0147127, 0.0147127

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122500, upper bound: 0.0123850
time: 3.40 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123457, upper bound: 0.0123926
time: 2.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032417, 0.0028070, -0.0042693, 0.0030647, -0.0063065, 0.0070763
1: 0.9893840, 1.0034519, 0.9887339, 1.0054924, -0.0161085, 0.0147180
2: -0.0131847, 0.0022193, -0.0143153, 0.0032835, -0.0159236, 0.0158472
3: 0.0008069, 0.0057470, 0.0005192, 0.0060166, -0.0052097, 0.0052278
4: -0.0024127, 0.0088374, -0.0037652, 0.0097310, -0.0121437, 0.0126026
5: -0.0006967, 0.0105700, -0.0011647, 0.0112018, -0.0118985, 0.0117348
6: -0.0036976, 0.0030541, -0.0041817, 0.0039406, -0.0076382, 0.0072358
7: -0.0112231, -0.0024020, -0.0116780, -0.0016566, -0.0095665, 0.0092760
8: -0.0100320, 0.0155746, -0.0110738, 0.0170602, -0.0268934, 0.0264866
9: -0.0091644, 0.0055483, -0.0099894, 0.0061594, -0.0153238, 0.0155376

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122500, upper bound: 0.0124253
time: 3.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123457, upper bound: 0.0124303
time: 3.07 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0046388, 0.0028599, -0.0032417, 0.0028070, -0.0074457, 0.0061016
1: 0.9892505, 1.0062261, 0.9893840, 1.0034519, -0.0142014, 0.0168421
2: -0.0134166, 0.0036661, -0.0131847, 0.0022193, -0.0149740, 0.0162048
3: 0.0004158, 0.0058023, 0.0008069, 0.0057470, -0.0053313, 0.0049955
4: -0.0042515, 0.0090208, -0.0024127, 0.0088374, -0.0130889, 0.0114334
5: -0.0013330, 0.0106996, -0.0006967, 0.0105700, -0.0119030, 0.0113964
6: -0.0049329, 0.0032360, -0.0036976, 0.0030541, -0.0079870, 0.0069336
7: -0.0113164, -0.0013886, -0.0112231, -0.0024020, -0.0089144, 0.0098345
8: -0.0114483, 0.0158794, -0.0100320, 0.0155746, -0.0268363, 0.0257165
9: -0.0093337, 0.0063792, -0.0091644, 0.0055483, -0.0148819, 0.0155436

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123250, upper bound: 0.0122351
time: 2.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123300, upper bound: 0.0123138
time: 2.37 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0046388, 0.0028599, -0.0042693, 0.0030647, -0.0077035, 0.0071292
1: 0.9892505, 1.0062261, 0.9887339, 1.0054924, -0.0162419, 0.0174921
2: -0.0134166, 0.0036661, -0.0143153, 0.0032835, -0.0161917, 0.0173543
3: 0.0004158, 0.0058023, 0.0005192, 0.0060166, -0.0056008, 0.0052832
4: -0.0042515, 0.0090208, -0.0037652, 0.0097310, -0.0139825, 0.0127859
5: -0.0013330, 0.0106996, -0.0011647, 0.0112018, -0.0125347, 0.0118644
6: -0.0049329, 0.0032360, -0.0041817, 0.0039406, -0.0088735, 0.0074177
7: -0.0113164, -0.0013886, -0.0116780, -0.0016566, -0.0096598, 0.0102894
8: -0.0114483, 0.0158794, -0.0110738, 0.0170602, -0.0283242, 0.0267970
9: -0.0093337, 0.0063792, -0.0099894, 0.0061594, -0.0154931, 0.0163685

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123250, upper bound: 0.0122920
time: 2.31 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123300, upper bound: 0.0123648
time: 2.69 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0032417, 0.0028070, -0.0046388, 0.0028599, -0.0061016, 0.0074457
1: 0.9893840, 1.0034519, 0.9892505, 1.0062261, -0.0168421, 0.0142014
2: -0.0131847, 0.0022193, -0.0134166, 0.0036661, -0.0162048, 0.0149740
3: 0.0008069, 0.0057470, 0.0004158, 0.0058023, -0.0049955, 0.0053313
4: -0.0024127, 0.0088374, -0.0042515, 0.0090208, -0.0114334, 0.0130889
5: -0.0006967, 0.0105700, -0.0013330, 0.0106996, -0.0113964, 0.0119030
6: -0.0036976, 0.0030541, -0.0049329, 0.0032360, -0.0069336, 0.0079870
7: -0.0112231, -0.0024020, -0.0113164, -0.0013886, -0.0098345, 0.0089144
8: -0.0100320, 0.0155746, -0.0114483, 0.0158794, -0.0257165, 0.0268363
9: -0.0091644, 0.0055483, -0.0093337, 0.0063792, -0.0155436, 0.0148819

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122003, upper bound: 0.0123684
time: 2.59 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122857, upper bound: 0.0123743
time: 2.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032417, 0.0028070, -0.0057187, 0.0037577, -0.0069995, 0.0085257
1: 0.9893840, 1.0034519, 0.9885953, 1.0083705, -0.0189865, 0.0148566
2: -0.0131847, 0.0022193, -0.0145566, 0.0047846, -0.0174773, 0.0161610
3: 0.0008069, 0.0057470, 0.0001134, 0.0060741, -0.0052673, 0.0056336
4: -0.0024127, 0.0088374, -0.0056728, 0.0099217, -0.0123344, 0.0145103
5: -0.0006967, 0.0105700, -0.0018248, 0.0113366, -0.0120333, 0.0123949
6: -0.0036976, 0.0030541, -0.0071285, 0.0041298, -0.0078274, 0.0101826
7: -0.0112231, -0.0024020, -0.0117751, -0.0006053, -0.0106179, 0.0093731
8: -0.0100320, 0.0155746, -0.0125432, 0.0173771, -0.0272217, 0.0279680
9: -0.0091644, 0.0055483, -0.0101654, 0.0070215, -0.0161859, 0.0157137

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122003, upper bound: 0.0124146
time: 4.10 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122857, upper bound: 0.0124203
time: 3.04 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0046388, 0.0028599, -0.0046388, 0.0028599, -0.0074986, 0.0074986
1: 0.9892505, 1.0062261, 0.9892505, 1.0062261, -0.0169755, 0.0169755
2: -0.0134166, 0.0036661, -0.0134166, 0.0036661, -0.0163445, 0.0163445
3: 0.0004158, 0.0058023, 0.0004158, 0.0058023, -0.0053866, 0.0053866
4: -0.0042515, 0.0090208, -0.0042515, 0.0090208, -0.0132723, 0.0132723
5: -0.0013330, 0.0106996, -0.0013330, 0.0106996, -0.0120326, 0.0120326
6: -0.0049329, 0.0032360, -0.0049329, 0.0032360, -0.0081689, 0.0081689
7: -0.0113164, -0.0013886, -0.0113164, -0.0013886, -0.0099278, 0.0099278
8: -0.0114483, 0.0158794, -0.0114483, 0.0158794, -0.0271161, 0.0271161
9: -0.0093337, 0.0063792, -0.0093337, 0.0063792, -0.0157129, 0.0157129

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122212, upper bound: 0.0123045
time: 2.89 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122975, upper bound: 0.0123130
time: 2.59 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0046388, 0.0028599, -0.0057187, 0.0037577, -0.0083965, 0.0085786
1: 0.9892505, 1.0062261, 0.9885953, 1.0083705, -0.0191200, 0.0176308
2: -0.0134166, 0.0036661, -0.0145566, 0.0047846, -0.0176441, 0.0175227
3: 0.0004158, 0.0058023, 0.0001134, 0.0060741, -0.0056584, 0.0056889
4: -0.0042515, 0.0090208, -0.0056728, 0.0099217, -0.0141732, 0.0146936
5: -0.0013330, 0.0106996, -0.0018248, 0.0113366, -0.0126695, 0.0125245
6: -0.0049329, 0.0032360, -0.0071285, 0.0041298, -0.0090627, 0.0103644
7: -0.0113164, -0.0013886, -0.0117751, -0.0006053, -0.0107112, 0.0103865
8: -0.0114483, 0.0158794, -0.0125432, 0.0173771, -0.0286205, 0.0282544
9: -0.0093337, 0.0063792, -0.0101654, 0.0070215, -0.0163551, 0.0165446

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122212, upper bound: 0.0123572
time: 2.95 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122975, upper bound: 0.0123648
time: 2.67 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042693, 0.0030647, -0.0032417, 0.0028070, -0.0070763, 0.0063065
1: 0.9887339, 1.0054924, 0.9893840, 1.0034519, -0.0147180, 0.0161085
2: -0.0143153, 0.0032835, -0.0131847, 0.0022193, -0.0158472, 0.0159236
3: 0.0005192, 0.0060166, 0.0008069, 0.0057470, -0.0052278, 0.0052097
4: -0.0037652, 0.0097310, -0.0024127, 0.0088374, -0.0126026, 0.0121437
5: -0.0011647, 0.0112018, -0.0006967, 0.0105700, -0.0117348, 0.0118985
6: -0.0041817, 0.0039406, -0.0036976, 0.0030541, -0.0072358, 0.0076382
7: -0.0116780, -0.0016566, -0.0112231, -0.0024020, -0.0092760, 0.0095665
8: -0.0110738, 0.0170602, -0.0100320, 0.0155746, -0.0264866, 0.0268934
9: -0.0099894, 0.0061594, -0.0091644, 0.0055483, -0.0155376, 0.0153238

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124149, upper bound: 0.0121998
time: 2.71 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124198, upper bound: 0.0122848
time: 2.89 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0057187, 0.0037577, -0.0032417, 0.0028070, -0.0085257, 0.0069995
1: 0.9885953, 1.0083705, 0.9893840, 1.0034519, -0.0148566, 0.0189865
2: -0.0145566, 0.0047846, -0.0131847, 0.0022193, -0.0161610, 0.0174773
3: 0.0001134, 0.0060741, 0.0008069, 0.0057470, -0.0056336, 0.0052673
4: -0.0056728, 0.0099217, -0.0024127, 0.0088374, -0.0145103, 0.0123344
5: -0.0018248, 0.0113366, -0.0006967, 0.0105700, -0.0123949, 0.0120333
6: -0.0071285, 0.0041298, -0.0036976, 0.0030541, -0.0101826, 0.0078274
7: -0.0117751, -0.0006053, -0.0112231, -0.0024020, -0.0093731, 0.0106179
8: -0.0125432, 0.0173771, -0.0100320, 0.0155746, -0.0279680, 0.0272217
9: -0.0101654, 0.0070215, -0.0091644, 0.0055483, -0.0157137, 0.0161859

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124149, upper bound: 0.0122003
time: 2.99 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124198, upper bound: 0.0122856
time: 2.74 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042693, 0.0030647, -0.0042693, 0.0030647, -0.0073340, 0.0073340
1: 0.9887339, 1.0054924, 0.9887339, 1.0054924, -0.0167585, 0.0167585
2: -0.0143153, 0.0032835, -0.0143153, 0.0032835, -0.0169838, 0.0169838
3: 0.0005192, 0.0060166, 0.0005192, 0.0060166, -0.0054974, 0.0054974
4: -0.0037652, 0.0097310, -0.0037652, 0.0097310, -0.0134962, 0.0134962
5: -0.0011647, 0.0112018, -0.0011647, 0.0112018, -0.0123665, 0.0123665
6: -0.0041817, 0.0039406, -0.0041817, 0.0039406, -0.0081223, 0.0081223
7: -0.0116780, -0.0016566, -0.0116780, -0.0016566, -0.0100213, 0.0100213
8: -0.0110738, 0.0170602, -0.0110738, 0.0170602, -0.0279535, 0.0279535
9: -0.0099894, 0.0061594, -0.0099894, 0.0061594, -0.0161488, 0.0161488

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123369, upper bound: 0.0122783
time: 2.85 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124143, upper bound: 0.0122856
time: 2.79 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0057187, 0.0037577, -0.0042693, 0.0030647, -0.0087834, 0.0080270
1: 0.9885953, 1.0083705, 0.9887339, 1.0054924, -0.0168971, 0.0196366
2: -0.0145566, 0.0047846, -0.0143153, 0.0032835, -0.0172800, 0.0185393
3: 0.0001134, 0.0060741, 0.0005192, 0.0060166, -0.0059032, 0.0055549
4: -0.0056728, 0.0099217, -0.0037652, 0.0097310, -0.0154039, 0.0136869
5: -0.0018248, 0.0113366, -0.0011647, 0.0112018, -0.0130266, 0.0125013
6: -0.0071285, 0.0041298, -0.0041817, 0.0039406, -0.0110691, 0.0083115
7: -0.0117751, -0.0006053, -0.0116780, -0.0016566, -0.0101184, 0.0110727
8: -0.0125432, 0.0173771, -0.0110738, 0.0170602, -0.0294349, 0.0282798
9: -0.0101654, 0.0070215, -0.0099894, 0.0061594, -0.0163248, 0.0170108

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124081, upper bound: 0.0121998
time: 2.81 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124143, upper bound: 0.0122853
time: 2.56 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042693, 0.0030647, -0.0046416, 0.0028598, -0.0071292, 0.0077064
1: 0.9887339, 1.0054924, 0.9892505, 1.0062318, -0.0174978, 0.0162419
2: -0.0143153, 0.0032835, -0.0134167, 0.0036691, -0.0173573, 0.0161917
3: 0.0005192, 0.0060166, 0.0004149, 0.0058023, -0.0052832, 0.0056017
4: -0.0037652, 0.0097310, -0.0042552, 0.0090208, -0.0127859, 0.0139863
5: -0.0011647, 0.0112018, -0.0013343, 0.0106997, -0.0118644, 0.0125361
6: -0.0041817, 0.0039406, -0.0049387, 0.0032360, -0.0074177, 0.0088793
7: -0.0116780, -0.0016566, -0.0113164, -0.0013865, -0.0102914, 0.0096598
8: -0.0110738, 0.0170602, -0.0114512, 0.0158794, -0.0267970, 0.0283271
9: -0.0099894, 0.0061594, -0.0093337, 0.0063809, -0.0163702, 0.0154931

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122924, upper bound: 0.0122911
time: 2.71 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123652, upper bound: 0.0122969
time: 2.64 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0057187, 0.0037577, -0.0046416, 0.0028598, -0.0085786, 0.0083994
1: 0.9885953, 1.0083705, 0.9892505, 1.0062318, -0.0176365, 0.0191200
2: -0.0145566, 0.0047846, -0.0134167, 0.0036691, -0.0175257, 0.0176441
3: 0.0001134, 0.0060741, 0.0004149, 0.0058023, -0.0056889, 0.0056592
4: -0.0056728, 0.0099217, -0.0042552, 0.0090208, -0.0146936, 0.0141769
5: -0.0018248, 0.0113366, -0.0013343, 0.0106997, -0.0125245, 0.0126709
6: -0.0071285, 0.0041298, -0.0049387, 0.0032360, -0.0103644, 0.0090685
7: -0.0117751, -0.0006053, -0.0113164, -0.0013865, -0.0103885, 0.0107112
8: -0.0125432, 0.0173771, -0.0114512, 0.0158794, -0.0282544, 0.0286235
9: -0.0101654, 0.0070215, -0.0093337, 0.0063809, -0.0165463, 0.0163551

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123572, upper bound: 0.0122003
time: 2.43 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123652, upper bound: 0.0122852
time: 2.92 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042693, 0.0030647, -0.0057217, 0.0037616, -0.0080309, 0.0087864
1: 0.9887339, 1.0054924, 0.9885953, 1.0083764, -0.0196424, 0.0168971
2: -0.0143153, 0.0032835, -0.0145566, 0.0047877, -0.0185424, 0.0172800
3: 0.0005192, 0.0060166, 0.0001126, 0.0060741, -0.0055549, 0.0059040
4: -0.0037652, 0.0097310, -0.0056768, 0.0099217, -0.0136869, 0.0154078
5: -0.0011647, 0.0112018, -0.0018262, 0.0113366, -0.0125013, 0.0130280
6: -0.0041817, 0.0039406, -0.0071345, 0.0041298, -0.0083115, 0.0110751
7: -0.0116780, -0.0016566, -0.0117750, -0.0006031, -0.0110749, 0.0101184
8: -0.0110738, 0.0170602, -0.0125462, 0.0173771, -0.0282798, 0.0294379
9: -0.0099894, 0.0061594, -0.0101654, 0.0070232, -0.0170126, 0.0163248

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122894, upper bound: 0.0122906
time: 3.04 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123639, upper bound: 0.0122974
time: 2.41 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0057187, 0.0037577, -0.0057217, 0.0037616, -0.0094803, 0.0094794
1: 0.9885953, 1.0083705, 0.9885953, 1.0083764, -0.0197811, 0.0197752
2: -0.0145566, 0.0047846, -0.0145566, 0.0047877, -0.0187346, 0.0187315
3: 0.0001134, 0.0060741, 0.0001126, 0.0060741, -0.0059607, 0.0059615
4: -0.0056728, 0.0099217, -0.0056768, 0.0099217, -0.0155945, 0.0155985
5: -0.0018248, 0.0113366, -0.0018262, 0.0113366, -0.0131614, 0.0131628
6: -0.0071285, 0.0041298, -0.0071345, 0.0041298, -0.0112582, 0.0112643
7: -0.0117751, -0.0006053, -0.0117750, -0.0006031, -0.0111720, 0.0111698
8: -0.0125432, 0.0173771, -0.0125462, 0.0173771, -0.0297375, 0.0297405
9: -0.0101654, 0.0070215, -0.0101654, 0.0070232, -0.0171886, 0.0171868

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122894, upper bound: 0.0122783
time: 2.61 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123639, upper bound: 0.0122850
time: 2.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.31 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0122500, upper bound: 0.0123850
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0123457, upper bound: 0.0123926
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0122500, upper bound: 0.0124253
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0123457, upper bound: 0.0124303
NS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0123250, upper bound: 0.0122351
NS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0123300, upper bound: 0.0123138
NS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0123250, upper bound: 0.0122920
NS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0123300, upper bound: 0.0123648
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0122003, upper bound: 0.0123684
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0122857, upper bound: 0.0123743
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0122003, upper bound: 0.0124146
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0122857, upper bound: 0.0124203
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0122212, upper bound: 0.0123045
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0122975, upper bound: 0.0123130
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0122212, upper bound: 0.0123572
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0122975, upper bound: 0.0123648
NS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0124149, upper bound: 0.0121998
NS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0124198, upper bound: 0.0122848
NS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0124149, upper bound: 0.0122003
NS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0124198, upper bound: 0.0122856
NS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0123369, upper bound: 0.0122783
NS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0124143, upper bound: 0.0122856
NS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0124081, upper bound: 0.0121998
NS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0124143, upper bound: 0.0122853
NS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0122924, upper bound: 0.0122911
NS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0123652, upper bound: 0.0122969
NS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0123572, upper bound: 0.0122003
NS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0123652, upper bound: 0.0122852
NS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0122894, upper bound: 0.0122906
NS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0123639, upper bound: 0.0122974
NS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0122894, upper bound: 0.0122783
NS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0123639, upper bound: 0.0122850

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0030213, 0.0025732, -0.0031722, 0.0027732, -0.0057944, 0.0057454
1: 0.9899734, 1.0030142, 0.9894690, 1.0033140, -0.0133406, 0.0135452
2: -0.0121592, 0.0019909, -0.0130364, 0.0021473, -0.0135944, 0.0143167
3: 0.0008686, 0.0055025, 0.0008263, 0.0057117, -0.0048431, 0.0046762
4: -0.0021225, 0.0080270, -0.0023212, 0.0087202, -0.0108427, 0.0103481
5: -0.0005963, 0.0099971, -0.0006650, 0.0104872, -0.0110834, 0.0106621
6: -0.0033417, 0.0022500, -0.0036461, 0.0029378, -0.0062794, 0.0058961
7: -0.0108105, -0.0025619, -0.0111635, -0.0024524, -0.0083581, 0.0086015
8: -0.0098084, 0.0142273, -0.0099615, 0.0153798, -0.0249841, 0.0239844
9: -0.0084162, 0.0054171, -0.0090562, 0.0055069, -0.0139231, 0.0144733

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122029, upper bound: 0.0122240
time: 3.14 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122383, upper bound: 0.0123202
time: 3.86 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0031307, 0.0027583, -0.0032417, 0.0028070, -0.0059376, 0.0060001
1: 0.9895065, 1.0032316, 0.9893840, 1.0034519, -0.0139455, 0.0138476
2: -0.0129715, 0.0021043, -0.0131847, 0.0022193, -0.0144559, 0.0145843
3: 0.0008379, 0.0056962, 0.0008069, 0.0057470, -0.0049091, 0.0048893
4: -0.0022665, 0.0086689, -0.0024127, 0.0088374, -0.0111039, 0.0110816
5: -0.0006461, 0.0104509, -0.0006967, 0.0105700, -0.0112162, 0.0111476
6: -0.0036236, 0.0028869, -0.0036976, 0.0030541, -0.0066777, 0.0065845
7: -0.0111373, -0.0024826, -0.0112231, -0.0024020, -0.0087353, 0.0087406
8: -0.0099194, 0.0152945, -0.0100320, 0.0155746, -0.0252932, 0.0251183
9: -0.0090089, 0.0054822, -0.0091644, 0.0055483, -0.0145571, 0.0146466

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123026, upper bound: 0.0122418
time: 2.61 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123265, upper bound: 0.0123264
time: 2.43 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0030213, 0.0025732, -0.0041942, 0.0030308, -0.0060521, 0.0067673
1: 0.9899734, 1.0030142, 0.9888196, 1.0053432, -0.0153698, 0.0141946
2: -0.0121592, 0.0019909, -0.0141664, 0.0032057, -0.0146848, 0.0154676
3: 0.0008686, 0.0055025, 0.0005402, 0.0059811, -0.0051125, 0.0049623
4: -0.0021225, 0.0080270, -0.0036662, 0.0096134, -0.0117358, 0.0116932
5: -0.0005963, 0.0099971, -0.0011305, 0.0111186, -0.0117149, 0.0111275
6: -0.0033417, 0.0022500, -0.0040384, 0.0038239, -0.0071655, 0.0062883
7: -0.0108105, -0.0025619, -0.0116181, -0.0017111, -0.0090994, 0.0090562
8: -0.0098084, 0.0142273, -0.0109976, 0.0168645, -0.0264731, 0.0250274
9: -0.0084162, 0.0054171, -0.0098807, 0.0061147, -0.0145309, 0.0152979

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121511, upper bound: 0.0122611
time: 2.94 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121860, upper bound: 0.0123621
time: 2.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0031307, 0.0027583, -0.0042693, 0.0030647, -0.0061954, 0.0070277
1: 0.9895065, 1.0032316, 0.9887339, 1.0054924, -0.0159860, 0.0144977
2: -0.0129715, 0.0021043, -0.0143153, 0.0032835, -0.0157123, 0.0157339
3: 0.0008379, 0.0056962, 0.0005192, 0.0060166, -0.0051787, 0.0051770
4: -0.0022665, 0.0086689, -0.0037652, 0.0097310, -0.0119975, 0.0124341
5: -0.0006461, 0.0104509, -0.0011647, 0.0112018, -0.0118479, 0.0116156
6: -0.0036236, 0.0028869, -0.0041817, 0.0039406, -0.0075642, 0.0070686
7: -0.0111373, -0.0024826, -0.0116780, -0.0016566, -0.0094807, 0.0091954
8: -0.0099194, 0.0152945, -0.0110738, 0.0170602, -0.0267811, 0.0262072
9: -0.0090089, 0.0054822, -0.0099894, 0.0061594, -0.0151683, 0.0154716

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121910, upper bound: 0.0123363
time: 2.52 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122845, upper bound: 0.0123674
time: 2.38 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0045712, 0.0028268, -0.0030213, 0.0025732, -0.0071443, 0.0058480
1: 0.9893340, 1.0060917, 0.9899734, 1.0030142, -0.0136802, 0.0161183
2: -0.0132715, 0.0035961, -0.0121592, 0.0019909, -0.0145973, 0.0151023
3: 0.0004347, 0.0057677, 0.0008686, 0.0055025, -0.0050678, 0.0048991
4: -0.0041624, 0.0089061, -0.0021225, 0.0080270, -0.0121894, 0.0110286
5: -0.0013022, 0.0106186, -0.0005963, 0.0099971, -0.0112992, 0.0112148
6: -0.0047954, 0.0031222, -0.0033417, 0.0022500, -0.0070454, 0.0064638
7: -0.0112581, -0.0014377, -0.0108105, -0.0025619, -0.0086962, 0.0093729
8: -0.0113798, 0.0156888, -0.0098084, 0.0142273, -0.0254169, 0.0253004
9: -0.0092278, 0.0063389, -0.0084162, 0.0054171, -0.0146449, 0.0147551

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122812, upper bound: 0.0120886
time: 3.02 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123048, upper bound: 0.0121692
time: 2.65 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0046388, 0.0028599, -0.0031307, 0.0027583, -0.0073971, 0.0059905
1: 0.9892505, 1.0062261, 0.9895065, 1.0032316, -0.0139811, 0.0167196
2: -0.0134166, 0.0036661, -0.0129715, 0.0021043, -0.0148607, 0.0159654
3: 0.0004158, 0.0058023, 0.0008379, 0.0056962, -0.0052804, 0.0049644
4: -0.0042515, 0.0090208, -0.0022665, 0.0086689, -0.0129204, 0.0112873
5: -0.0013330, 0.0106996, -0.0006461, 0.0104509, -0.0117839, 0.0113458
6: -0.0049329, 0.0032360, -0.0036236, 0.0028869, -0.0078198, 0.0068596
7: -0.0113164, -0.0013886, -0.0111373, -0.0024826, -0.0088339, 0.0097487
8: -0.0114483, 0.0158794, -0.0099194, 0.0152945, -0.0265489, 0.0256042
9: -0.0093337, 0.0063792, -0.0090089, 0.0054822, -0.0148159, 0.0153880

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122892, upper bound: 0.0121758
time: 2.47 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123096, upper bound: 0.0122475
time: 2.33 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0045712, 0.0028268, -0.0040592, 0.0028335, -0.0074047, 0.0068860
1: 0.9893340, 1.0060917, 0.9893169, 1.0050751, -0.0157411, 0.0167748
2: -0.0132715, 0.0035961, -0.0133013, 0.0030659, -0.0157136, 0.0162689
3: 0.0004347, 0.0057677, 0.0005780, 0.0057748, -0.0053401, 0.0051897
4: -0.0041624, 0.0089061, -0.0034886, 0.0089296, -0.0130920, 0.0123947
5: -0.0013022, 0.0106186, -0.0010690, 0.0106352, -0.0119374, 0.0116876
6: -0.0047954, 0.0031222, -0.0037546, 0.0031455, -0.0079409, 0.0068768
7: -0.0112581, -0.0014377, -0.0112700, -0.0018090, -0.0094491, 0.0098323
8: -0.0113798, 0.0156888, -0.0108608, 0.0157278, -0.0269212, 0.0263615
9: -0.0092278, 0.0063389, -0.0092495, 0.0060345, -0.0152623, 0.0155884

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122341, upper bound: 0.0121384
time: 2.82 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122628, upper bound: 0.0122271
time: 2.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0046388, 0.0028599, -0.0041496, 0.0030145, -0.0076533, 0.0070094
1: 0.9892505, 1.0062261, 0.9888607, 1.0052546, -0.0160041, 0.0173653
2: -0.0134166, 0.0036661, -0.0140949, 0.0031595, -0.0159478, 0.0171103
3: 0.0004158, 0.0058023, 0.0005527, 0.0059641, -0.0055483, 0.0052496
4: -0.0042515, 0.0090208, -0.0036075, 0.0095569, -0.0138084, 0.0126283
5: -0.0013330, 0.0106996, -0.0011102, 0.0110787, -0.0124116, 0.0118098
6: -0.0049329, 0.0032360, -0.0040136, 0.0037678, -0.0087007, 0.0072495
7: -0.0113164, -0.0013886, -0.0115893, -0.0017435, -0.0095730, 0.0102007
8: -0.0114483, 0.0158794, -0.0109523, 0.0167707, -0.0280283, 0.0266440
9: -0.0093337, 0.0063792, -0.0098286, 0.0060882, -0.0154219, 0.0162078

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122410, upper bound: 0.0122266
time: 3.13 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122676, upper bound: 0.0122994
time: 2.69 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0030213, 0.0025732, -0.0045712, 0.0028268, -0.0058480, 0.0071443
1: 0.9899734, 1.0030142, 0.9893340, 1.0060917, -0.0161183, 0.0136802
2: -0.0121592, 0.0019909, -0.0132715, 0.0035961, -0.0151023, 0.0145973
3: 0.0008686, 0.0055025, 0.0004347, 0.0057677, -0.0048991, 0.0050678
4: -0.0021225, 0.0080270, -0.0041624, 0.0089061, -0.0110286, 0.0121894
5: -0.0005963, 0.0099971, -0.0013022, 0.0106186, -0.0112148, 0.0112992
6: -0.0033417, 0.0022500, -0.0047954, 0.0031222, -0.0064638, 0.0070454
7: -0.0108105, -0.0025619, -0.0112581, -0.0014377, -0.0093729, 0.0086962
8: -0.0098084, 0.0142273, -0.0113798, 0.0156888, -0.0253004, 0.0254169
9: -0.0084162, 0.0054171, -0.0092278, 0.0063389, -0.0147551, 0.0146449

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0120890, upper bound: 0.0122803
time: 2.80 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121692, upper bound: 0.0123047
time: 2.87 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0031307, 0.0027583, -0.0046388, 0.0028599, -0.0059905, 0.0073971
1: 0.9895065, 1.0032316, 0.9892505, 1.0062261, -0.0167196, 0.0139811
2: -0.0129715, 0.0021043, -0.0134166, 0.0036661, -0.0159654, 0.0148607
3: 0.0008379, 0.0056962, 0.0004158, 0.0058023, -0.0049644, 0.0052804
4: -0.0022665, 0.0086689, -0.0042515, 0.0090208, -0.0112873, 0.0129204
5: -0.0006461, 0.0104509, -0.0013330, 0.0106996, -0.0113458, 0.0117839
6: -0.0036236, 0.0028869, -0.0049329, 0.0032360, -0.0068596, 0.0078198
7: -0.0111373, -0.0024826, -0.0113164, -0.0013886, -0.0097487, 0.0088339
8: -0.0099194, 0.0152945, -0.0114483, 0.0158794, -0.0256042, 0.0265489
9: -0.0090089, 0.0054822, -0.0093337, 0.0063792, -0.0153880, 0.0148159

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121759, upper bound: 0.0122890
time: 2.44 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122472, upper bound: 0.0123096
time: 2.52 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0030213, 0.0025732, -0.0056266, 0.0036371, -0.0066584, 0.0081997
1: 0.9899734, 1.0030142, 0.9886786, 1.0081874, -0.0182140, 0.0143356
2: -0.0121592, 0.0019909, -0.0144112, 0.0046891, -0.0162338, 0.0157839
3: 0.0008686, 0.0055025, 0.0001392, 0.0060395, -0.0051709, 0.0053633
4: -0.0021225, 0.0080270, -0.0055516, 0.0098069, -0.0119293, 0.0135785
5: -0.0005963, 0.0099971, -0.0017829, 0.0112554, -0.0118517, 0.0117800
6: -0.0033417, 0.0022500, -0.0069412, 0.0040158, -0.0073575, 0.0091911
7: -0.0108105, -0.0025619, -0.0117166, -0.0006721, -0.0101384, 0.0091546
8: -0.0098084, 0.0142273, -0.0124498, 0.0171862, -0.0268055, 0.0264926
9: -0.0084162, 0.0054171, -0.0100594, 0.0069667, -0.0153829, 0.0154765

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0120559, upper bound: 0.0123224
time: 3.39 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121351, upper bound: 0.0123523
time: 3.40 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0031307, 0.0027583, -0.0057187, 0.0037577, -0.0068884, 0.0084771
1: 0.9895065, 1.0032316, 0.9885953, 1.0083705, -0.0188640, 0.0146363
2: -0.0129715, 0.0021043, -0.0145566, 0.0047846, -0.0172659, 0.0160477
3: 0.0008379, 0.0056962, 0.0001134, 0.0060741, -0.0052362, 0.0055827
4: -0.0022665, 0.0086689, -0.0056728, 0.0099217, -0.0121882, 0.0143418
5: -0.0006461, 0.0104509, -0.0018248, 0.0113366, -0.0119827, 0.0122757
6: -0.0036236, 0.0028869, -0.0071285, 0.0041298, -0.0077534, 0.0100154
7: -0.0111373, -0.0024826, -0.0117751, -0.0006053, -0.0105321, 0.0092925
8: -0.0099194, 0.0152945, -0.0125432, 0.0173771, -0.0271094, 0.0276886
9: -0.0090089, 0.0054822, -0.0101654, 0.0070215, -0.0160303, 0.0156476

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121427, upper bound: 0.0123286
time: 2.88 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122201, upper bound: 0.0123564
time: 2.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0044253, 0.0026286, -0.0045712, 0.0028268, -0.0072521, 0.0071997
1: 0.9898338, 1.0058022, 0.9893340, 1.0060917, -0.0162579, 0.0164682
2: -0.0124021, 0.0034450, -0.0132715, 0.0035961, -0.0152624, 0.0159817
3: 0.0004755, 0.0055604, 0.0004347, 0.0057677, -0.0052922, 0.0051257
4: -0.0039704, 0.0082190, -0.0041624, 0.0089061, -0.0128766, 0.0123814
5: -0.0012358, 0.0101328, -0.0013022, 0.0106186, -0.0118543, 0.0114350
6: -0.0044988, 0.0024405, -0.0047954, 0.0031222, -0.0076210, 0.0072358
7: -0.0109083, -0.0015435, -0.0112581, -0.0014377, -0.0094706, 0.0097146
8: -0.0112319, 0.0145465, -0.0113798, 0.0156888, -0.0267050, 0.0257151
9: -0.0085935, 0.0062522, -0.0092278, 0.0063389, -0.0149324, 0.0154800

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A2_B1_A1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121518, upper bound: 0.0121574
time: 2.54 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121851, upper bound: 0.0122386
time: 2.68 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0045276, 0.0028078, -0.0046388, 0.0028599, -0.0073875, 0.0074466
1: 0.9893818, 1.0060053, 0.9892505, 1.0062261, -0.0168443, 0.0167547
2: -0.0131884, 0.0035510, -0.0134166, 0.0036661, -0.0160942, 0.0162303
3: 0.0004469, 0.0057479, 0.0004158, 0.0058023, -0.0053555, 0.0053321
4: -0.0041051, 0.0088404, -0.0042515, 0.0090208, -0.0131259, 0.0130919
5: -0.0012824, 0.0105721, -0.0013330, 0.0106996, -0.0119820, 0.0119051
6: -0.0047069, 0.0030570, -0.0049329, 0.0032360, -0.0079429, 0.0079899
7: -0.0112246, -0.0014692, -0.0113164, -0.0013886, -0.0098360, 0.0098472
8: -0.0113357, 0.0155796, -0.0114483, 0.0158794, -0.0270034, 0.0268083
9: -0.0091672, 0.0063130, -0.0093337, 0.0063792, -0.0155463, 0.0156467

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122418, upper bound: 0.0121759
time: 2.20 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122604, upper bound: 0.0122471
time: 2.76 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0044253, 0.0026286, -0.0056266, 0.0036371, -0.0080624, 0.0082551
1: 0.9898338, 1.0058022, 0.9886786, 1.0081874, -0.0183536, 0.0171235
2: -0.0124021, 0.0034450, -0.0144112, 0.0046891, -0.0163924, 0.0171595
3: 0.0004755, 0.0055604, 0.0001392, 0.0060395, -0.0055640, 0.0054212
4: -0.0039704, 0.0082190, -0.0055516, 0.0098069, -0.0137773, 0.0137705
5: -0.0012358, 0.0101328, -0.0017829, 0.0112554, -0.0124911, 0.0119157
6: -0.0044988, 0.0024405, -0.0069412, 0.0040158, -0.0085147, 0.0093816
7: -0.0109083, -0.0015435, -0.0117166, -0.0006721, -0.0102362, 0.0101731
8: -0.0112319, 0.0145465, -0.0124498, 0.0171862, -0.0282096, 0.0267900
9: -0.0085935, 0.0062522, -0.0100594, 0.0069667, -0.0155601, 0.0163115

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A2_B2_A1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121231, upper bound: 0.0122065
time: 2.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121564, upper bound: 0.0122917
time: 3.75 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0045276, 0.0028078, -0.0057187, 0.0037577, -0.0082853, 0.0085266
1: 0.9893818, 1.0060053, 0.9885953, 1.0083705, -0.0189887, 0.0174100
2: -0.0131884, 0.0035510, -0.0145566, 0.0047846, -0.0174184, 0.0174085
3: 0.0004469, 0.0057479, 0.0001134, 0.0060741, -0.0056273, 0.0056345
4: -0.0041051, 0.0088404, -0.0056728, 0.0099217, -0.0140268, 0.0145133
5: -0.0012824, 0.0105721, -0.0018248, 0.0113366, -0.0126189, 0.0123970
6: -0.0047069, 0.0030570, -0.0071285, 0.0041298, -0.0088367, 0.0101855
7: -0.0112246, -0.0014692, -0.0117751, -0.0006053, -0.0106194, 0.0103058
8: -0.0113357, 0.0155796, -0.0125432, 0.0173771, -0.0285079, 0.0279528
9: -0.0091672, 0.0063130, -0.0101654, 0.0070215, -0.0161886, 0.0164784

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121524, upper bound: 0.0122723
time: 2.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122320, upper bound: 0.0122985
time: 3.09 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0041942, 0.0030308, -0.0030213, 0.0025732, -0.0067673, 0.0060521
1: 0.9888196, 1.0053432, 0.9899734, 1.0030142, -0.0141946, 0.0153698
2: -0.0141664, 0.0032057, -0.0121592, 0.0019909, -0.0154676, 0.0146848
3: 0.0005402, 0.0059811, 0.0008686, 0.0055025, -0.0049623, 0.0051125
4: -0.0036662, 0.0096134, -0.0021225, 0.0080270, -0.0116932, 0.0117358
5: -0.0011305, 0.0111186, -0.0005963, 0.0099971, -0.0111275, 0.0117149
6: -0.0040384, 0.0038239, -0.0033417, 0.0022500, -0.0062883, 0.0071655
7: -0.0116181, -0.0017111, -0.0108105, -0.0025619, -0.0090562, 0.0090994
8: -0.0109976, 0.0168645, -0.0098084, 0.0142273, -0.0250274, 0.0264731
9: -0.0098807, 0.0061147, -0.0084162, 0.0054171, -0.0152979, 0.0145309

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122612, upper bound: 0.0121506
time: 2.61 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123625, upper bound: 0.0121858
time: 2.61 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042693, 0.0030647, -0.0031307, 0.0027583, -0.0070277, 0.0061954
1: 0.9887339, 1.0054924, 0.9895065, 1.0032316, -0.0144977, 0.0159860
2: -0.0143153, 0.0032835, -0.0129715, 0.0021043, -0.0157339, 0.0157123
3: 0.0005192, 0.0060166, 0.0008379, 0.0056962, -0.0051770, 0.0051787
4: -0.0037652, 0.0097310, -0.0022665, 0.0086689, -0.0124341, 0.0119975
5: -0.0011647, 0.0112018, -0.0006461, 0.0104509, -0.0116156, 0.0118479
6: -0.0041817, 0.0039406, -0.0036236, 0.0028869, -0.0070686, 0.0075642
7: -0.0116780, -0.0016566, -0.0111373, -0.0024826, -0.0091954, 0.0094807
8: -0.0110738, 0.0170602, -0.0099194, 0.0152945, -0.0262072, 0.0267811
9: -0.0099894, 0.0061594, -0.0090089, 0.0054822, -0.0154716, 0.0151683

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123363, upper bound: 0.0121909
time: 2.62 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123675, upper bound: 0.0122845
time: 3.01 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0056266, 0.0036371, -0.0030213, 0.0025732, -0.0081997, 0.0066584
1: 0.9886786, 1.0081874, 0.9899734, 1.0030142, -0.0143356, 0.0182140
2: -0.0144112, 0.0046891, -0.0121592, 0.0019909, -0.0157839, 0.0162338
3: 0.0001392, 0.0060395, 0.0008686, 0.0055025, -0.0053633, 0.0051709
4: -0.0055516, 0.0098069, -0.0021225, 0.0080270, -0.0135785, 0.0119293
5: -0.0017829, 0.0112554, -0.0005963, 0.0099971, -0.0117800, 0.0118517
6: -0.0069412, 0.0040158, -0.0033417, 0.0022500, -0.0091911, 0.0073575
7: -0.0117166, -0.0006721, -0.0108105, -0.0025619, -0.0091546, 0.0101384
8: -0.0124498, 0.0171862, -0.0098084, 0.0142273, -0.0264926, 0.0268055
9: -0.0100594, 0.0069667, -0.0084162, 0.0054171, -0.0154765, 0.0153829

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123225, upper bound: 0.0120554
time: 3.43 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123520, upper bound: 0.0121351
time: 2.44 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0057187, 0.0037577, -0.0031307, 0.0027583, -0.0084771, 0.0068884
1: 0.9885953, 1.0083705, 0.9895065, 1.0032316, -0.0146363, 0.0188640
2: -0.0145566, 0.0047846, -0.0129715, 0.0021043, -0.0160477, 0.0172659
3: 0.0001134, 0.0060741, 0.0008379, 0.0056962, -0.0055827, 0.0052362
4: -0.0056728, 0.0099217, -0.0022665, 0.0086689, -0.0143418, 0.0121882
5: -0.0018248, 0.0113366, -0.0006461, 0.0104509, -0.0122757, 0.0119827
6: -0.0071285, 0.0041298, -0.0036236, 0.0028869, -0.0100154, 0.0077534
7: -0.0117751, -0.0006053, -0.0111373, -0.0024826, -0.0092925, 0.0105321
8: -0.0125432, 0.0173771, -0.0099194, 0.0152945, -0.0276886, 0.0271094
9: -0.0101654, 0.0070215, -0.0090089, 0.0054822, -0.0156476, 0.0160303

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123290, upper bound: 0.0121419
time: 3.04 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123565, upper bound: 0.0122198
time: 2.55 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0040592, 0.0028335, -0.0041942, 0.0030308, -0.0070900, 0.0070277
1: 0.9893169, 1.0050751, 0.9888196, 1.0053432, -0.0160263, 0.0162555
2: -0.0133013, 0.0030659, -0.0141664, 0.0032057, -0.0157387, 0.0164710
3: 0.0005780, 0.0057748, 0.0005402, 0.0059811, -0.0054031, 0.0052346
4: -0.0034886, 0.0089296, -0.0036662, 0.0096134, -0.0131020, 0.0125958
5: -0.0010690, 0.0106352, -0.0011305, 0.0111186, -0.0121876, 0.0117657
6: -0.0037546, 0.0031455, -0.0040384, 0.0038239, -0.0075785, 0.0071838
7: -0.0112700, -0.0018090, -0.0116181, -0.0017111, -0.0095589, 0.0098091
8: -0.0108608, 0.0157278, -0.0109976, 0.0168645, -0.0275074, 0.0265047
9: -0.0092495, 0.0060345, -0.0098807, 0.0061147, -0.0153642, 0.0159152

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_B2_A1_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122380, upper bound: 0.0121741
time: 3.16 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122839, upper bound: 0.0122783
time: 2.41 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0041496, 0.0030145, -0.0042693, 0.0030647, -0.0072143, 0.0072838
1: 0.9888607, 1.0052546, 0.9887339, 1.0054924, -0.0166317, 0.0165207
2: -0.0140949, 0.0031595, -0.0143153, 0.0032835, -0.0167662, 0.0167119
3: 0.0005527, 0.0059641, 0.0005192, 0.0060166, -0.0054639, 0.0054449
4: -0.0036075, 0.0095569, -0.0037652, 0.0097310, -0.0133386, 0.0133221
5: -0.0011102, 0.0110787, -0.0011647, 0.0112018, -0.0123119, 0.0122434
6: -0.0040136, 0.0037678, -0.0041817, 0.0039406, -0.0079542, 0.0079496
7: -0.0115893, -0.0017435, -0.0116780, -0.0016566, -0.0099327, 0.0099345
8: -0.0109523, 0.0167707, -0.0110738, 0.0170602, -0.0277945, 0.0276644
9: -0.0098286, 0.0060882, -0.0099894, 0.0061594, -0.0159880, 0.0160775

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_B2_A1_A2_B1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122701, upper bound: 0.0122552
time: 2.59 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B2

### Relational analysis result of NS_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123598, upper bound: 0.0122839
time: 2.88 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0056266, 0.0036371, -0.0040592, 0.0028335, -0.0084601, 0.0076963
1: 0.9886786, 1.0081874, 0.9893169, 1.0050751, -0.0163965, 0.0188705
2: -0.0144112, 0.0046891, -0.0133013, 0.0030659, -0.0167842, 0.0172866
3: 0.0001392, 0.0060395, 0.0005780, 0.0057748, -0.0056356, 0.0054615
4: -0.0055516, 0.0098069, -0.0034886, 0.0089296, -0.0144812, 0.0132955
5: -0.0017829, 0.0112554, -0.0010690, 0.0106352, -0.0124181, 0.0123244
6: -0.0069412, 0.0040158, -0.0037546, 0.0031455, -0.0100867, 0.0077704
7: -0.0117166, -0.0006721, -0.0112700, -0.0018090, -0.0099076, 0.0105979
8: -0.0124498, 0.0171862, -0.0108608, 0.0157278, -0.0279697, 0.0278398
9: -0.0100594, 0.0069667, -0.0092495, 0.0060345, -0.0160938, 0.0162162

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123181, upper bound: 0.0120554
time: 2.50 seconds

## Relational analysis of NS_A2_B1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123459, upper bound: 0.0121349
time: 4.78 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0057187, 0.0037577, -0.0041496, 0.0030145, -0.0087332, 0.0079073
1: 0.9885953, 1.0083705, 0.9888607, 1.0052546, -0.0166593, 0.0195098
2: -0.0145566, 0.0047846, -0.0140949, 0.0031595, -0.0170217, 0.0183217
3: 0.0001134, 0.0060741, 0.0005527, 0.0059641, -0.0058506, 0.0055214
4: -0.0056728, 0.0099217, -0.0036075, 0.0095569, -0.0152298, 0.0135292
5: -0.0018248, 0.0113366, -0.0011102, 0.0110787, -0.0129035, 0.0124467
6: -0.0071285, 0.0041298, -0.0040136, 0.0037678, -0.0108963, 0.0081433
7: -0.0117751, -0.0006053, -0.0115893, -0.0017435, -0.0100316, 0.0109841
8: -0.0125432, 0.0173771, -0.0109523, 0.0167707, -0.0291458, 0.0281231
9: -0.0101654, 0.0070215, -0.0098286, 0.0060882, -0.0162536, 0.0168501

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123248, upper bound: 0.0121418
time: 2.27 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123516, upper bound: 0.0122200
time: 2.31 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0040592, 0.0028335, -0.0045740, 0.0028268, -0.0068860, 0.0074076
1: 0.9893169, 1.0050751, 0.9893340, 1.0060974, -0.0167805, 0.0157411
2: -0.0133013, 0.0030659, -0.0132715, 0.0035991, -0.0162719, 0.0157136
3: 0.0005780, 0.0057748, 0.0004339, 0.0057677, -0.0051897, 0.0053409
4: -0.0034886, 0.0089296, -0.0041662, 0.0089061, -0.0123947, 0.0130958
5: -0.0010690, 0.0106352, -0.0013035, 0.0106186, -0.0116876, 0.0119387
6: -0.0037546, 0.0031455, -0.0048012, 0.0031222, -0.0068768, 0.0079467
7: -0.0112700, -0.0018090, -0.0112581, -0.0014356, -0.0098344, 0.0094491
8: -0.0108608, 0.0157278, -0.0113827, 0.0156888, -0.0263615, 0.0269241
9: -0.0092495, 0.0060345, -0.0092278, 0.0063406, -0.0155901, 0.0152623

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121384, upper bound: 0.0122333
time: 2.95 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122274, upper bound: 0.0122630
time: 2.57 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0041496, 0.0030145, -0.0046416, 0.0028598, -0.0070094, 0.0076561
1: 0.9888607, 1.0052546, 0.9892505, 1.0062318, -0.0173711, 0.0160041
2: -0.0140949, 0.0031595, -0.0134167, 0.0036691, -0.0171133, 0.0159478
3: 0.0005527, 0.0059641, 0.0004149, 0.0058023, -0.0052496, 0.0055491
4: -0.0036075, 0.0095569, -0.0042552, 0.0090208, -0.0126283, 0.0138121
5: -0.0011102, 0.0110787, -0.0013343, 0.0106997, -0.0118098, 0.0124130
6: -0.0040136, 0.0037678, -0.0049387, 0.0032360, -0.0072495, 0.0087065
7: -0.0115893, -0.0017435, -0.0113164, -0.0013865, -0.0102028, 0.0095729
8: -0.0109523, 0.0167707, -0.0114512, 0.0158794, -0.0266440, 0.0280312
9: -0.0098286, 0.0060882, -0.0093337, 0.0063809, -0.0162095, 0.0154219

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122263, upper bound: 0.0122403
time: 3.33 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122991, upper bound: 0.0122678
time: 2.41 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0056266, 0.0036371, -0.0044310, 0.0026286, -0.0082551, 0.0080681
1: 0.9886786, 1.0081874, 0.9898338, 1.0058135, -0.0171348, 0.0183536
2: -0.0144112, 0.0046891, -0.0124021, 0.0034509, -0.0171655, 0.0163924
3: 0.0001392, 0.0060395, 0.0004739, 0.0055604, -0.0054212, 0.0055656
4: -0.0055516, 0.0098069, -0.0039779, 0.0082190, -0.0137705, 0.0137848
5: -0.0017829, 0.0112554, -0.0012384, 0.0101328, -0.0119157, 0.0124937
6: -0.0069412, 0.0040158, -0.0045104, 0.0024405, -0.0093816, 0.0085262
7: -0.0117166, -0.0006721, -0.0109083, -0.0015394, -0.0101772, 0.0102362
8: -0.0124498, 0.0171862, -0.0112377, 0.0145465, -0.0267900, 0.0282154
9: -0.0100594, 0.0069667, -0.0085935, 0.0062556, -0.0163149, 0.0155601

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122236, upper bound: 0.0121012
time: 2.79 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123132, upper bound: 0.0121352
time: 2.96 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0057187, 0.0037577, -0.0045305, 0.0028078, -0.0085265, 0.0082882
1: 0.9885953, 1.0083705, 0.9893818, 1.0060110, -0.0174157, 0.0189887
2: -0.0145566, 0.0047846, -0.0131884, 0.0035540, -0.0174114, 0.0174184
3: 0.0001134, 0.0060741, 0.0004461, 0.0057479, -0.0056345, 0.0056280
4: -0.0056728, 0.0099217, -0.0041089, 0.0088404, -0.0145133, 0.0140306
5: -0.0018248, 0.0113366, -0.0012837, 0.0105721, -0.0123970, 0.0126202
6: -0.0071285, 0.0041298, -0.0047127, 0.0030570, -0.0101855, 0.0088424
7: -0.0117751, -0.0006053, -0.0112246, -0.0014672, -0.0103079, 0.0106193
8: -0.0125432, 0.0173771, -0.0113385, 0.0155796, -0.0279528, 0.0285108
9: -0.0101654, 0.0070215, -0.0091672, 0.0063147, -0.0164801, 0.0161886

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122982, upper bound: 0.0121421
time: 2.42 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123190, upper bound: 0.0122203
time: 2.83 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0040592, 0.0028335, -0.0056295, 0.0036410, -0.0077002, 0.0084631
1: 0.9893169, 1.0050751, 0.9886786, 1.0081931, -0.0188762, 0.0163965
2: -0.0133013, 0.0030659, -0.0144112, 0.0046922, -0.0172897, 0.0167842
3: 0.0005780, 0.0057748, 0.0001384, 0.0060395, -0.0054615, 0.0056364
4: -0.0034886, 0.0089296, -0.0055555, 0.0098069, -0.0132955, 0.0144851
5: -0.0010690, 0.0106352, -0.0017842, 0.0112554, -0.0123244, 0.0124194
6: -0.0037546, 0.0031455, -0.0069472, 0.0040158, -0.0077704, 0.0100927
7: -0.0112700, -0.0018090, -0.0117166, -0.0006699, -0.0106001, 0.0099076
8: -0.0108608, 0.0157278, -0.0124528, 0.0171862, -0.0278398, 0.0279727
9: -0.0092495, 0.0060345, -0.0100594, 0.0069684, -0.0162179, 0.0160938

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121350, upper bound: 0.0122341
time: 2.81 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122244, upper bound: 0.0122627
time: 2.52 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0041496, 0.0030145, -0.0057217, 0.0037616, -0.0079112, 0.0087362
1: 0.9888607, 1.0052546, 0.9885953, 1.0083764, -0.0195156, 0.0166593
2: -0.0140949, 0.0031595, -0.0145566, 0.0047877, -0.0183248, 0.0170217
3: 0.0005527, 0.0059641, 0.0001126, 0.0060741, -0.0055214, 0.0058515
4: -0.0036075, 0.0095569, -0.0056768, 0.0099217, -0.0135292, 0.0152337
5: -0.0011102, 0.0110787, -0.0018262, 0.0113366, -0.0124467, 0.0129049
6: -0.0040136, 0.0037678, -0.0071345, 0.0041298, -0.0081433, 0.0109024
7: -0.0115893, -0.0017435, -0.0117750, -0.0006031, -0.0109862, 0.0100315
8: -0.0109523, 0.0167707, -0.0125462, 0.0173771, -0.0281231, 0.0291488
9: -0.0098286, 0.0060882, -0.0101654, 0.0070232, -0.0168518, 0.0162536

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122231, upper bound: 0.0122409
time: 2.65 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122980, upper bound: 0.0122670
time: 2.75 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0055040, 0.0034767, -0.0056295, 0.0036410, -0.0091450, 0.0091062
1: 0.9891715, 1.0079441, 0.9886786, 1.0081931, -0.0190216, 0.0192655
2: -0.0135542, 0.0045622, -0.0144112, 0.0046922, -0.0174709, 0.0182102
3: 0.0001735, 0.0058351, 0.0001384, 0.0060395, -0.0058660, 0.0056968
4: -0.0053903, 0.0091295, -0.0055555, 0.0098069, -0.0151971, 0.0146850
5: -0.0017271, 0.0107766, -0.0017842, 0.0112554, -0.0129824, 0.0125608
6: -0.0066920, 0.0033438, -0.0069472, 0.0040158, -0.0107078, 0.0102910
7: -0.0113718, -0.0007610, -0.0117166, -0.0006699, -0.0107019, 0.0109556
8: -0.0123255, 0.0160602, -0.0124528, 0.0171862, -0.0292882, 0.0282882
9: -0.0094341, 0.0068938, -0.0100594, 0.0069684, -0.0164025, 0.0169531

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_B2_A2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121963, upper bound: 0.0121236
time: 2.85 seconds

## Relational analysis of NS_A2_B2_B2_A2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122422, upper bound: 0.0122134
time: 2.27 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0055719, 0.0035655, -0.0057217, 0.0037616, -0.0093335, 0.0092872
1: 0.9887300, 1.0080789, 0.9885953, 1.0083764, -0.0196464, 0.0194836
2: -0.0143222, 0.0046325, -0.0145566, 0.0047877, -0.0185001, 0.0184192
3: 0.0001545, 0.0060182, 0.0001126, 0.0060741, -0.0059196, 0.0059057
4: -0.0054797, 0.0097365, -0.0056768, 0.0099217, -0.0154014, 0.0154132
5: -0.0017580, 0.0112056, -0.0018262, 0.0113366, -0.0130946, 0.0130318
6: -0.0068301, 0.0039460, -0.0071345, 0.0041298, -0.0109598, 0.0110805
7: -0.0116807, -0.0007117, -0.0117750, -0.0006031, -0.0110776, 0.0110633
8: -0.0123944, 0.0170692, -0.0125462, 0.0173771, -0.0295469, 0.0294326
9: -0.0099944, 0.0069342, -0.0101654, 0.0070232, -0.0170176, 0.0170995

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122410, upper bound: 0.0121979
time: 3.26 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123185, upper bound: 0.0122195
time: 3.16 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 8.15 seconds
NS_A1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122029, upper bound: 0.0122240
NS_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122383, upper bound: 0.0123202
NS_A1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123026, upper bound: 0.0122418
NS_A1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123265, upper bound: 0.0123264
NS_A1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0121511, upper bound: 0.0122611
NS_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0121860, upper bound: 0.0123621
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0121910, upper bound: 0.0123363
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122845, upper bound: 0.0123674
NS_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122812, upper bound: 0.0120886
NS_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123048, upper bound: 0.0121692
NS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122892, upper bound: 0.0121758
NS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123096, upper bound: 0.0122475
NS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122341, upper bound: 0.0121384
NS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122628, upper bound: 0.0122271
NS_A1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122410, upper bound: 0.0122266
NS_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122676, upper bound: 0.0122994
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0120890, upper bound: 0.0122803
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0121692, upper bound: 0.0123047
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0121759, upper bound: 0.0122890
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122472, upper bound: 0.0123096
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0120559, upper bound: 0.0123224
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0121351, upper bound: 0.0123523
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0121427, upper bound: 0.0123286
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122201, upper bound: 0.0123564
NS_A1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0121518, upper bound: 0.0121574
NS_A1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0121851, upper bound: 0.0122386
NS_A1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122418, upper bound: 0.0121759
NS_A1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122604, upper bound: 0.0122471
NS_A1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0121231, upper bound: 0.0122065
NS_A1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0121564, upper bound: 0.0122917
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0121524, upper bound: 0.0122723
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122320, upper bound: 0.0122985
NS_A2_B1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122612, upper bound: 0.0121506
NS_A2_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123625, upper bound: 0.0121858
NS_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123363, upper bound: 0.0121909
NS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123675, upper bound: 0.0122845
NS_A2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123225, upper bound: 0.0120554
NS_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123520, upper bound: 0.0121351
NS_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123290, upper bound: 0.0121419
NS_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123565, upper bound: 0.0122198
NS_A2_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122380, upper bound: 0.0121741
NS_A2_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122839, upper bound: 0.0122783
NS_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122701, upper bound: 0.0122552
NS_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123598, upper bound: 0.0122839
NS_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123181, upper bound: 0.0120554
NS_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123459, upper bound: 0.0121349
NS_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123248, upper bound: 0.0121418
NS_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123516, upper bound: 0.0122200
NS_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0121384, upper bound: 0.0122333
NS_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122274, upper bound: 0.0122630
NS_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122263, upper bound: 0.0122403
NS_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122991, upper bound: 0.0122678
NS_A2_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122236, upper bound: 0.0121012
NS_A2_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123132, upper bound: 0.0121352
NS_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122982, upper bound: 0.0121421
NS_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123190, upper bound: 0.0122203
NS_A2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0121350, upper bound: 0.0122341
NS_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122244, upper bound: 0.0122627
NS_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122231, upper bound: 0.0122409
NS_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122980, upper bound: 0.0122670
NS_A2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0121963, upper bound: 0.0121236
NS_A2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122422, upper bound: 0.0122134
NS_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0122410, upper bound: 0.0121979
NS_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.15
Output dim: 1, lower bound: -0.0123185, upper bound: 0.0122195

## BFS NS instance: NS_A1_B1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0023486, 0.0025797, -0.0030799, 0.0027710, -0.0051196, 0.0056596
1: 0.9899571, 1.0016785, 0.9894745, 1.0031308, -0.0131737, 0.0122040
2: -0.0121878, 0.0012943, -0.0130268, 0.0020517, -0.0135240, 0.0136164
3: 0.0010569, 0.0055093, 0.0008522, 0.0057094, -0.0046525, 0.0046572
4: -0.0013550, 0.0080496, -0.0021997, 0.0087127, -0.0100677, 0.0102492
5: -0.0002899, 0.0100131, -0.0006230, 0.0104819, -0.0107718, 0.0106361
6: -0.0033516, 0.0022724, -0.0036428, 0.0029303, -0.0062819, 0.0059152
7: -0.0108221, -0.0030499, -0.0111596, -0.0025194, -0.0083027, 0.0081097
8: -0.0091264, 0.0142649, -0.0098679, 0.0153673, -0.0242908, 0.0239274
9: -0.0084371, 0.0050170, -0.0090493, 0.0054520, -0.0138891, 0.0140663

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 162

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122029, upper bound: 0.0121571
time: 2.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122029, upper bound: 0.0122236
time: 2.73 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0026653, 0.0025669, -0.0031722, 0.0027732, -0.0054385, 0.0057391
1: 0.9899893, 1.0023073, 0.9894690, 1.0033140, -0.0133247, 0.0128383
2: -0.0121315, 0.0016223, -0.0130364, 0.0021473, -0.0135658, 0.0139353
3: 0.0009682, 0.0054959, 0.0008263, 0.0057117, -0.0047434, 0.0046696
4: -0.0016539, 0.0080051, -0.0023212, 0.0087202, -0.0103742, 0.0103263
5: -0.0004342, 0.0099816, -0.0006650, 0.0104872, -0.0109213, 0.0106466
6: -0.0033320, 0.0022283, -0.0036461, 0.0029378, -0.0062698, 0.0058744
7: -0.0107994, -0.0028201, -0.0111635, -0.0024524, -0.0083470, 0.0083433
8: -0.0094476, 0.0141910, -0.0099615, 0.0153798, -0.0246201, 0.0239482
9: -0.0083961, 0.0052054, -0.0090562, 0.0055069, -0.0139030, 0.0142616

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121570, upper bound: 0.0122952
time: 3.08 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121573, upper bound: 0.0123199
time: 2.78 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0024652, 0.0027612, -0.0031497, 0.0028048, -0.0052700, 0.0059109
1: 0.9894993, 1.0019102, 0.9893895, 1.0032692, -0.0137699, 0.0125207
2: -0.0129838, 0.0014150, -0.0131752, 0.0021240, -0.0143725, 0.0138889
3: 0.0010242, 0.0056991, 0.0008326, 0.0057448, -0.0047205, 0.0048665
4: -0.0021492, 0.0086787, -0.0023401, 0.0088299, -0.0109792, 0.0110189
5: -0.0003430, 0.0104578, -0.0006548, 0.0105647, -0.0109077, 0.0111126
6: -0.0036279, 0.0028966, -0.0036943, 0.0030466, -0.0066745, 0.0065909
7: -0.0111423, -0.0029653, -0.0112193, -0.0024688, -0.0086736, 0.0082540
8: -0.0092447, 0.0153108, -0.0099387, 0.0155622, -0.0246043, 0.0250402
9: -0.0090179, 0.0050864, -0.0091575, 0.0054935, -0.0145114, 0.0142439

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122949, upper bound: 0.0121570
time: 2.50 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122953, upper bound: 0.0121959
time: 2.90 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0027757, 0.0027519, -0.0032417, 0.0028070, -0.0055827, 0.0059936
1: 0.9895227, 1.0025266, 0.9893840, 1.0034519, -0.0139292, 0.0131426
2: -0.0129431, 0.0017367, -0.0131847, 0.0022193, -0.0144267, 0.0142006
3: 0.0009373, 0.0056894, 0.0008069, 0.0057470, -0.0048097, 0.0048825
4: -0.0021086, 0.0086465, -0.0024127, 0.0088374, -0.0109460, 0.0110592
5: -0.0004844, 0.0104351, -0.0006967, 0.0105700, -0.0110545, 0.0111318
6: -0.0036138, 0.0028646, -0.0036976, 0.0030541, -0.0066678, 0.0065622
7: -0.0111259, -0.0027400, -0.0112231, -0.0024020, -0.0087239, 0.0084831
8: -0.0095595, 0.0152572, -0.0100320, 0.0155746, -0.0249295, 0.0250811
9: -0.0089882, 0.0052711, -0.0091644, 0.0055483, -0.0145364, 0.0144355

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123205, upper bound: 0.0122378
time: 2.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123208, upper bound: 0.0122724
time: 3.50 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0023486, 0.0025797, -0.0041012, 0.0030286, -0.0053771, 0.0066809
1: 0.9899571, 1.0016785, 0.9888251, 1.0051587, -0.0152016, 0.0128534
2: -0.0121878, 0.0012943, -0.0141568, 0.0031094, -0.0146144, 0.0147670
3: 0.0010569, 0.0055093, 0.0005662, 0.0059788, -0.0049219, 0.0049431
4: -0.0013550, 0.0080496, -0.0035439, 0.0096057, -0.0109608, 0.0115934
5: -0.0002899, 0.0100131, -0.0010881, 0.0111132, -0.0114031, 0.0111012
6: -0.0033516, 0.0022724, -0.0040350, 0.0038163, -0.0071679, 0.0063074
7: -0.0108221, -0.0030499, -0.0116142, -0.0017786, -0.0090435, 0.0085643
8: -0.0091264, 0.0142649, -0.0109033, 0.0168518, -0.0257796, 0.0249706
9: -0.0084371, 0.0050170, -0.0098737, 0.0060594, -0.0144965, 0.0148907

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 162

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121511, upper bound: 0.0121968
time: 4.15 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121511, upper bound: 0.0122611
time: 3.07 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0026653, 0.0025669, -0.0041942, 0.0030308, -0.0056961, 0.0067610
1: 0.9899893, 1.0023073, 0.9888196, 1.0053432, -0.0153539, 0.0134877
2: -0.0121315, 0.0016223, -0.0141664, 0.0032057, -0.0146563, 0.0150904
3: 0.0009682, 0.0054959, 0.0005402, 0.0059811, -0.0050129, 0.0049557
4: -0.0016539, 0.0080051, -0.0036662, 0.0096134, -0.0112673, 0.0116714
5: -0.0004342, 0.0099816, -0.0011305, 0.0111186, -0.0115527, 0.0111121
6: -0.0033320, 0.0022283, -0.0040384, 0.0038239, -0.0071559, 0.0062667
7: -0.0107994, -0.0028201, -0.0116181, -0.0017111, -0.0090883, 0.0087980
8: -0.0094476, 0.0141910, -0.0109976, 0.0168645, -0.0261097, 0.0249911
9: -0.0083961, 0.0052054, -0.0098807, 0.0061147, -0.0145108, 0.0150862

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121053, upper bound: 0.0123278
time: 2.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121053, upper bound: 0.0123616
time: 3.34 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0030382, 0.0027562, -0.0036235, 0.0030706, -0.0061089, 0.0063797
1: 0.9895118, 1.0030481, 0.9887191, 1.0042102, -0.0146984, 0.0143290
2: -0.0129620, 0.0020085, -0.0143412, 0.0026147, -0.0150359, 0.0156613
3: 0.0008638, 0.0056939, 0.0007000, 0.0060228, -0.0051590, 0.0049940
4: -0.0021448, 0.0086615, -0.0035035, 0.0097515, -0.0118963, 0.0121650
5: -0.0006040, 0.0104456, -0.0008706, 0.0112162, -0.0118202, 0.0113162
6: -0.0036203, 0.0028795, -0.0040990, 0.0039609, -0.0075813, 0.0069785
7: -0.0111335, -0.0025496, -0.0116884, -0.0021251, -0.0090085, 0.0091388
8: -0.0098257, 0.0152821, -0.0104190, 0.0170942, -0.0267205, 0.0255404
9: -0.0090020, 0.0054272, -0.0100083, 0.0057753, -0.0147773, 0.0154355

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121747, upper bound: 0.0122461
time: 2.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121743, upper bound: 0.0122767
time: 2.96 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0031307, 0.0027583, -0.0039118, 0.0030581, -0.0061888, 0.0066702
1: 0.9895065, 1.0032316, 0.9887508, 1.0047826, -0.0152761, 0.0144808
2: -0.0129715, 0.0021043, -0.0142860, 0.0029133, -0.0153264, 0.0157034
3: 0.0008379, 0.0056962, 0.0006193, 0.0060096, -0.0051717, 0.0050769
4: -0.0022665, 0.0086689, -0.0034484, 0.0097079, -0.0119744, 0.0121173
5: -0.0006461, 0.0104509, -0.0010019, 0.0111854, -0.0118315, 0.0114528
6: -0.0036236, 0.0028869, -0.0040799, 0.0039177, -0.0075412, 0.0069668
7: -0.0111373, -0.0024826, -0.0116662, -0.0019159, -0.0092214, 0.0091836
8: -0.0099194, 0.0152945, -0.0107113, 0.0170217, -0.0267427, 0.0258402
9: -0.0090089, 0.0054822, -0.0099680, 0.0059468, -0.0149557, 0.0154502

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122784, upper bound: 0.0122936
time: 2.44 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122784, upper bound: 0.0123154
time: 2.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0038847, 0.0028267, -0.0029290, 0.0025710, -0.0064557, 0.0057557
1: 0.9893342, 1.0047288, 0.9899789, 1.0028311, -0.0134969, 0.0147499
2: -0.0132712, 0.0028852, -0.0121497, 0.0018954, -0.0144964, 0.0143922
3: 0.0006269, 0.0057677, 0.0008944, 0.0055003, -0.0048734, 0.0048733
4: -0.0032589, 0.0089059, -0.0020011, 0.0080195, -0.0112784, 0.0109070
5: -0.0009895, 0.0106184, -0.0005543, 0.0099918, -0.0109813, 0.0111727
6: -0.0037277, 0.0031219, -0.0033384, 0.0022426, -0.0059702, 0.0064603
7: -0.0112580, -0.0019356, -0.0108067, -0.0026288, -0.0086291, 0.0088711
8: -0.0106838, 0.0156884, -0.0097149, 0.0142149, -0.0247090, 0.0252077
9: -0.0092276, 0.0059306, -0.0084093, 0.0053623, -0.0145899, 0.0143400

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121883, upper bound: 0.0120882
time: 2.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121883, upper bound: 0.0120884
time: 2.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042364, 0.0028202, -0.0030213, 0.0025732, -0.0068096, 0.0058415
1: 0.9893506, 1.0054271, 0.9899734, 1.0030142, -0.0136636, 0.0154537
2: -0.0132428, 0.0032494, -0.0121592, 0.0019909, -0.0145668, 0.0147376
3: 0.0005284, 0.0057609, 0.0008686, 0.0055025, -0.0049741, 0.0048923
4: -0.0037219, 0.0088834, -0.0021225, 0.0080270, -0.0117488, 0.0110059
5: -0.0011497, 0.0106026, -0.0005963, 0.0099971, -0.0111468, 0.0111988
6: -0.0041149, 0.0030997, -0.0033417, 0.0022500, -0.0063648, 0.0064413
7: -0.0112465, -0.0016805, -0.0108105, -0.0025619, -0.0086846, 0.0091301
8: -0.0110404, 0.0156510, -0.0098084, 0.0142273, -0.0250700, 0.0252628
9: -0.0092069, 0.0061399, -0.0084162, 0.0054171, -0.0146240, 0.0145561

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122089, upper bound: 0.0121335
time: 2.51 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122089, upper bound: 0.0121694
time: 3.16 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0039524, 0.0028601, -0.0030382, 0.0027562, -0.0067086, 0.0058983
1: 0.9892500, 1.0048631, 0.9895118, 1.0030481, -0.0137981, 0.0153513
2: -0.0134175, 0.0029553, -0.0129620, 0.0020085, -0.0147598, 0.0152556
3: 0.0006079, 0.0058025, 0.0008638, 0.0056939, -0.0050860, 0.0049387
4: -0.0033480, 0.0090215, -0.0021448, 0.0086615, -0.0120095, 0.0111663
5: -0.0010204, 0.0107002, -0.0006040, 0.0104456, -0.0114660, 0.0113042
6: -0.0037784, 0.0032366, -0.0036203, 0.0028795, -0.0066579, 0.0068570
7: -0.0113168, -0.0018865, -0.0111335, -0.0025496, -0.0087672, 0.0092470
8: -0.0107525, 0.0158806, -0.0098257, 0.0152821, -0.0258409, 0.0255125
9: -0.0093343, 0.0059709, -0.0090020, 0.0054272, -0.0147615, 0.0149729

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121883, upper bound: 0.0121568
time: 2.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121883, upper bound: 0.0121305
time: 2.73 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0043041, 0.0028533, -0.0031307, 0.0027583, -0.0070624, 0.0059840
1: 0.9892672, 1.0055614, 0.9895065, 1.0032316, -0.0139644, 0.0160549
2: -0.0133879, 0.0033195, -0.0129715, 0.0021043, -0.0148300, 0.0155951
3: 0.0005095, 0.0057955, 0.0008379, 0.0056962, -0.0051867, 0.0049575
4: -0.0038109, 0.0089980, -0.0022665, 0.0086689, -0.0124798, 0.0112645
5: -0.0011806, 0.0106836, -0.0006461, 0.0104509, -0.0116315, 0.0113297
6: -0.0042524, 0.0032134, -0.0036236, 0.0028869, -0.0071392, 0.0068370
7: -0.0113049, -0.0016314, -0.0111373, -0.0024826, -0.0088223, 0.0095059
8: -0.0111090, 0.0158416, -0.0099194, 0.0152945, -0.0262005, 0.0255665
9: -0.0093127, 0.0061801, -0.0090089, 0.0054822, -0.0147949, 0.0151889

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122231, upper bound: 0.0122385
time: 2.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122231, upper bound: 0.0121986
time: 2.82 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0038847, 0.0028267, -0.0039686, 0.0028313, -0.0067160, 0.0067953
1: 0.9893342, 1.0047288, 0.9893225, 1.0048953, -0.0155611, 0.0154063
2: -0.0132712, 0.0028852, -0.0132914, 0.0029721, -0.0156156, 0.0155583
3: 0.0006269, 0.0057677, 0.0006034, 0.0057725, -0.0051456, 0.0051643
4: -0.0032589, 0.0089059, -0.0033694, 0.0089218, -0.0121807, 0.0122752
5: -0.0009895, 0.0106184, -0.0010278, 0.0106297, -0.0116192, 0.0116462
6: -0.0037277, 0.0031219, -0.0037346, 0.0031378, -0.0068654, 0.0068566
7: -0.0112580, -0.0019356, -0.0112661, -0.0018747, -0.0093832, 0.0093305
8: -0.0106838, 0.0156884, -0.0107689, 0.0157149, -0.0262129, 0.0262709
9: -0.0092276, 0.0059306, -0.0092423, 0.0059806, -0.0152082, 0.0151730

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121413, upper bound: 0.0121386
time: 3.24 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121413, upper bound: 0.0121382
time: 3.05 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042364, 0.0028202, -0.0040592, 0.0028335, -0.0070700, 0.0068794
1: 0.9893506, 1.0054271, 0.9893169, 1.0050751, -0.0157245, 0.0161102
2: -0.0132428, 0.0032494, -0.0133013, 0.0030659, -0.0156832, 0.0159083
3: 0.0005284, 0.0057609, 0.0005780, 0.0057748, -0.0052464, 0.0051829
4: -0.0037219, 0.0088834, -0.0034886, 0.0089296, -0.0126515, 0.0123720
5: -0.0011497, 0.0106026, -0.0010690, 0.0106352, -0.0117849, 0.0116716
6: -0.0041149, 0.0030997, -0.0037546, 0.0031455, -0.0072603, 0.0068543
7: -0.0112465, -0.0016805, -0.0112700, -0.0018090, -0.0094375, 0.0095896
8: -0.0110404, 0.0156510, -0.0108608, 0.0157278, -0.0265751, 0.0263239
9: -0.0092069, 0.0061399, -0.0092495, 0.0060345, -0.0152413, 0.0153894

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121612, upper bound: 0.0121827
time: 2.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121612, upper bound: 0.0122266
time: 2.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0039524, 0.0028601, -0.0040568, 0.0030123, -0.0069647, 0.0069169
1: 0.9892500, 1.0048631, 0.9888662, 1.0050704, -0.0158204, 0.0159969
2: -0.0134175, 0.0029553, -0.0140854, 0.0030635, -0.0158472, 0.0164001
3: 0.0006079, 0.0058025, 0.0005787, 0.0059618, -0.0053539, 0.0052239
4: -0.0033480, 0.0090215, -0.0034855, 0.0095493, -0.0128974, 0.0125070
5: -0.0010204, 0.0107002, -0.0010679, 0.0110733, -0.0120937, 0.0117681
6: -0.0037784, 0.0032366, -0.0040103, 0.0037603, -0.0075387, 0.0072469
7: -0.0113168, -0.0018865, -0.0115855, -0.0018107, -0.0095060, 0.0096990
8: -0.0107525, 0.0158806, -0.0108584, 0.0167581, -0.0273200, 0.0265529
9: -0.0093343, 0.0059709, -0.0098216, 0.0060330, -0.0153674, 0.0157926

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121413, upper bound: 0.0122065
time: 3.04 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121413, upper bound: 0.0121806
time: 2.82 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0043041, 0.0028533, -0.0041496, 0.0030145, -0.0073186, 0.0070029
1: 0.9892672, 1.0055614, 0.9888607, 1.0052546, -0.0159874, 0.0167006
2: -0.0133879, 0.0033195, -0.0140949, 0.0031595, -0.0159171, 0.0167432
3: 0.0005095, 0.0057955, 0.0005527, 0.0059641, -0.0054546, 0.0052427
4: -0.0038109, 0.0089980, -0.0036075, 0.0095569, -0.0133678, 0.0126056
5: -0.0011806, 0.0106836, -0.0011102, 0.0110787, -0.0122592, 0.0117937
6: -0.0042524, 0.0032134, -0.0040136, 0.0037678, -0.0080202, 0.0072270
7: -0.0113049, -0.0016314, -0.0115893, -0.0017435, -0.0095614, 0.0099579
8: -0.0111090, 0.0158416, -0.0109523, 0.0167707, -0.0276799, 0.0266063
9: -0.0093127, 0.0061801, -0.0098286, 0.0060882, -0.0154009, 0.0160087

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121741, upper bound: 0.0122917
time: 3.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121741, upper bound: 0.0122498
time: 2.97 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0029290, 0.0025710, -0.0038847, 0.0028267, -0.0057557, 0.0064557
1: 0.9899789, 1.0028311, 0.9893342, 1.0047288, -0.0147499, 0.0134969
2: -0.0121497, 0.0018954, -0.0132712, 0.0028852, -0.0143922, 0.0144964
3: 0.0008944, 0.0055003, 0.0006269, 0.0057677, -0.0048733, 0.0048734
4: -0.0020011, 0.0080195, -0.0032589, 0.0089059, -0.0109070, 0.0112784
5: -0.0005543, 0.0099918, -0.0009895, 0.0106184, -0.0111727, 0.0109813
6: -0.0033384, 0.0022426, -0.0037277, 0.0031219, -0.0064603, 0.0059702
7: -0.0108067, -0.0026288, -0.0112580, -0.0019356, -0.0088711, 0.0086291
8: -0.0097149, 0.0142149, -0.0106838, 0.0156884, -0.0252077, 0.0247090
9: -0.0084093, 0.0053623, -0.0092276, 0.0059306, -0.0143400, 0.0145899

Time for backsubstitution: 1.71 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.71 + 595.96 = 600.68 seconds
